// Gated DeltaNet's recurrent step, fused.
//
// One decode token updates a [Dk, Dv] memory per head:
//
//     rec   = rec * exp(g)
//     delta = (v - rec^T k) * beta
//     rec  += k delta^T
//     out   = rec^T q
//
// In PyTorch that is a dozen elementwise kernels over the state, so a 36-layer
// Qwen3-Next reads its 2 MB-per-layer memory about seven times per token. The
// fused version reads it twice and writes it once — the second read is what the
// data dependency costs, because `delta` needs every row before any row can be
// updated — and out falls out of the same pass:
//
//     out = exp(g) * (rec^T q) + delta * (k . q)
//
// so the q projection is accumulated alongside the k one instead of needing
// another sweep over the updated state.

#include "common.cuh"

namespace infer {
namespace {

__global__ void gdn_decode_kernel(
    const float* __restrict__ q,       // [B, H, Dk]
    const float* __restrict__ k,       // [B, H, Dk]
    const float* __restrict__ v,       // [B, H, Dv]
    const float* __restrict__ g_log,   // [B, H]
    const float* __restrict__ beta,    // [B, H]
    float* __restrict__ state,         // [B, H, Dk, Dv]
    float* __restrict__ out,           // [B, H, Dv]
    int k_dim,
    int v_dim) {
  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const int heads = gridDim.x;
  const int col = threadIdx.x;

  extern __shared__ float sh[];
  float* sq = sh;              // [k_dim]
  float* sk = sh + k_dim;      // [k_dim]

  const int64_t head_id = static_cast<int64_t>(batch) * heads + head;
  const float* qp = q + head_id * k_dim;
  const float* kp = k + head_id * k_dim;
  for (int i = threadIdx.x; i < k_dim; i += blockDim.x) {
    sq[i] = qp[i];
    sk[i] = kp[i];
  }
  __syncthreads();
  if (col >= v_dim) {
    return;
  }

  const float decay = expf(g_log[head_id]);
  const float b = beta[head_id];
  float* st = state + head_id * k_dim * v_dim;

  float acc_k = 0.0f;
  float acc_q = 0.0f;
  for (int i = 0; i < k_dim; ++i) {
    const float s = st[static_cast<int64_t>(i) * v_dim + col];
    acc_k = fmaf(s, sk[i], acc_k);
    acc_q = fmaf(s, sq[i], acc_q);
  }
  const float delta = (v[head_id * v_dim + col] - decay * acc_k) * b;

  float qk = 0.0f;
  for (int i = 0; i < k_dim; ++i) {
    qk = fmaf(sq[i], sk[i], qk);
  }
  out[head_id * v_dim + col] = decay * acc_q + delta * qk;

  for (int i = 0; i < k_dim; ++i) {
    const int64_t idx = static_cast<int64_t>(i) * v_dim + col;
    st[idx] = fmaf(decay, st[idx], sk[i] * delta);
  }
}

}  // namespace

// q/k: [B, H, Dk] already l2-normalized and scaled, v: [B, H, Dv],
// g_log/beta: [B, H], state: [B, H, Dk, Dv] updated in place. All fp32.
at::Tensor gdn_decode_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& g_log,
    const at::Tensor& beta,
    at::Tensor state) {
  TORCH_CHECK(q.is_cuda() && state.is_cuda(), "gdn_decode expects CUDA tensors");
  TORCH_CHECK(q.scalar_type() == at::kFloat && state.scalar_type() == at::kFloat,
              "gdn_decode is fp32");
  TORCH_CHECK(q.dim() == 3 && state.dim() == 4, "gdn_decode shape");
  const int64_t batch = q.size(0);
  const int64_t heads = q.size(1);
  const int64_t k_dim = q.size(2);
  const int64_t v_dim = v.size(2);
  TORCH_CHECK(k.sizes() == q.sizes(), "gdn_decode q/k mismatch");
  TORCH_CHECK(state.size(0) == batch && state.size(1) == heads, "gdn_decode state batch");
  TORCH_CHECK(state.size(2) == k_dim && state.size(3) == v_dim, "gdn_decode state dims");
  TORCH_CHECK(v_dim <= 1024, "gdn_decode v_dim > 1024");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "gdn_decode needs contiguous q/k/v");
  TORCH_CHECK(state.is_contiguous(), "gdn_decode needs a contiguous state");
  TORCH_CHECK(g_log.is_contiguous() && beta.is_contiguous(), "gdn_decode gates");

  const at::cuda::CUDAGuard guard(q.device());
  auto out = at::empty({batch, heads, v_dim}, q.options());
  const int threads = static_cast<int>(v_dim);
  const size_t smem = sizeof(float) * 2 * k_dim;
  gdn_decode_kernel<<<dim3(heads, batch), threads, smem, at::cuda::getCurrentCUDAStream()>>>(
      q.data_ptr<float>(),
      k.data_ptr<float>(),
      v.data_ptr<float>(),
      g_log.data_ptr<float>(),
      beta.data_ptr<float>(),
      state.data_ptr<float>(),
      out.data_ptr<float>(),
      static_cast<int>(k_dim),
      static_cast<int>(v_dim));
  return out;
}

}  // namespace infer
