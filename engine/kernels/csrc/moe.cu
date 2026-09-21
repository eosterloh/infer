// Grouped GEMV for MoE decode.
//
// The Python dispatch loop asks "which tokens went to expert i?" once per
// expert, and each answer is a device-to-host sync. At 128 experts and 26 MoE
// layers that is thousands of syncs per token. This kernel takes the routing
// table on the device instead: output row r reads expert `row_expert[r]` and
// activation row `row_input[r]`, so the whole layer is one launch with no
// synchronization and no gather of the expert weights.
#include "common.cuh"

namespace infer {
namespace {

template <typename scalar_t>
__global__ void moe_gemv_kernel(
    const scalar_t* __restrict__ x,        // [T, K]
    const scalar_t* __restrict__ w,        // [E, N, K]
    const int32_t* __restrict__ row_expert,  // [M]
    const int32_t* __restrict__ row_input,   // [M] or null for identity
    const scalar_t* __restrict__ bias,     // [E, N] or null
    scalar_t* __restrict__ out,            // [M, N]
    int m_rows,
    int n_cols,
    int k_dim) {
  const int warps = blockDim.x >> 5;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int col = blockIdx.x * warps + (threadIdx.x >> 5);
  const int row = blockIdx.y;
  if (col >= n_cols || row >= m_rows) {
    return;
  }

  const int expert = row_expert[row];
  const int token = row_input ? row_input[row] : row;
  const scalar_t* xr = x + static_cast<int64_t>(token) * k_dim;
  const scalar_t* wr =
      w + (static_cast<int64_t>(expert) * n_cols + col) * k_dim;

  float acc = 0.0f;
  constexpr int kVec = 16 / sizeof(scalar_t);
  const int64_t vectors = k_dim / kVec;
  if (k_dim % kVec == 0) {
    const auto* xv = reinterpret_cast<const Vec<scalar_t, kVec>*>(xr);
    const auto* wv = reinterpret_cast<const Vec<scalar_t, kVec>*>(wr);
    for (int64_t i = lane; i < vectors; i += kWarpSize) {
      const Vec<scalar_t, kVec> a = xv[i];
      const Vec<scalar_t, kVec> b = wv[i];
#pragma unroll
      for (int j = 0; j < kVec; ++j) {
        acc += static_cast<float>(a.v[j]) * static_cast<float>(b.v[j]);
      }
    }
  } else {
    for (int i = lane; i < k_dim; i += kWarpSize) {
      acc += static_cast<float>(xr[i]) * static_cast<float>(wr[i]);
    }
  }

  acc = warp_reduce_sum(acc);
  if (lane == 0) {
    if (bias) {
      acc += static_cast<float>(bias[static_cast<int64_t>(expert) * n_cols + col]);
    }
    out[static_cast<int64_t>(row) * n_cols + col] = static_cast<scalar_t>(acc);
  }
}

// Weighted reduction of the per-expert results back onto the token:
// out[t, n] = sum_k weight[t, k] * expert_out[t * topk + k, n]
// The router computes its weights in fp32 on purpose, so they stay fp32 here:
// rounding them to the activation dtype first would throw away seven mantissa
// bits of exactly the quantity the router worked in fp32 to get right, and both
// references (ops.cpp and the Python fallback) keep the full precision.
template <typename scalar_t>
__global__ void moe_combine_kernel(
    const scalar_t* __restrict__ expert_out,  // [T * topk, N]
    const float* __restrict__ weights,        // [T, topk], fp32
    scalar_t* __restrict__ out,               // [T, N]
    int tokens,
    int topk,
    int n_cols) {
  const int64_t total = static_cast<int64_t>(tokens) * n_cols;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       idx < total; idx += stride) {
    const int64_t t = idx / n_cols;
    const int64_t n = idx - t * n_cols;
    float acc = 0.0f;
    for (int k = 0; k < topk; ++k) {
      const float w = weights[t * topk + k];
      acc += w * static_cast<float>(expert_out[(t * topk + k) * n_cols + n]);
    }
    out[idx] = static_cast<scalar_t>(acc);
  }
}

}  // namespace

at::Tensor moe_gemv_cuda(
    const at::Tensor& x,
    const at::Tensor& w,
    const at::Tensor& row_expert,
    const c10::optional<at::Tensor>& row_input,
    const c10::optional<at::Tensor>& bias) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "moe_gemv expects CUDA tensors");
  TORCH_CHECK(w.dim() == 3, "expert weight must be [E, N, K]");
  TORCH_CHECK(x.dim() == 2, "activations must be [T, K]");
  TORCH_CHECK(x.scalar_type() == w.scalar_type(), "dtype mismatch");
  TORCH_CHECK(row_expert.scalar_type() == at::kInt, "row_expert must be int32");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous(), "moe_gemv needs contiguous inputs");

  const at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  const int64_t m_rows = row_expert.numel();
  const int64_t n_cols = w.size(1);
  const int64_t k_dim = w.size(2);
  TORCH_CHECK(x.size(1) == k_dim, "activation width does not match expert K");

  auto out = at::empty({m_rows, n_cols}, x.options());
  const int threads = 256;
  const int warps = threads / kWarpSize;
  const dim3 blocks(
      static_cast<unsigned>((n_cols + warps - 1) / warps),
      static_cast<unsigned>(m_rows));
  auto stream = at::cuda::getCurrentCUDAStream();

  const int32_t* input_ptr =
      (row_input.has_value() && row_input->defined()) ? row_input->data_ptr<int32_t>()
                                                     : nullptr;

  if (x.scalar_type() == at::kBFloat16) {
    using scalar_t = at::BFloat16;
    moe_gemv_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        x.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), row_expert.data_ptr<int32_t>(),
        input_ptr,
        (bias.has_value() && bias->defined()) ? bias->data_ptr<scalar_t>() : nullptr,
        out.data_ptr<scalar_t>(), static_cast<int>(m_rows), static_cast<int>(n_cols),
        static_cast<int>(k_dim));
  } else if (x.scalar_type() == at::kHalf) {
    using scalar_t = at::Half;
    moe_gemv_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        x.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), row_expert.data_ptr<int32_t>(),
        input_ptr,
        (bias.has_value() && bias->defined()) ? bias->data_ptr<scalar_t>() : nullptr,
        out.data_ptr<scalar_t>(), static_cast<int>(m_rows), static_cast<int>(n_cols),
        static_cast<int>(k_dim));
  } else {
    TORCH_CHECK(false, "moe_gemv supports bf16 and fp16");
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

at::Tensor moe_combine_cuda(
    const at::Tensor& expert_out, const at::Tensor& weights, int64_t topk) {
  TORCH_CHECK(expert_out.is_cuda() && weights.is_cuda(), "moe_combine expects CUDA");
  TORCH_CHECK(expert_out.dim() == 2, "expert_out must be [T*topk, N]");
  TORCH_CHECK(topk > 0, "moe_combine needs a positive topk");
  TORCH_CHECK(
      expert_out.size(0) % topk == 0,
      "moe_combine: ", expert_out.size(0), " rows is not a whole number of topk groups");
  const at::cuda::OptionalCUDAGuard guard(at::device_of(expert_out));
  const int64_t n_cols = expert_out.size(1);
  const int64_t tokens = expert_out.size(0) / topk;
  TORCH_CHECK(
      weights.numel() == tokens * topk,
      "moe_combine: ", weights.numel(), " weights for ", tokens * topk, " routed rows");
  TORCH_CHECK(
      expert_out.scalar_type() == at::kBFloat16 || expert_out.scalar_type() == at::kHalf,
      "moe_combine supports bf16 and fp16");
  auto out = at::empty({tokens, n_cols}, expert_out.options());
  const int threads = 256;
  const int64_t total = tokens * n_cols;
  const int blocks =
      static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 8192));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto w = weights.to(at::kFloat).contiguous();
  if (expert_out.scalar_type() == at::kBFloat16) {
    using scalar_t = at::BFloat16;
    moe_combine_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        expert_out.data_ptr<scalar_t>(), w.data_ptr<float>(), out.data_ptr<scalar_t>(),
        static_cast<int>(tokens), static_cast<int>(topk), static_cast<int>(n_cols));
  } else {
    using scalar_t = at::Half;
    moe_combine_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        expert_out.data_ptr<scalar_t>(), w.data_ptr<float>(), out.data_ptr<scalar_t>(),
        static_cast<int>(tokens), static_cast<int>(topk), static_cast<int>(n_cols));
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

}  // namespace infer
