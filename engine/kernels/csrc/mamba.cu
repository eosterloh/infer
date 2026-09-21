// Fused Mamba-2 selective scan, one kernel for prefill and decode.
//
// The readable PyTorch version walks the sequence in Python and materializes
// [B, H, D, N] temporaries for dA, dB and dBx at every step — on a 30B hybrid
// that is hundreds of megabytes of traffic and tens of thousands of launches
// per prefill. Here the recurrent state lives in registers for the whole
// sequence, so it is read and written exactly once no matter how long the
// sequence is, and dA collapses to the scalar it actually is.
#include "common.cuh"

namespace infer {
namespace {

constexpr int kScanWarps = 8;
constexpr int kScanThreads = kScanWarps * kWarpSize;

__device__ __forceinline__ float softplus(float x) {
  // log1p(exp(x)) with the standard large-x shortcut.
  return x > 20.0f ? x : log1pf(expf(x));
}

// kDPerWarp = head_dim / 8 and kNPerLane = state_size / 32 are compile time so
// the state array stays in registers instead of spilling to local memory.
template <typename scalar_t, int kDPerWarp, int kNPerLane>
__global__ void mamba2_scan_kernel(
    const scalar_t* __restrict__ x,      // [B, S, H, D]
    const scalar_t* __restrict__ dt_raw,  // [B, S, H]
    const float* __restrict__ dt_bias,    // [H]
    const float* __restrict__ a_log,      // [H]
    const scalar_t* __restrict__ b_mat,   // [B, S, G, N]
    const scalar_t* __restrict__ c_mat,   // [B, S, G, N]
    const float* __restrict__ d_skip,     // [H]
    float* __restrict__ state,            // [B, H, D, N]
    scalar_t* __restrict__ y,             // [B, S, H, D]
    int seq,
    int n_heads,
    int head_dim,
    int n_groups,
    int state_size,
    bool has_state,
    float dt_lo,
    float dt_hi) {
  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int group = head / (n_heads / n_groups);

  extern __shared__ float smem[];
  float* s_b = smem;                       // [state_size]
  float* s_c = smem + state_size;          // [state_size]
  float* s_x = smem + 2 * state_size;      // [head_dim]

  const int64_t state_base =
      ((static_cast<int64_t>(batch) * n_heads + head) * head_dim) * state_size;

  float acc[kDPerWarp * kNPerLane];
#pragma unroll
  for (int r = 0; r < kDPerWarp * kNPerLane; ++r) {
    acc[r] = 0.0f;
  }
  if (has_state) {
#pragma unroll
    for (int i = 0; i < kDPerWarp; ++i) {
      const int d = warp + i * kScanWarps;
#pragma unroll
      for (int j = 0; j < kNPerLane; ++j) {
        const int n = lane + j * kWarpSize;
        acc[i * kNPerLane + j] = state[state_base + static_cast<int64_t>(d) * state_size + n];
      }
    }
  }

  const float a_head = -expf(a_log[head]);
  const float d_head = d_skip[head];

  for (int t = 0; t < seq; ++t) {
    const int64_t bc_off =
        ((static_cast<int64_t>(batch) * seq + t) * n_groups + group) * state_size;
    const int64_t x_off =
        ((static_cast<int64_t>(batch) * seq + t) * n_heads + head) * head_dim;
    for (int i = threadIdx.x; i < state_size; i += kScanThreads) {
      s_b[i] = static_cast<float>(b_mat[bc_off + i]);
      s_c[i] = static_cast<float>(c_mat[bc_off + i]);
    }
    for (int i = threadIdx.x; i < head_dim; i += kScanThreads) {
      s_x[i] = static_cast<float>(x[x_off + i]);
    }
    __syncthreads();

    float dt = static_cast<float>(dt_raw[(static_cast<int64_t>(batch) * seq + t) * n_heads + head]);
    dt = softplus(dt + dt_bias[head]);
    dt = fminf(fmaxf(dt, dt_lo), dt_hi);
    // dA depends on the head and the step only; the elementwise expansion the
    // reference builds is [B, H, D, N] of one repeated value.
    const float da = expf(dt * a_head);

#pragma unroll
    for (int i = 0; i < kDPerWarp; ++i) {
      const int d = warp + i * kScanWarps;
      const float xv = s_x[d];
      const float dbx = dt * xv;
      float partial = 0.0f;
#pragma unroll
      for (int j = 0; j < kNPerLane; ++j) {
        const int n = lane + j * kWarpSize;
        const int r = i * kNPerLane + j;
        acc[r] = acc[r] * da + dbx * s_b[n];
        partial += acc[r] * s_c[n];
      }
      partial = warp_reduce_sum(partial);
      if (lane == 0) {
        y[x_off + d] = static_cast<scalar_t>(partial + xv * d_head);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < kDPerWarp; ++i) {
    const int d = warp + i * kScanWarps;
#pragma unroll
    for (int j = 0; j < kNPerLane; ++j) {
      const int n = lane + j * kWarpSize;
      state[state_base + static_cast<int64_t>(d) * state_size + n] = acc[i * kNPerLane + j];
    }
  }
}

template <typename scalar_t>
void launch_scan(
    const at::Tensor& x,
    const at::Tensor& dt_raw,
    const at::Tensor& dt_bias,
    const at::Tensor& a_log,
    const at::Tensor& b_mat,
    const at::Tensor& c_mat,
    const at::Tensor& d_skip,
    at::Tensor& state,
    at::Tensor& y,
    int seq,
    int n_heads,
    int head_dim,
    int n_groups,
    int state_size,
    bool has_state,
    float dt_lo,
    float dt_hi) {
  const dim3 blocks(static_cast<unsigned>(n_heads), static_cast<unsigned>(x.size(0)));
  const size_t shared = (2 * state_size + head_dim) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int d_per_warp = head_dim / kScanWarps;
  const int n_per_lane = state_size / kWarpSize;

#define INFER_SCAN_LAUNCH(D_PER_WARP, N_PER_LANE)                             \
  mamba2_scan_kernel<scalar_t, D_PER_WARP, N_PER_LANE>                        \
      <<<blocks, kScanThreads, shared, stream>>>(                             \
          x.data_ptr<scalar_t>(), dt_raw.data_ptr<scalar_t>(),                \
          dt_bias.data_ptr<float>(), a_log.data_ptr<float>(),                 \
          b_mat.data_ptr<scalar_t>(), c_mat.data_ptr<scalar_t>(),             \
          d_skip.data_ptr<float>(), state.data_ptr<float>(),                  \
          y.data_ptr<scalar_t>(), seq, n_heads, head_dim, n_groups,           \
          state_size, has_state, dt_lo, dt_hi);
#define INFER_SCAN_CASE(D_PER_WARP, N_PER_LANE)         \
  if (d_per_warp == (D_PER_WARP) && n_per_lane == (N_PER_LANE)) { \
    INFER_SCAN_LAUNCH(D_PER_WARP, N_PER_LANE)           \
    return;                                             \
  }

  // head_dim 8..128 by state_size 32/64/128, capped at 64 registers of state.
  INFER_SCAN_CASE(1, 1)
  INFER_SCAN_CASE(1, 2)
  INFER_SCAN_CASE(1, 4)
  INFER_SCAN_CASE(2, 1)
  INFER_SCAN_CASE(2, 2)
  INFER_SCAN_CASE(2, 4)
  INFER_SCAN_CASE(4, 1)
  INFER_SCAN_CASE(4, 2)
  INFER_SCAN_CASE(4, 4)
  INFER_SCAN_CASE(8, 1)
  INFER_SCAN_CASE(8, 2)
  INFER_SCAN_CASE(8, 4)
  INFER_SCAN_CASE(16, 1)
  INFER_SCAN_CASE(16, 2)
  INFER_SCAN_CASE(16, 4)
#undef INFER_SCAN_CASE
#undef INFER_SCAN_LAUNCH
  TORCH_CHECK(
      false, "mamba2_scan: unsupported head_dim ", head_dim, " / state_size ", state_size);
}

}  // namespace

at::Tensor mamba2_scan_cuda(
    const at::Tensor& x,
    const at::Tensor& dt_raw,
    const at::Tensor& dt_bias,
    const at::Tensor& a_log,
    const at::Tensor& b_mat,
    const at::Tensor& c_mat,
    const at::Tensor& d_skip,
    at::Tensor state,
    bool has_state,
    double dt_lo,
    double dt_hi) {
  TORCH_CHECK(x.is_cuda(), "mamba2_scan expects CUDA tensors");
  TORCH_CHECK(x.dim() == 4, "x must be [B, S, H, D]");
  TORCH_CHECK(b_mat.dim() == 4 && c_mat.dim() == 4, "B/C must be [B, S, G, N]");
  TORCH_CHECK(state.dim() == 4 && state.scalar_type() == at::kFloat, "state must be fp32 [B,H,D,N]");
  TORCH_CHECK(x.is_contiguous() && b_mat.is_contiguous() && c_mat.is_contiguous(),
              "mamba2_scan needs contiguous inputs");
  TORCH_CHECK(state.is_contiguous(), "state must be contiguous");

  const at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  const int batch = static_cast<int>(x.size(0));
  const int seq = static_cast<int>(x.size(1));
  const int n_heads = static_cast<int>(x.size(2));
  const int head_dim = static_cast<int>(x.size(3));
  const int n_groups = static_cast<int>(b_mat.size(2));
  const int state_size = static_cast<int>(b_mat.size(3));
  TORCH_CHECK(head_dim % kScanWarps == 0, "head_dim must be a multiple of 8");
  TORCH_CHECK(state_size % kWarpSize == 0, "state_size must be a multiple of 32");
  TORCH_CHECK(n_heads % n_groups == 0, "heads must divide evenly into groups");
  // The state offset is built from x and b_mat, so a state of any other shape
  // is written past its end.
  TORCH_CHECK(
      state.size(0) == batch && state.size(1) == n_heads && state.size(2) == head_dim &&
          state.size(3) == state_size,
      "mamba2_scan: state must be [B, H, D, N]");
  TORCH_CHECK(
      dt_raw.numel() == static_cast<int64_t>(batch) * seq * n_heads,
      "mamba2_scan: dt_raw must be [B, S, H]");
  TORCH_CHECK(
      dt_bias.numel() == n_heads && a_log.numel() == n_heads && d_skip.numel() == n_heads,
      "mamba2_scan: dt_bias, a_log and d_skip are indexed per head");

  auto y = at::empty_like(x);
  if (x.scalar_type() == at::kBFloat16) {
    launch_scan<at::BFloat16>(
        x, dt_raw, dt_bias, a_log, b_mat, c_mat, d_skip, state, y, seq, n_heads,
        head_dim, n_groups, state_size, has_state, static_cast<float>(dt_lo),
        static_cast<float>(dt_hi));
  } else if (x.scalar_type() == at::kFloat) {
    launch_scan<float>(
        x, dt_raw, dt_bias, a_log, b_mat, c_mat, d_skip, state, y, seq, n_heads,
        head_dim, n_groups, state_size, has_state, static_cast<float>(dt_lo),
        static_cast<float>(dt_hi));
  } else if (x.scalar_type() == at::kHalf) {
    launch_scan<at::Half>(
        x, dt_raw, dt_bias, a_log, b_mat, c_mat, d_skip, state, y, seq, n_heads,
        head_dim, n_groups, state_size, has_state, static_cast<float>(dt_lo),
        static_cast<float>(dt_hi));
  } else {
    TORCH_CHECK(false, "mamba2_scan supports bf16, fp16 and fp32");
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return y;
}

}  // namespace infer
