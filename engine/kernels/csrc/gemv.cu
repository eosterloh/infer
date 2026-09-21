// Matrix-vector product for batch-1 decode: y = W @ x, W is [N, K] row major.
//
// Decode reads every weight once and does two flops per element, so the only
// goal is to keep the memory pipe full: one warp per output row, 16-byte loads,
// fp32 accumulation, and enough rows in flight to cover latency.
#include "common.cuh"

namespace infer {

constexpr int kGemvVec = 8;

template <typename T, int WARPS, int UNROLL>
__global__ void gemv_kernel(
    const T* __restrict__ w,
    const T* __restrict__ x,
    const T* __restrict__ bias,
    T* __restrict__ y,
    int n,
    int k) {
  using V = Vec<T, kGemvVec>;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * WARPS + warp;
  if (row >= n) {
    return;
  }
  const int vectors = k / kGemvVec;
  const V* __restrict__ wv = reinterpret_cast<const V*>(w + static_cast<int64_t>(row) * k);
  const V* __restrict__ xv = reinterpret_cast<const V*>(x);

  float acc = 0.0f;
  int i = lane;
  // Several vectors per thread per step: the loads issue back to back instead
  // of serializing on one outstanding request per thread.
  const int step = kWarpSize * UNROLL;
  for (; i + kWarpSize * (UNROLL - 1) < vectors; i += step) {
    V wr[UNROLL];
    V xr[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      wr[u] = wv[i + u * kWarpSize];
      xr[u] = xv[i + u * kWarpSize];
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
#pragma unroll
      for (int j = 0; j < kGemvVec; ++j) {
        acc += static_cast<float>(wr[u].v[j]) * static_cast<float>(xr[u].v[j]);
      }
    }
  }
  for (; i < vectors; i += kWarpSize) {
    const V wr = wv[i];
    const V xr = xv[i];
#pragma unroll
    for (int j = 0; j < kGemvVec; ++j) {
      acc += static_cast<float>(wr.v[j]) * static_cast<float>(xr.v[j]);
    }
  }
  for (int t = vectors * kGemvVec + lane; t < k; t += kWarpSize) {
    acc += static_cast<float>(w[static_cast<int64_t>(row) * k + t]) *
           static_cast<float>(x[t]);
  }
  acc = warp_reduce_sum(acc);
  if (lane == 0) {
    if (bias != nullptr) {
      acc += static_cast<float>(bias[row]);
    }
    y[row] = static_cast<T>(acc);
  }
}

template <typename T>
static void launch_gemv(
    const T* w,
    const T* x,
    const T* bias,
    T* y,
    int n,
    int k,
    cudaStream_t stream) {
  constexpr int kWarps = 4;
  const int threads = kWarps * kWarpSize;
  const int blocks = (n + kWarps - 1) / kWarps;
  const int vectors = k / kGemvVec;
  if (vectors >= 4 * kWarpSize) {
    gemv_kernel<T, kWarps, 4><<<blocks, threads, 0, stream>>>(w, x, bias, y, n, k);
  } else if (vectors >= 2 * kWarpSize) {
    gemv_kernel<T, kWarps, 2><<<blocks, threads, 0, stream>>>(w, x, bias, y, n, k);
  } else {
    gemv_kernel<T, kWarps, 1><<<blocks, threads, 0, stream>>>(w, x, bias, y, n, k);
  }
}

at::Tensor gemv_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias) {
  const at::cuda::OptionalCUDAGuard guard(at::device_of(weight));
  TORCH_CHECK(weight.dim() == 2, "gemv expects a 2-D weight");
  TORCH_CHECK(weight.is_contiguous(), "gemv expects a contiguous weight");
  const int64_t k = weight.size(1);
  const int64_t n = weight.size(0);
  TORCH_CHECK(x.size(-1) == k, "gemv: x and weight disagree on K");
  TORCH_CHECK(x.numel() == k, "gemv handles a single row; use a GEMM for more");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "gemv dtype mismatch");
  auto xc = x.contiguous();

  auto sizes = x.sizes().vec();
  sizes.back() = n;
  auto out = at::empty(sizes, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const at::Tensor* b = bias.has_value() ? &bias.value() : nullptr;
  if (b != nullptr) {
    TORCH_CHECK(b->numel() == n, "gemv bias size mismatch");
    TORCH_CHECK(b->scalar_type() == weight.scalar_type(), "gemv bias dtype mismatch");
  }

  AT_DISPATCH_SWITCH(
      weight.scalar_type(),
      "gemv_cuda",
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        using T = at::BFloat16;
        launch_gemv<T>(
            weight.data_ptr<T>(),
            xc.data_ptr<T>(),
            b ? b->data_ptr<T>() : nullptr,
            out.data_ptr<T>(),
            static_cast<int>(n),
            static_cast<int>(k),
            stream);
      })
      AT_DISPATCH_CASE(at::kHalf, [&] {
        using T = at::Half;
        launch_gemv<T>(
            weight.data_ptr<T>(),
            xc.data_ptr<T>(),
            b ? b->data_ptr<T>() : nullptr,
            out.data_ptr<T>(),
            static_cast<int>(n),
            static_cast<int>(k),
            stream);
      }));
  return out;
}

}  // namespace infer
