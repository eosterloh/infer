// Shared device helpers: vector types, reductions, activations.
#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

namespace infer {

// 16-byte loads whenever the element count allows it. Reading a bf16 row eight
// elements at a time is what keeps these kernels on the memory roofline
// instead of the instruction-issue roofline.
template <typename T, int N>
struct alignas(sizeof(T) * N) Vec {
  T v[N];
};

constexpr int kMaxThreads = 1024;
constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

// One reduction across the block, result broadcast to every thread.
__device__ __forceinline__ float block_reduce_sum(float value) {
  __shared__ float warp_sums[kWarpSize];
  __shared__ float total;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x >> 5;
  const int warps = (blockDim.x + kWarpSize - 1) / kWarpSize;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
    float acc = (lane < warps) ? warp_sums[lane] : 0.0f;
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
      total = acc;
    }
  }
  __syncthreads();
  return total;
}

// Exact transcendentals, not the fast-math approximations: these kernels are
// memory bound, so the extra accuracy costs nothing measurable and keeps the
// results inside a bf16 ulp of PyTorch.
__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float gelu_tanh(float x) {
  const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float gelu_erf(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

__device__ __forceinline__ float relu2(float x) {
  const float r = x > 0.0f ? x : 0.0f;
  return r * r;
}

inline int threads_for(int64_t vectors) {
  int threads = 128;
  while (threads < kMaxThreads && threads < vectors) {
    threads *= 2;
  }
  return threads;
}

// A row can be read with 16-byte loads only when its length and its base
// offset are both a whole number of vectors.
inline bool vectorizable(const at::Tensor& t, int64_t width, int vec) {
  if (width % vec != 0) {
    return false;
  }
  const int64_t bytes = vec * t.element_size();
  return (reinterpret_cast<uintptr_t>(t.data_ptr()) % bytes) == 0;
}

}  // namespace infer
