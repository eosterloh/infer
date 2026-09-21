// RMSNorm variants. One pass over the row, fp32 accumulation, 16-byte loads.
#include "common.cuh"

namespace infer {

// x is kept in registers between the reduce and the scale so the row is read
// from global memory exactly once.
template <typename T, int VEC, int MAX_VEC_PER_THREAD>
__global__ void rms_norm_cached_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int hidden,
    float eps,
    float weight_offset) {
  using V = Vec<T, VEC>;
  const int64_t row = blockIdx.x;
  const int vectors = hidden / VEC;
  const V* __restrict__ xv = reinterpret_cast<const V*>(x + row * hidden);
  const V* __restrict__ wv = reinterpret_cast<const V*>(weight);
  V* __restrict__ ov = reinterpret_cast<V*>(out + row * hidden);

  V cached[MAX_VEC_PER_THREAD];
  float acc = 0.0f;
  int slot = 0;
  for (int i = threadIdx.x; i < vectors; i += blockDim.x, ++slot) {
    V value = xv[i];
    cached[slot] = value;
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const float f = static_cast<float>(value.v[j]);
      acc += f * f;
    }
  }
  const float inv = rsqrtf(block_reduce_sum(acc) / static_cast<float>(hidden) + eps);
  slot = 0;
  for (int i = threadIdx.x; i < vectors; i += blockDim.x, ++slot) {
    V value = cached[slot];
    const V w = wv[i];
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const float scaled = static_cast<float>(value.v[j]) * inv;
      value.v[j] = static_cast<T>(scaled * (static_cast<float>(w.v[j]) + weight_offset));
    }
    ov[i] = value;
  }
}

// Fallback for rows too wide to cache in registers, and for hidden sizes that
// are not a multiple of the vector width.
template <typename T>
__global__ void rms_norm_scalar_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int hidden,
    float eps,
    float weight_offset) {
  const int64_t row = blockIdx.x;
  const T* __restrict__ row_x = x + row * hidden;
  T* __restrict__ row_o = out + row * hidden;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float f = static_cast<float>(row_x[i]);
    acc += f * f;
  }
  const float inv = rsqrtf(block_reduce_sum(acc) / static_cast<float>(hidden) + eps);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float w = static_cast<float>(weight[i]) + weight_offset;
    row_o[i] = static_cast<T>(static_cast<float>(row_x[i]) * inv * w);
  }
}

// residual += x ; x = rms_norm(residual) * weight. Saves a full read/write of
// the hidden state per block compared with add-then-norm.
template <typename T, int VEC, int MAX_VEC_PER_THREAD>
__global__ void fused_add_rms_norm_kernel(
    T* __restrict__ x,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    int hidden,
    float eps,
    float weight_offset) {
  using V = Vec<T, VEC>;
  const int64_t row = blockIdx.x;
  const int vectors = hidden / VEC;
  V* __restrict__ xv = reinterpret_cast<V*>(x + row * hidden);
  V* __restrict__ rv = reinterpret_cast<V*>(residual + row * hidden);
  const V* __restrict__ wv = reinterpret_cast<const V*>(weight);

  V cached[MAX_VEC_PER_THREAD];
  float acc = 0.0f;
  int slot = 0;
  for (int i = threadIdx.x; i < vectors; i += blockDim.x, ++slot) {
    V value = xv[i];
    const V res = rv[i];
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const float sum = static_cast<float>(value.v[j]) + static_cast<float>(res.v[j]);
      value.v[j] = static_cast<T>(sum);
      acc += sum * sum;
    }
    cached[slot] = value;
    rv[i] = value;
  }
  const float inv = rsqrtf(block_reduce_sum(acc) / static_cast<float>(hidden) + eps);
  slot = 0;
  for (int i = threadIdx.x; i < vectors; i += blockDim.x, ++slot) {
    V value = cached[slot];
    const V w = wv[i];
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const float scaled = static_cast<float>(value.v[j]) * inv;
      value.v[j] = static_cast<T>(scaled * (static_cast<float>(w.v[j]) + weight_offset));
    }
    xv[i] = value;
  }
}

template <typename T>
__global__ void fused_add_rms_norm_scalar_kernel(
    T* __restrict__ x,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    int hidden,
    float eps,
    float weight_offset) {
  const int64_t row = blockIdx.x;
  T* __restrict__ row_x = x + row * hidden;
  T* __restrict__ row_r = residual + row * hidden;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float sum = static_cast<float>(row_x[i]) + static_cast<float>(row_r[i]);
    row_r[i] = static_cast<T>(sum);
    acc += sum * sum;
  }
  const float inv = rsqrtf(block_reduce_sum(acc) / static_cast<float>(hidden) + eps);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float w = static_cast<float>(weight[i]) + weight_offset;
    row_x[i] = static_cast<T>(static_cast<float>(row_r[i]) * inv * w);
  }
}

#define INFER_LAUNCH_NORM(T, VEC)                                              \
  do {                                                                         \
    const int vectors = static_cast<int>(hidden / VEC);                        \
    const int threads = threads_for(vectors);                                  \
    const int per_thread = (vectors + threads - 1) / threads;                  \
    if (per_thread <= 4) {                                                      \
      rms_norm_cached_kernel<T, VEC, 4><<<rows, threads, 0, stream>>>(         \
          x_ptr, w_ptr, o_ptr, hidden, eps_f, offset_f);                        \
    } else if (per_thread <= 16) {                                              \
      rms_norm_cached_kernel<T, VEC, 16><<<rows, threads, 0, stream>>>(        \
          x_ptr, w_ptr, o_ptr, hidden, eps_f, offset_f);                        \
    } else {                                                                    \
      rms_norm_scalar_kernel<T><<<rows, threads, 0, stream>>>(                 \
          x_ptr, w_ptr, o_ptr, hidden, eps_f, offset_f);                        \
    }                                                                           \
  } while (0)

at::Tensor rms_norm_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    double eps,
    double weight_offset) {
  const at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  auto input = x.contiguous();
  auto w = weight.contiguous().view(-1);
  TORCH_CHECK(input.size(-1) == w.numel(), "rms_norm: weight must match hidden dim");
  TORCH_CHECK(input.scalar_type() == w.scalar_type(), "rms_norm: dtype mismatch");
  auto out = at::empty_like(input);
  const int hidden = static_cast<int>(input.size(-1));
  const int64_t rows = input.numel() / hidden;
  if (rows == 0) {
    return out;
  }
  const float eps_f = static_cast<float>(eps);
  const float offset_f = static_cast<float>(weight_offset);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "rms_norm_cuda",
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        using T = at::BFloat16;
        const T* x_ptr = input.data_ptr<T>();
        const T* w_ptr = w.data_ptr<T>();
        T* o_ptr = out.data_ptr<T>();
        if (vectorizable(input, hidden, 8) && vectorizable(w, hidden, 8)) {
          INFER_LAUNCH_NORM(T, 8);
        } else {
          rms_norm_scalar_kernel<T><<<rows, 256, 0, stream>>>(
              x_ptr, w_ptr, o_ptr, hidden, eps_f, offset_f);
        }
      })
      AT_DISPATCH_CASE(at::kHalf, [&] {
        using T = at::Half;
        const T* x_ptr = input.data_ptr<T>();
        const T* w_ptr = w.data_ptr<T>();
        T* o_ptr = out.data_ptr<T>();
        if (vectorizable(input, hidden, 8) && vectorizable(w, hidden, 8)) {
          INFER_LAUNCH_NORM(T, 8);
        } else {
          rms_norm_scalar_kernel<T><<<rows, 256, 0, stream>>>(
              x_ptr, w_ptr, o_ptr, hidden, eps_f, offset_f);
        }
      })
      AT_DISPATCH_CASE(at::kFloat, [&] {
        using T = float;
        const T* x_ptr = input.data_ptr<T>();
        const T* w_ptr = w.data_ptr<T>();
        T* o_ptr = out.data_ptr<T>();
        if (vectorizable(input, hidden, 4) && vectorizable(w, hidden, 4)) {
          INFER_LAUNCH_NORM(T, 4);
        } else {
          rms_norm_scalar_kernel<T><<<rows, 256, 0, stream>>>(
              x_ptr, w_ptr, o_ptr, hidden, eps_f, offset_f);
        }
      }));
  return out;
}

#define INFER_LAUNCH_FUSED(T, VEC)                                             \
  do {                                                                         \
    const int vectors = static_cast<int>(hidden / VEC);                        \
    const int threads = threads_for(vectors);                                  \
    const int per_thread = (vectors + threads - 1) / threads;                  \
    if (per_thread <= 4) {                                                      \
      fused_add_rms_norm_kernel<T, VEC, 4><<<rows, threads, 0, stream>>>(      \
          x_ptr, r_ptr, w_ptr, hidden, eps_f, offset_f);                        \
    } else if (per_thread <= 16) {                                              \
      fused_add_rms_norm_kernel<T, VEC, 16><<<rows, threads, 0, stream>>>(     \
          x_ptr, r_ptr, w_ptr, hidden, eps_f, offset_f);                        \
    } else {                                                                    \
      fused_add_rms_norm_scalar_kernel<T><<<rows, threads, 0, stream>>>(       \
          x_ptr, r_ptr, w_ptr, hidden, eps_f, offset_f);                        \
    }                                                                           \
  } while (0)

void fused_add_rms_norm_cuda(
    at::Tensor& x,
    at::Tensor& residual,
    const at::Tensor& weight,
    double eps,
    double weight_offset) {
  const at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  TORCH_CHECK(x.is_contiguous() && residual.is_contiguous(),
              "fused_add_rms_norm needs contiguous x and residual");
  TORCH_CHECK(x.sizes() == residual.sizes(), "fused_add_rms_norm shape mismatch");
  auto w = weight.contiguous().view(-1);
  const int hidden = static_cast<int>(x.size(-1));
  TORCH_CHECK(hidden == w.numel(), "fused_add_rms_norm: weight must match hidden dim");
  const int64_t rows = x.numel() / hidden;
  if (rows == 0) {
    return;
  }
  const float eps_f = static_cast<float>(eps);
  const float offset_f = static_cast<float>(weight_offset);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_SWITCH(
      x.scalar_type(),
      "fused_add_rms_norm_cuda",
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        using T = at::BFloat16;
        T* x_ptr = x.data_ptr<T>();
        T* r_ptr = residual.data_ptr<T>();
        const T* w_ptr = w.data_ptr<T>();
        if (vectorizable(x, hidden, 8) && vectorizable(residual, hidden, 8) &&
            vectorizable(w, hidden, 8)) {
          INFER_LAUNCH_FUSED(T, 8);
        } else {
          fused_add_rms_norm_scalar_kernel<T><<<rows, 256, 0, stream>>>(
              x_ptr, r_ptr, w_ptr, hidden, eps_f, offset_f);
        }
      })
      AT_DISPATCH_CASE(at::kHalf, [&] {
        using T = at::Half;
        T* x_ptr = x.data_ptr<T>();
        T* r_ptr = residual.data_ptr<T>();
        const T* w_ptr = w.data_ptr<T>();
        if (vectorizable(x, hidden, 8) && vectorizable(residual, hidden, 8) &&
            vectorizable(w, hidden, 8)) {
          INFER_LAUNCH_FUSED(T, 8);
        } else {
          fused_add_rms_norm_scalar_kernel<T><<<rows, 256, 0, stream>>>(
              x_ptr, r_ptr, w_ptr, hidden, eps_f, offset_f);
        }
      })
      AT_DISPATCH_CASE(at::kFloat, [&] {
        using T = float;
        T* x_ptr = x.data_ptr<T>();
        T* r_ptr = residual.data_ptr<T>();
        const T* w_ptr = w.data_ptr<T>();
        if (vectorizable(x, hidden, 4) && vectorizable(residual, hidden, 4) &&
            vectorizable(w, hidden, 4)) {
          INFER_LAUNCH_FUSED(T, 4);
        } else {
          fused_add_rms_norm_scalar_kernel<T><<<rows, 256, 0, stream>>>(
              x_ptr, r_ptr, w_ptr, hidden, eps_f, offset_f);
        }
      }));
}

}  // namespace infer
