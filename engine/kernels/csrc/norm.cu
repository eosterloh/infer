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
      value.v[j] = static_cast<T>(
          static_cast<float>(value.v[j]) + static_cast<float>(res.v[j]));
      // Square what was stored, not the fp32 sum behind it. The residual is a
      // BF16 tensor either way, so norming the wider value would put this kernel
      // a fraction of an ulp off its own definition in every layer of every
      // model — the sort of standing difference that makes a later numerical bug
      // hard to bisect. Rounding first costs a register move.
      const float kept = static_cast<float>(value.v[j]);
      acc += kept * kept;
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
    row_r[i] = static_cast<T>(
        static_cast<float>(row_x[i]) + static_cast<float>(row_r[i]));
    const float kept = static_cast<float>(row_r[i]);
    acc += kept * kept;
  }
  const float inv = rsqrtf(block_reduce_sum(acc) / static_cast<float>(hidden) + eps);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float w = static_cast<float>(weight[i]) + weight_offset;
    row_x[i] = static_cast<T>(static_cast<float>(row_r[i]) * inv * w);
  }
}

// The gated norm the recurrent mixers need. Mamba-2 gates before the norm
// (RMS over x * silu(gate), per group), Gated DeltaNet gates after it. Either
// way the reference builds half a dozen fp32 temporaries the width of the
// hidden state, per layer, per token.
//
// The variance is per group, but the scale is per channel: Nemotron-H's eight
// groups share one `inter`-wide norm weight, so the weight has its own group
// index. `weight_groups == 1` is the case where every group scales alike.
template <typename T>
__global__ void gated_rms_norm_kernel(
    const T* __restrict__ x,
    const T* __restrict__ gate,
    const T* __restrict__ weight,
    T* __restrict__ out,
    int group,
    int weight_groups,
    float eps,
    bool gate_first) {
  const int64_t row = blockIdx.x;
  const T* __restrict__ row_x = x + row * group;
  const T* __restrict__ row_g = gate + row * group;
  const T* __restrict__ row_w =
      weight + (weight_groups > 1 ? (row % weight_groups) * group : 0);
  T* __restrict__ row_o = out + row * group;

  float acc = 0.0f;
  for (int i = threadIdx.x; i < group; i += blockDim.x) {
    float value = static_cast<float>(row_x[i]);
    if (gate_first) {
      value *= silu(static_cast<float>(row_g[i]));
    }
    acc += value * value;
  }
  const float inv = rsqrtf(block_reduce_sum(acc) / static_cast<float>(group) + eps);
  for (int i = threadIdx.x; i < group; i += blockDim.x) {
    const float g = silu(static_cast<float>(row_g[i]));
    const float w = static_cast<float>(row_w[i]);
    float value = static_cast<float>(row_x[i]);
    if (gate_first) {
      value = value * g * inv * w;
    } else {
      value = value * inv * w * g;
    }
    row_o[i] = static_cast<T>(value);
  }
}

at::Tensor gated_rms_norm_cuda(
    const at::Tensor& x,
    const at::Tensor& gate,
    const at::Tensor& weight,
    double eps,
    int64_t group,
    bool gate_first) {
  const at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  auto input = x.contiguous();
  auto g = gate.contiguous();
  auto w = weight.contiguous().view(-1);
  TORCH_CHECK(input.sizes() == g.sizes(), "gated_rms_norm: gate shape mismatch");
  TORCH_CHECK(group > 0 && input.numel() % group == 0, "gated_rms_norm: bad group");
  TORCH_CHECK(
      w.numel() % group == 0, "gated_rms_norm: weight must be a whole number of groups");
  TORCH_CHECK(input.scalar_type() == w.scalar_type(), "gated_rms_norm: dtype mismatch");
  auto out = at::empty_like(input);
  const int64_t rows = input.numel() / group;
  const int64_t weight_groups = w.numel() / group;
  TORCH_CHECK(
      rows % weight_groups == 0, "gated_rms_norm: rows do not divide by the weight groups");
  if (rows == 0) {
    return out;
  }
  const int threads = threads_for(static_cast<int>(group));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "gated_rms_norm_cuda",
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        using T = at::BFloat16;
        gated_rms_norm_kernel<T><<<rows, threads, 0, stream>>>(
            input.data_ptr<T>(), g.data_ptr<T>(), w.data_ptr<T>(), out.data_ptr<T>(),
            static_cast<int>(group), static_cast<int>(weight_groups),
            static_cast<float>(eps), gate_first);
      })
      AT_DISPATCH_CASE(at::kHalf, [&] {
        using T = at::Half;
        gated_rms_norm_kernel<T><<<rows, threads, 0, stream>>>(
            input.data_ptr<T>(), g.data_ptr<T>(), w.data_ptr<T>(), out.data_ptr<T>(),
            static_cast<int>(group), static_cast<int>(weight_groups),
            static_cast<float>(eps), gate_first);
      })
      AT_DISPATCH_CASE(at::kFloat, [&] {
        using T = float;
        gated_rms_norm_kernel<T><<<rows, threads, 0, stream>>>(
            input.data_ptr<T>(), g.data_ptr<T>(), w.data_ptr<T>(), out.data_ptr<T>(),
            static_cast<int>(group), static_cast<int>(weight_groups),
            static_cast<float>(eps), gate_first);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
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
  AT_CUDA_CHECK(cudaGetLastError());
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
  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace infer
