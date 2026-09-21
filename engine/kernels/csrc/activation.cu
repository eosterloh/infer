// Gated activations. act(gate) * up in one pass instead of three.
#include "common.cuh"

namespace infer {

enum class Act { Silu, GeluTanh, GeluErf, Relu2 };

__device__ __forceinline__ float apply_act(float x, Act act) {
  switch (act) {
    case Act::Silu:
      return silu(x);
    case Act::GeluTanh:
      return gelu_tanh(x);
    case Act::GeluErf:
      return gelu_erf(x);
    default:
      return relu2(x);
  }
}

// gate and up in separate tensors (Llama-style split gate_proj / up_proj).
template <typename T, int VEC, Act ACT>
__global__ void act_mul_kernel(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    T* __restrict__ out,
    int64_t vectors) {
  using V = Vec<T, VEC>;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const V* __restrict__ gv = reinterpret_cast<const V*>(gate);
  const V* __restrict__ uv = reinterpret_cast<const V*>(up);
  V* __restrict__ ov = reinterpret_cast<V*>(out);
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < vectors; i += stride) {
    V g = gv[i];
    const V u = uv[i];
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      g.v[j] = static_cast<T>(
          apply_act(static_cast<float>(g.v[j]), ACT) * static_cast<float>(u.v[j]));
    }
    ov[i] = g;
  }
}

template <typename T, Act ACT>
__global__ void act_mul_scalar_kernel(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    T* __restrict__ out,
    int64_t n) {
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    out[i] = static_cast<T>(
        apply_act(static_cast<float>(gate[i]), ACT) * static_cast<float>(up[i]));
  }
}

// One packed [.., 2 * inter] tensor (Phi-3 / MoE gate_up layout).
template <typename T, int VEC, Act ACT>
__global__ void act_and_mul_packed_kernel(
    const T* __restrict__ in,
    T* __restrict__ out,
    int inter,
    int64_t rows) {
  using V = Vec<T, VEC>;
  const int vectors = inter / VEC;
  const int64_t total = rows * vectors;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    const int64_t row = idx / vectors;
    const int col = static_cast<int>(idx % vectors);
    const V* __restrict__ gv =
        reinterpret_cast<const V*>(in + row * 2 * inter);
    const V* __restrict__ uv =
        reinterpret_cast<const V*>(in + row * 2 * inter + inter);
    V g = gv[col];
    const V u = uv[col];
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      g.v[j] = static_cast<T>(
          apply_act(static_cast<float>(g.v[j]), ACT) * static_cast<float>(u.v[j]));
    }
    reinterpret_cast<V*>(out + row * inter)[col] = g;
  }
}

static inline int blocks_for(int64_t work, int threads, int cap = 8192) {
  const int64_t needed = (work + threads - 1) / threads;
  return static_cast<int>(needed < cap ? (needed > 0 ? needed : 1) : cap);
}

template <typename T, int VEC>
static void dispatch_act_mul(
    const T* gate,
    const T* up,
    T* out,
    int64_t n,
    bool vectorized,
    Act act,
    cudaStream_t stream) {
  const int threads = 256;
  if (vectorized) {
    const int64_t vectors = n / VEC;
    const int blocks = blocks_for(vectors, threads);
    switch (act) {
      case Act::Silu:
        act_mul_kernel<T, VEC, Act::Silu><<<blocks, threads, 0, stream>>>(gate, up, out, vectors);
        return;
      case Act::GeluTanh:
        act_mul_kernel<T, VEC, Act::GeluTanh><<<blocks, threads, 0, stream>>>(gate, up, out, vectors);
        return;
      case Act::GeluErf:
        act_mul_kernel<T, VEC, Act::GeluErf><<<blocks, threads, 0, stream>>>(gate, up, out, vectors);
        return;
      default:
        act_mul_kernel<T, VEC, Act::Relu2><<<blocks, threads, 0, stream>>>(gate, up, out, vectors);
        return;
    }
  }
  const int blocks = blocks_for(n, threads);
  switch (act) {
    case Act::Silu:
      act_mul_scalar_kernel<T, Act::Silu><<<blocks, threads, 0, stream>>>(gate, up, out, n);
      return;
    case Act::GeluTanh:
      act_mul_scalar_kernel<T, Act::GeluTanh><<<blocks, threads, 0, stream>>>(gate, up, out, n);
      return;
    case Act::GeluErf:
      act_mul_scalar_kernel<T, Act::GeluErf><<<blocks, threads, 0, stream>>>(gate, up, out, n);
      return;
    default:
      act_mul_scalar_kernel<T, Act::Relu2><<<blocks, threads, 0, stream>>>(gate, up, out, n);
      return;
  }
}

template <typename T, int VEC>
static void launch_packed(
    const T* in,
    T* out,
    int inter,
    int64_t rows,
    Act act,
    cudaStream_t stream) {
  const int threads = 256;
  const int blocks = blocks_for(rows * (inter / VEC), threads);
  switch (act) {
    case Act::Silu:
      act_and_mul_packed_kernel<T, VEC, Act::Silu><<<blocks, threads, 0, stream>>>(
          in, out, inter, rows);
      return;
    case Act::GeluTanh:
      act_and_mul_packed_kernel<T, VEC, Act::GeluTanh><<<blocks, threads, 0, stream>>>(
          in, out, inter, rows);
      return;
    case Act::GeluErf:
      act_and_mul_packed_kernel<T, VEC, Act::GeluErf><<<blocks, threads, 0, stream>>>(
          in, out, inter, rows);
      return;
    default:
      act_and_mul_packed_kernel<T, VEC, Act::Relu2><<<blocks, threads, 0, stream>>>(
          in, out, inter, rows);
      return;
  }
}

static Act parse_act(const std::string& name) {
  if (name == "silu" || name == "swiglu") return Act::Silu;
  if (name == "gelu_tanh" || name == "gelu_pytorch_tanh" || name == "gelu_new") {
    return Act::GeluTanh;
  }
  if (name == "gelu" || name == "gelu_erf") return Act::GeluErf;
  if (name == "relu2" || name == "relu_squared" || name == "squared_relu") {
    return Act::Relu2;
  }
  TORCH_CHECK(false, "unsupported fused activation: ", name);
}

at::Tensor act_mul_cuda(
    const at::Tensor& gate,
    const at::Tensor& up,
    const std::string& act_name) {
  const at::cuda::OptionalCUDAGuard guard(at::device_of(gate));
  TORCH_CHECK(gate.sizes() == up.sizes(), "act_mul shape mismatch");
  TORCH_CHECK(gate.scalar_type() == up.scalar_type(), "act_mul dtype mismatch");
  auto g = gate.contiguous();
  auto u = up.contiguous();
  auto out = at::empty_like(g);
  const int64_t n = g.numel();
  if (n == 0) {
    return out;
  }
  const Act act = parse_act(act_name);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_SWITCH(
      g.scalar_type(),
      "act_mul_cuda",
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        using T = at::BFloat16;
        const bool vec = vectorizable(g, n, 8) && vectorizable(u, n, 8);
        dispatch_act_mul<T, 8>(
            g.data_ptr<T>(), u.data_ptr<T>(), out.data_ptr<T>(), n, vec, act, stream);
      })
      AT_DISPATCH_CASE(at::kHalf, [&] {
        using T = at::Half;
        const bool vec = vectorizable(g, n, 8) && vectorizable(u, n, 8);
        dispatch_act_mul<T, 8>(
            g.data_ptr<T>(), u.data_ptr<T>(), out.data_ptr<T>(), n, vec, act, stream);
      })
      AT_DISPATCH_CASE(at::kFloat, [&] {
        using T = float;
        const bool vec = vectorizable(g, n, 4) && vectorizable(u, n, 4);
        dispatch_act_mul<T, 4>(
            g.data_ptr<T>(), u.data_ptr<T>(), out.data_ptr<T>(), n, vec, act, stream);
      }));
  return out;
}

at::Tensor act_and_mul_cuda(const at::Tensor& gate_up, const std::string& act_name) {
  const at::cuda::OptionalCUDAGuard guard(at::device_of(gate_up));
  auto input = gate_up.contiguous();
  const int64_t width = input.size(-1);
  TORCH_CHECK(width % 2 == 0, "act_and_mul needs an even last dim");
  const int inter = static_cast<int>(width / 2);
  const int64_t rows = input.numel() / width;
  auto sizes = input.sizes().vec();
  sizes.back() = inter;
  auto out = at::empty(sizes, input.options());
  if (rows == 0) {
    return out;
  }
  const Act act = parse_act(act_name);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "act_and_mul_cuda",
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        using T = at::BFloat16;
        const T* in = input.data_ptr<T>();
        T* dst = out.data_ptr<T>();
        if (inter % 8 == 0 && vectorizable(input, width, 8)) {
          launch_packed<T, 8>(in, dst, inter, rows, act, stream);
        } else {
          launch_packed<T, 1>(in, dst, inter, rows, act, stream);
        }
      })
      AT_DISPATCH_CASE(at::kHalf, [&] {
        using T = at::Half;
        const T* in = input.data_ptr<T>();
        T* dst = out.data_ptr<T>();
        if (inter % 8 == 0 && vectorizable(input, width, 8)) {
          launch_packed<T, 8>(in, dst, inter, rows, act, stream);
        } else {
          launch_packed<T, 1>(in, dst, inter, rows, act, stream);
        }
      })
      AT_DISPATCH_CASE(at::kFloat, [&] {
        using T = float;
        const T* in = input.data_ptr<T>();
        T* dst = out.data_ptr<T>();
        if (inter % 4 == 0 && vectorizable(input, width, 4)) {
          launch_packed<T, 4>(in, dst, inter, rows, act, stream);
        } else {
          launch_packed<T, 1>(in, dst, inter, rows, act, stream);
        }
      }));
  return out;
}

}  // namespace infer
