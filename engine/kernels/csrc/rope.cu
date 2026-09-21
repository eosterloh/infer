// RoPE applied in place to Q and K in a single launch.
//
// The PyTorch version costs roughly ten kernels and two allocations per layer
// (slice, neg, cat, four multiplies, one add, twice). Here each thread owns one
// rotated pair and both tensors are handled by the same grid, so a layer's
// rotary cost is one launch that reads and writes only the rotary slice.
#include "common.cuh"

namespace infer {

template <typename T, bool INTERLEAVED>
__global__ void rope_inplace_kernel(
    T* __restrict__ q,
    T* __restrict__ k,
    const T* __restrict__ cos,
    const T* __restrict__ sin,
    int heads_q,
    int heads_k,
    int seq,
    int head_dim,
    int rotary_dim,
    int64_t q_stride_b,
    int64_t q_stride_h,
    int64_t q_stride_s,
    int64_t k_stride_b,
    int64_t k_stride_h,
    int64_t k_stride_s,
    int64_t cos_stride_b,
    int64_t cos_stride_s,
    int64_t total) {
  const int half = rotary_dim / 2;
  const int heads = heads_q + heads_k;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    int64_t rest = idx;
    const int pair = static_cast<int>(rest % half);
    rest /= half;
    const int s = static_cast<int>(rest % seq);
    rest /= seq;
    const int head = static_cast<int>(rest % heads);
    const int b = static_cast<int>(rest / heads);

    T* base;
    if (head < heads_q) {
      base = q + b * q_stride_b + head * q_stride_h + s * q_stride_s;
    } else {
      base = k + b * k_stride_b + (head - heads_q) * k_stride_h + s * k_stride_s;
    }
    const T* c = cos + b * cos_stride_b + s * cos_stride_s;
    const T* sn = sin + b * cos_stride_b + s * cos_stride_s;

    if (INTERLEAVED) {
      // GPT-J / Cohere / GLM: rotate adjacent even/odd lanes.
      const float cv = static_cast<float>(c[pair]);
      const float sv = static_cast<float>(sn[pair]);
      const float x0 = static_cast<float>(base[2 * pair]);
      const float x1 = static_cast<float>(base[2 * pair + 1]);
      base[2 * pair] = static_cast<T>(x0 * cv - x1 * sv);
      base[2 * pair + 1] = static_cast<T>(x1 * cv + x0 * sv);
    } else {
      // Llama / NeoX: rotate the two halves of the rotary slice.
      const float x0 = static_cast<float>(base[pair]);
      const float x1 = static_cast<float>(base[pair + half]);
      base[pair] = static_cast<T>(
          x0 * static_cast<float>(c[pair]) - x1 * static_cast<float>(sn[pair]));
      base[pair + half] = static_cast<T>(
          x1 * static_cast<float>(c[pair + half]) +
          x0 * static_cast<float>(sn[pair + half]));
    }
  }
}

void rope_inplace_cuda(
    at::Tensor& q,
    at::Tensor& k,
    const at::Tensor& cos,
    const at::Tensor& sin,
    bool interleaved) {
  const at::cuda::OptionalCUDAGuard guard(at::device_of(q));
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "rope_inplace expects [B, H, S, D]");
  TORCH_CHECK(cos.dim() == 3 && sin.dim() == 3, "rope_inplace expects cos/sin [B, S, R]");
  TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1, "rope_inplace needs a contiguous head dim");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "rope_inplace dtype mismatch");
  TORCH_CHECK(cos.scalar_type() == q.scalar_type(), "rope_inplace cos dtype mismatch");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(2) == k.size(2), "rope_inplace batch/seq mismatch");
  TORCH_CHECK(q.size(3) == k.size(3), "rope_inplace head_dim mismatch");

  auto cos_c = cos.contiguous();
  auto sin_c = sin.contiguous();
  const int batch = static_cast<int>(q.size(0));
  const int heads_q = static_cast<int>(q.size(1));
  const int heads_k = static_cast<int>(k.size(1));
  const int seq = static_cast<int>(q.size(2));
  const int head_dim = static_cast<int>(q.size(3));
  const int rotary_dim = static_cast<int>(cos_c.size(2));
  TORCH_CHECK(rotary_dim % 2 == 0, "rope_inplace needs an even rotary dim");
  TORCH_CHECK(rotary_dim <= head_dim, "rope_inplace rotary dim exceeds head dim");
  TORCH_CHECK(cos_c.size(0) == batch || cos_c.size(0) == 1, "rope_inplace cos batch mismatch");
  TORCH_CHECK(cos_c.size(1) == seq, "rope_inplace cos seq mismatch");

  const int64_t total =
      static_cast<int64_t>(batch) * (heads_q + heads_k) * seq * (rotary_dim / 2);
  if (total == 0) {
    return;
  }
  const int threads = 256;
  const int64_t blocks_needed = (total + threads - 1) / threads;
  const int blocks = static_cast<int>(blocks_needed < 8192 ? blocks_needed : 8192);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t cos_stride_b = cos_c.size(0) == 1 ? 0 : cos_c.stride(0);

  AT_DISPATCH_SWITCH(
      q.scalar_type(),
      "rope_inplace_cuda",
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        using T = at::BFloat16;
        auto launch = [&](auto tag) {
          constexpr bool kInter = decltype(tag)::value;
          rope_inplace_kernel<T, kInter><<<blocks, threads, 0, stream>>>(
              q.data_ptr<T>(), k.data_ptr<T>(), cos_c.data_ptr<T>(), sin_c.data_ptr<T>(),
              heads_q, heads_k, seq, head_dim, rotary_dim,
              q.stride(0), q.stride(1), q.stride(2),
              k.stride(0), k.stride(1), k.stride(2),
              cos_stride_b, cos_c.stride(1), total);
        };
        if (interleaved) {
          launch(std::true_type{});
        } else {
          launch(std::false_type{});
        }
      })
      AT_DISPATCH_CASE(at::kHalf, [&] {
        using T = at::Half;
        auto launch = [&](auto tag) {
          constexpr bool kInter = decltype(tag)::value;
          rope_inplace_kernel<T, kInter><<<blocks, threads, 0, stream>>>(
              q.data_ptr<T>(), k.data_ptr<T>(), cos_c.data_ptr<T>(), sin_c.data_ptr<T>(),
              heads_q, heads_k, seq, head_dim, rotary_dim,
              q.stride(0), q.stride(1), q.stride(2),
              k.stride(0), k.stride(1), k.stride(2),
              cos_stride_b, cos_c.stride(1), total);
        };
        if (interleaved) {
          launch(std::true_type{});
        } else {
          launch(std::false_type{});
        }
      })
      AT_DISPATCH_CASE(at::kFloat, [&] {
        using T = float;
        auto launch = [&](auto tag) {
          constexpr bool kInter = decltype(tag)::value;
          rope_inplace_kernel<T, kInter><<<blocks, threads, 0, stream>>>(
              q.data_ptr<T>(), k.data_ptr<T>(), cos_c.data_ptr<T>(), sin_c.data_ptr<T>(),
              heads_q, heads_k, seq, head_dim, rotary_dim,
              q.stride(0), q.stride(1), q.stride(2),
              k.stride(0), k.stride(1), k.stride(2),
              cos_stride_b, cos_c.stride(1), total);
        };
        if (interleaved) {
          launch(std::true_type{});
        } else {
          launch(std::false_type{});
        }
      }));
}

}  // namespace infer
