// Single-query attention straight out of the KV cache (flash-decode).
//
// Decode reads the whole cache to produce one token, so the step costs exactly
// what the cache weighs: 2 * L * n_kv * head_dim * sizeof(T) bytes. SDPA gets
// close on the read, but it pays for things a decode step does not need — a
// materialized [1, nq, 1, L] mask, a GQA expansion of K/V when the backend
// refuses enable_gqa, and a kernel that is written for many query rows.
//
// This kernel keeps the query in registers, streams K and V once, and carries
// an online softmax so nothing of length L is ever written out. Long caches are
// split across blocks (flash-decoding's split-K) and merged by a second pass,
// because one block per (batch, head) leaves most of GB10's SMs idle.
//
// Supported in-kernel, because decode models actually use them: GQA, sliding
// windows, logit soft-capping, attention sinks, and a key-validity mask (the
// CUDA-graph path reads a fixed window that runs past the live length).

#include "common.cuh"

namespace infer {
namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
// head_dim <= 256, held as ceil(256/32) floats per lane.
constexpr int kMaxPerLane = 8;

template <typename T>
__device__ __forceinline__ float load_as_float(const T* p) {
  return static_cast<float>(*p);
}

struct Strides {
  int64_t b, h, s;
};

// Every lane owns dims {lane, lane + 32, ...}: for one i the warp covers 32
// consecutive dims, so a K row arrives in fully coalesced 64-byte chunks.
template <typename T>
__global__ void attn_decode_split_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    const bool* __restrict__ kv_mask,
    float* __restrict__ part_out,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    Strides qs,
    Strides ks,
    Strides vs,
    int head_dim,
    int kv_len,
    int group,
    int splits,
    int chunk,
    int window,
    float scale,
    float softcap) {
  const int head = blockIdx.x;
  const int split = blockIdx.y;
  const int batch = blockIdx.z;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x >> 5;
  const int kv_head = head / group;

  extern __shared__ float smem[];
  float* s_acc = smem;                                  // [warps][head_dim]
  float* s_m = smem + kWarpsPerBlock * head_dim;        // [warps]
  float* s_l = s_m + kWarpsPerBlock;                    // [warps]

  const T* qp = q + batch * qs.b + head * qs.h;
  float q_reg[kMaxPerLane];
  int held = 0;
  for (int i = lane; i < head_dim; i += kWarpSize) {
    q_reg[held++] = load_as_float(qp + i);
  }

  int begin = split * chunk;
  int end = min(begin + chunk, kv_len);
  if (window > 0) {
    // A windowed layer keeps the last `window` keys; the query is the newest.
    begin = max(begin, max(0, kv_len - window));
  }

  float m = -INFINITY;
  float l = 0.0f;
  float acc[kMaxPerLane];
#pragma unroll
  for (int i = 0; i < kMaxPerLane; ++i) {
    acc[i] = 0.0f;
  }

  for (int t = begin + warp; t < end; t += kWarpsPerBlock) {
    if (kv_mask != nullptr && !kv_mask[batch * kv_len + t]) {
      continue;
    }
    const T* kp = k + batch * ks.b + kv_head * ks.h + static_cast<int64_t>(t) * ks.s;
    float dot = 0.0f;
    int c = 0;
    for (int i = lane; i < head_dim; i += kWarpSize) {
      dot += q_reg[c++] * load_as_float(kp + i);
    }
    dot = warp_reduce_sum(dot) * scale;
    if (softcap > 0.0f) {
      dot = tanhf(dot / softcap) * softcap;
    }
    const float m_new = fmaxf(m, dot);
    const float corr = (m == -INFINITY) ? 0.0f : expf(m - m_new);
    const float p = expf(dot - m_new);
    l = l * corr + p;
    const T* vp = v + batch * vs.b + kv_head * vs.h + static_cast<int64_t>(t) * vs.s;
    c = 0;
    for (int i = lane; i < head_dim; i += kWarpSize) {
      acc[c] = acc[c] * corr + p * load_as_float(vp + i);
      ++c;
    }
    m = m_new;
  }

  int c = 0;
  for (int i = lane; i < head_dim; i += kWarpSize) {
    s_acc[warp * head_dim + i] = acc[c++];
  }
  if (lane == 0) {
    s_m[warp] = m;
    s_l[warp] = l;
  }
  __syncthreads();

  if (warp != 0) {
    return;
  }
  float m_all = -INFINITY;
  for (int w = 0; w < kWarpsPerBlock; ++w) {
    m_all = fmaxf(m_all, s_m[w]);
  }
  const int64_t base = (static_cast<int64_t>(batch) * gridDim.x + head) * splits + split;
  if (m_all == -INFINITY) {
    // Nothing survived the window or the mask; a zero-weight partial.
    for (int i = lane; i < head_dim; i += kWarpSize) {
      part_out[base * head_dim + i] = 0.0f;
    }
    if (lane == 0) {
      part_m[base] = -INFINITY;
      part_l[base] = 0.0f;
    }
    return;
  }
  float l_all = 0.0f;
  for (int w = 0; w < kWarpsPerBlock; ++w) {
    l_all += s_l[w] * ((s_m[w] == -INFINITY) ? 0.0f : expf(s_m[w] - m_all));
  }
  for (int i = lane; i < head_dim; i += kWarpSize) {
    float o = 0.0f;
    for (int w = 0; w < kWarpsPerBlock; ++w) {
      const float w_scale = (s_m[w] == -INFINITY) ? 0.0f : expf(s_m[w] - m_all);
      o += s_acc[w * head_dim + i] * w_scale;
    }
    part_out[base * head_dim + i] = o;
  }
  if (lane == 0) {
    part_m[base] = m_all;
    part_l[base] = l_all;
  }
}

template <typename T>
__global__ void attn_decode_combine_kernel(
    const float* __restrict__ part_out,
    const float* __restrict__ part_m,
    const float* __restrict__ part_l,
    const float* __restrict__ sinks,
    T* __restrict__ out,
    Strides os,
    int head_dim,
    int splits) {
  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const int64_t base = (static_cast<int64_t>(batch) * gridDim.x + head) * splits;

  float m = -INFINITY;
  for (int s = 0; s < splits; ++s) {
    m = fmaxf(m, part_m[base + s]);
  }
  // An attention sink is one extra logit with no value vector: it lands in the
  // denominator only, which an online softmax can absorb after the fact.
  const float sink = (sinks != nullptr) ? sinks[head] : -INFINITY;
  if (sinks != nullptr) {
    m = fmaxf(m, sink);
  }
  T* op = out + batch * os.b + head * os.h;
  if (m == -INFINITY) {
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      op[i] = static_cast<T>(0.0f);
    }
    return;
  }
  float l = 0.0f;
  for (int s = 0; s < splits; ++s) {
    const float ms = part_m[base + s];
    l += part_l[base + s] * ((ms == -INFINITY) ? 0.0f : expf(ms - m));
  }
  if (sinks != nullptr) {
    l += expf(sink - m);
  }
  const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    float o = 0.0f;
    for (int s = 0; s < splits; ++s) {
      const float ms = part_m[base + s];
      const float w = (ms == -INFINITY) ? 0.0f : expf(ms - m);
      o += part_out[(base + s) * head_dim + i] * w;
    }
    op[i] = static_cast<T>(o * inv);
  }
}

int pick_splits(int64_t kv_len, int64_t independent) {
  // Enough blocks to cover the SMs a few times over, but never so many that a
  // block reads fewer keys than a warp can amortize.
  const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  const int64_t want = std::max<int64_t>(1, (2 * sms) / std::max<int64_t>(independent, 1));
  const int64_t by_len = (kv_len + 63) / 64;
  return static_cast<int>(std::max<int64_t>(1, std::min(want, by_len)));
}

}  // namespace

// q: [B, nq, D] (one query position), k/v: [B, n_kv, L, D] with a contiguous
// last dim. Returns [B, nq, D].
at::Tensor attn_decode_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& kv_mask,
    const c10::optional<at::Tensor>& sinks,
    double scale,
    int64_t window,
    double softcap) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "attn_decode expects CUDA tensors");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && v.dim() == 4, "attn_decode shape");
  TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1,
              "attn_decode needs a contiguous head dim");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && k.scalar_type() == v.scalar_type(),
              "attn_decode dtype mismatch");

  const int64_t batch = q.size(0);
  const int64_t heads = q.size(1);
  const int64_t head_dim = q.size(2);
  const int64_t kv_heads = k.size(1);
  const int64_t kv_len = k.size(2);
  TORCH_CHECK(head_dim <= kMaxPerLane * kWarpSize, "attn_decode head_dim > 256");
  TORCH_CHECK(v.size(3) == head_dim, "attn_decode needs matching k/v head dims");
  TORCH_CHECK(heads % kv_heads == 0, "attn_decode GQA group");
  TORCH_CHECK(v.size(2) == kv_len, "attn_decode k/v length mismatch");

  const at::cuda::CUDAGuard guard(q.device());
  auto out = at::empty({batch, heads, head_dim}, q.options());
  if (kv_len == 0) {
    out.zero_();
    return out;
  }

  const int splits = pick_splits(kv_len, batch * heads);
  const int chunk = static_cast<int>((kv_len + splits - 1) / splits);
  auto fopts = q.options().dtype(at::kFloat);
  auto part_out = at::empty({batch, heads, splits, head_dim}, fopts);
  auto part_m = at::empty({batch, heads, splits}, fopts);
  auto part_l = at::empty({batch, heads, splits}, fopts);

  const bool* mask_ptr = nullptr;
  if (kv_mask.has_value() && kv_mask->defined()) {
    TORCH_CHECK(kv_mask->scalar_type() == at::kBool, "kv_mask must be bool");
    TORCH_CHECK(kv_mask->size(-1) == kv_len, "kv_mask length mismatch");
    TORCH_CHECK(kv_mask->is_contiguous(), "kv_mask must be contiguous");
    mask_ptr = kv_mask->data_ptr<bool>();
  }
  const float* sink_ptr = nullptr;
  at::Tensor sink_f;
  if (sinks.has_value() && sinks->defined()) {
    sink_f = sinks->to(at::kFloat).contiguous();
    TORCH_CHECK(sink_f.numel() == heads, "one sink per query head");
    sink_ptr = sink_f.data_ptr<float>();
  }

  const Strides qs{q.stride(0), q.stride(1), 0};
  const Strides ks{k.stride(0), k.stride(1), k.stride(2)};
  const Strides vs{v.stride(0), v.stride(1), v.stride(2)};
  const Strides os{out.stride(0), out.stride(1), 0};
  const int group = static_cast<int>(heads / kv_heads);
  const dim3 grid(heads, splits, batch);
  const size_t smem = sizeof(float) * (kWarpsPerBlock * head_dim + 2 * kWarpsPerBlock);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_SWITCH(
      q.scalar_type(), "attn_decode",
      AT_DISPATCH_CASE(at::kBFloat16, [&] {
        using T = at::BFloat16;
        attn_decode_split_kernel<T><<<grid, kThreadsPerBlock, smem, stream>>>(
            q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(), mask_ptr,
            part_out.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
            qs, ks, vs, head_dim, kv_len, group, splits, chunk, window, scale, softcap);
        attn_decode_combine_kernel<T><<<dim3(heads, batch), kThreadsPerBlock, 0, stream>>>(
            part_out.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
            sink_ptr, out.data_ptr<T>(), os, head_dim, splits);
      })
      AT_DISPATCH_CASE(at::kHalf, [&] {
        using T = at::Half;
        attn_decode_split_kernel<T><<<grid, kThreadsPerBlock, smem, stream>>>(
            q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(), mask_ptr,
            part_out.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
            qs, ks, vs, head_dim, kv_len, group, splits, chunk, window, scale, softcap);
        attn_decode_combine_kernel<T><<<dim3(heads, batch), kThreadsPerBlock, 0, stream>>>(
            part_out.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
            sink_ptr, out.data_ptr<T>(), os, head_dim, splits);
      })
      AT_DISPATCH_CASE(at::kFloat, [&] {
        using T = float;
        attn_decode_split_kernel<T><<<grid, kThreadsPerBlock, smem, stream>>>(
            q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(), mask_ptr,
            part_out.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
            qs, ks, vs, head_dim, kv_len, group, splits, chunk, window, scale, softcap);
        attn_decode_combine_kernel<T><<<dim3(heads, batch), kThreadsPerBlock, 0, stream>>>(
            part_out.data_ptr<float>(), part_m.data_ptr<float>(), part_l.data_ptr<float>(),
            sink_ptr, out.data_ptr<T>(), os, head_dim, splits);
      }));
  return out;
}

}  // namespace infer
