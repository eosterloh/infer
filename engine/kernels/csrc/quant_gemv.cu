// Weight-only quantized GEMV for decode: INT4 group-wise, NVFP4 (E2M1 + FP8
// block scales), and FP8 E4M3. Decode at batch 1 is a pure bandwidth problem,
// so the win comes from never materializing the BF16 weight: each warp streams
// its own output row's packed bytes and dequantizes in registers.
#include "common.cuh"

namespace infer {
namespace {

constexpr int kQuantInt4 = 0;
constexpr int kQuantNvfp4 = 1;
constexpr int kQuantFp8 = 2;

// Weights handled per lane per iteration. 32 nibbles is one 16-byte load, the
// widest the hardware offers, and 32 lanes then cover 1024 contiguous weights.
constexpr int kPerLane = 32;

__device__ __forceinline__ float fp8_e4m3_to_float(uint32_t byte) {
  const uint32_t sign = byte & 0x80u;
  const uint32_t exp = (byte >> 3) & 0x0Fu;
  const uint32_t man = byte & 0x07u;
  if (exp == 0u) {
    // Subnormal: man * 2^-9, exactly representable.
    const float mag = static_cast<float>(man) * 0.001953125f;
    return sign ? -mag : mag;
  }
  const uint32_t bits = (sign << 24) | ((exp + 120u) << 23) | (man << 20);
  return __int_as_float(bits);
}

// E2M1: 1 sign, 2 exponent (bias 1), 1 mantissa → {0, .5, 1, 1.5, 2, 3, 4, 6}.
__device__ __forceinline__ float fp4_e2m1_to_float(uint32_t code) {
  const uint32_t mag = code & 0x07u;
  const uint32_t exp = mag >> 1;
  const float man = static_cast<float>(mag & 1u);
  const float value =
      (exp == 0u) ? (0.5f * man)
                  : ((1.0f + 0.5f * man) * static_cast<float>(1u << (exp - 1u)));
  return (code & 0x08u) ? -value : value;
}

template <typename scalar_t>
__device__ __forceinline__ float to_f(scalar_t v) {
  return static_cast<float>(v);
}

// Dequantize one lane's kPerLane weights of row `row` starting at column `k0`.
// Shared by the dense and the grouped (MoE) GEMV so both read bits identically.
template <typename scalar_t, int KIND>
__device__ __forceinline__ void dequant_lane(
    const uint8_t* __restrict__ wrow,
    int k0,
    int row,
    const scalar_t* __restrict__ scales,
    const uint8_t* __restrict__ scales_u8,
    const scalar_t* __restrict__ zeros,
    const float* __restrict__ channel_scale,
    float global_scale,
    int group_size,
    int groups_per_row,
    float (&wq)[kPerLane],
    float& scale_a,
    float& scale_b,
    float& zero) {
  scale_a = 1.0f;
  scale_b = 1.0f;
  zero = 0.0f;
  if (KIND == kQuantFp8) {
    // One byte per weight, so kPerLane weights are two 16-byte loads.
    const uint4 lo = *reinterpret_cast<const uint4*>(wrow + k0);
    const uint4 hi = *reinterpret_cast<const uint4*>(wrow + k0 + 16);
    const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
    for (int w = 0; w < 8; ++w) {
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        wq[w * 4 + b] = fp8_e4m3_to_float((words[w] >> (b * 8)) & 0xFFu);
      }
    }
    // FP8 rows carry one scale each (per-channel) or one for the tensor.
    scale_a = scale_b = channel_scale ? channel_scale[row] : global_scale;
    return;
  }
  const uint4 raw = *reinterpret_cast<const uint4*>(wrow + (k0 >> 1));
  const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
  if (KIND == kQuantInt4) {
#pragma unroll
    for (int w = 0; w < 4; ++w) {
#pragma unroll
      for (int nib = 0; nib < 8; ++nib) {
        wq[w * 8 + nib] = static_cast<float>((words[w] >> (nib * 4)) & 0xFu);
      }
    }
    const int64_t g = static_cast<int64_t>(row) * groups_per_row + k0 / group_size;
    scale_a = scale_b = to_f(scales[g]);
    zero = zeros ? to_f(zeros[g]) : 0.0f;
    return;
  }
#pragma unroll
  for (int w = 0; w < 4; ++w) {
#pragma unroll
    for (int nib = 0; nib < 8; ++nib) {
      wq[w * 8 + nib] = fp4_e2m1_to_float((words[w] >> (nib * 4)) & 0xFu);
    }
  }
  // NVFP4 keeps a 16-element block scale in FP8 plus one FP32 global. A stack
  // of concatenated experts carries one global per row instead, which is what
  // lets independently packed blocks be joined without re-encoding.
  const int64_t base = static_cast<int64_t>(row) * groups_per_row + k0 / group_size;
  const float gs = channel_scale ? channel_scale[row] : global_scale;
  scale_a = fp8_e4m3_to_float(scales_u8[base]) * gs;
  scale_b =
      (group_size < kPerLane) ? fp8_e4m3_to_float(scales_u8[base + 1]) * gs : scale_a;
}

// Dot kPerLane dequantized weights against one activation slice. NVFP4 blocks
// are 16 wide, so a lane spans two of them; the halves are a compile-time split
// to keep the inner loop branch free.
template <typename scalar_t, bool kSplit>
__device__ __forceinline__ float lane_dot(
    const scalar_t* __restrict__ xr,
    const float (&wq)[kPerLane],
    float scale_a,
    float scale_b,
    float zero) {
  float dot_a = 0.0f;
  float dot_b = 0.0f;
  float xsum = 0.0f;
#pragma unroll
  for (int j = 0; j < kPerLane / 8; ++j) {
    const Vec<scalar_t, 8> xv = *reinterpret_cast<const Vec<scalar_t, 8>*>(xr + j * 8);
    float part = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float xf = to_f(xv.v[i]);
      part += wq[j * 8 + i] * xf;
      xsum += xf;
    }
    if (kSplit && j >= (kPerLane / 16)) {
      dot_b += part;
    } else {
      dot_a += part;
    }
  }
  return scale_a * dot_a + (kSplit ? scale_b * dot_b : 0.0f) + zero * xsum;
}

// One warp per output row. MMAX is the compile-time cap on rows of x handled in
// a single pass; holding several activation rows against one weight load is
// what keeps short prefills and speculative batches off the dequantize path.
template <typename scalar_t, int KIND, int MMAX>
__global__ void qgemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    const scalar_t* __restrict__ scales,
    const uint8_t* __restrict__ scales_u8,
    const scalar_t* __restrict__ zeros,
    const float* __restrict__ channel_scale,
    float global_scale,
    scalar_t* __restrict__ out,
    int m_rows,
    int n_cols,
    int k_dim,
    int group_size,
    int groups_per_row) {
  const int warps = blockDim.x >> 5;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int row = blockIdx.x * warps + (threadIdx.x >> 5);
  if (row >= n_cols) {
    return;
  }

  float acc[MMAX];
#pragma unroll
  for (int i = 0; i < MMAX; ++i) {
    acc[i] = 0.0f;
  }

  const int64_t row_bytes = (KIND == kQuantFp8) ? k_dim : (k_dim / 2);
  const uint8_t* wrow = qweight + static_cast<int64_t>(row) * row_bytes;
  const int stride = kWarpSize * kPerLane;

  for (int k0 = lane * kPerLane; k0 < k_dim; k0 += stride) {
    float wq[kPerLane];
    float scale_a, scale_b, zero;
    dequant_lane<scalar_t, KIND>(
        wrow, k0, row, scales, scales_u8, zeros, channel_scale, global_scale, group_size,
        groups_per_row, wq, scale_a, scale_b, zero);

    constexpr bool kSplit = (KIND == kQuantNvfp4);
    for (int mi = 0; mi < MMAX; ++mi) {
      if (mi >= m_rows) {
        break;
      }
      acc[mi] += lane_dot<scalar_t, kSplit>(
          x + static_cast<int64_t>(mi) * k_dim + k0, wq, scale_a, scale_b, zero);
    }
  }

#pragma unroll
  for (int mi = 0; mi < MMAX; ++mi) {
    if (mi >= m_rows) {
      break;
    }
    const float total = warp_reduce_sum(acc[mi]);
    if (lane == 0) {
      out[static_cast<int64_t>(mi) * n_cols + row] = static_cast<scalar_t>(total);
    }
  }
}

// Grouped GEMV over a stack of experts packed as one [E * N, K] weight. One
// warp per (routed row, output column); `row_expert` and `row_input` keep MoE
// routing on the device, so an activated expert costs nothing on the host.
template <typename scalar_t, int KIND>
__global__ void qmoe_gemv_kernel(
    const scalar_t* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    const scalar_t* __restrict__ scales,
    const uint8_t* __restrict__ scales_u8,
    const scalar_t* __restrict__ zeros,
    const float* __restrict__ channel_scale,
    const int32_t* __restrict__ row_expert,
    const int32_t* __restrict__ row_input,
    float global_scale,
    scalar_t* __restrict__ out,
    int expert_cols,
    int k_dim,
    int group_size,
    int groups_per_row) {
  const int warps = blockDim.x >> 5;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int col = blockIdx.x * warps + (threadIdx.x >> 5);
  const int m = blockIdx.y;
  if (col >= expert_cols) {
    return;
  }
  const int row = row_expert[m] * expert_cols + col;
  const int x_row = row_input ? row_input[m] : m;

  const int64_t row_bytes = (KIND == kQuantFp8) ? k_dim : (k_dim / 2);
  const uint8_t* wrow = qweight + static_cast<int64_t>(row) * row_bytes;
  const scalar_t* xr = x + static_cast<int64_t>(x_row) * k_dim;
  const int stride = kWarpSize * kPerLane;
  constexpr bool kSplit = (KIND == kQuantNvfp4);

  float acc = 0.0f;
  for (int k0 = lane * kPerLane; k0 < k_dim; k0 += stride) {
    float wq[kPerLane];
    float scale_a, scale_b, zero;
    dequant_lane<scalar_t, KIND>(
        wrow, k0, row, scales, scales_u8, zeros, channel_scale, global_scale, group_size,
        groups_per_row, wq, scale_a, scale_b, zero);
    acc += lane_dot<scalar_t, kSplit>(xr + k0, wq, scale_a, scale_b, zero);
  }
  const float total = warp_reduce_sum(acc);
  if (lane == 0) {
    out[static_cast<int64_t>(m) * expert_cols + col] = static_cast<scalar_t>(total);
  }
}

// Straight unpack to the compute dtype, used for prefill (hand the result to
// cuBLAS) and as the parity reference.
//
// This runs over every weight of every packed layer on every prefill, so it is
// the entire cost of a quantized prefill beyond the GEMM itself. Eight elements
// per thread: one packed word in, one 16-byte vector out, and the row index
// comes from blockIdx.y instead of a 64-bit division of the flat index — that
// division alone cost more than the conversion it indexed.
template <typename scalar_t, int KIND>
__global__ void dequant_vec_kernel(
    const uint8_t* __restrict__ qweight,
    const scalar_t* __restrict__ scales,
    const uint8_t* __restrict__ scales_u8,
    const scalar_t* __restrict__ zeros,
    const float* __restrict__ channel_scale,
    float global_scale,
    scalar_t* __restrict__ out,
    int n_cols,
    int k_dim,
    int group_size,
    int group_shift,
    int groups_per_row) {
  constexpr int kPer = 8;
  const int chunks_per_row = k_dim / kPer;
  const int chunk = blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= chunks_per_row) {
    return;
  }
  const int col = chunk * kPer;

  for (int64_t row = blockIdx.y; row < n_cols; row += gridDim.y) {
    const int64_t flat = row * k_dim + col;
    float value[kPer];
    float scale = 1.0f;
    float zero = 0.0f;

    if (KIND == kQuantFp8) {
      const uint2 raw = *reinterpret_cast<const uint2*>(qweight + flat);
      const uint32_t words[2] = {raw.x, raw.y};
#pragma unroll
      for (int w = 0; w < 2; ++w) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          value[w * 4 + b] = fp8_e4m3_to_float((words[w] >> (b * 8)) & 0xFFu);
        }
      }
      scale = channel_scale ? channel_scale[row] : global_scale;
    } else {
      const uint32_t word = *reinterpret_cast<const uint32_t*>(qweight + (flat >> 1));
      // Eight columns starting at a multiple of eight never straddle a group,
      // so one scale covers the whole vector.
      const int64_t g = row * groups_per_row +
                        (group_shift >= 0 ? (col >> group_shift) : (col / group_size));
      if (KIND == kQuantInt4) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
          value[i] = static_cast<float>((word >> (i * 4)) & 0xFu);
        }
        scale = to_f(scales[g]);
        zero = zeros ? to_f(zeros[g]) : 0.0f;
      } else {
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
          value[i] = fp4_e2m1_to_float((word >> (i * 4)) & 0xFu);
        }
        scale = fp8_e4m3_to_float(scales_u8[g]) *
                (channel_scale ? channel_scale[row] : global_scale);
      }
    }

    Vec<scalar_t, kPer> packed;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      packed.v[i] = static_cast<scalar_t>(value[i] * scale + zero);
    }
    *reinterpret_cast<Vec<scalar_t, kPer>*>(out + flat) = packed;
  }
}

template <typename scalar_t, int KIND>
__global__ void dequant_kernel(
    const uint8_t* __restrict__ qweight,
    const scalar_t* __restrict__ scales,
    const uint8_t* __restrict__ scales_u8,
    const scalar_t* __restrict__ zeros,
    const float* __restrict__ channel_scale,
    float global_scale,
    scalar_t* __restrict__ out,
    int n_cols,
    int k_dim,
    int group_size,
    int groups_per_row) {
  const int64_t total = static_cast<int64_t>(n_cols) * k_dim;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       idx < total; idx += stride) {
    const int64_t row = idx / k_dim;
    const int64_t col = idx - row * k_dim;
    float value;
    if (KIND == kQuantFp8) {
      value = fp8_e4m3_to_float(qweight[row * k_dim + col]);
      value *= channel_scale ? channel_scale[row] : global_scale;
    } else {
      const uint8_t byte = qweight[(row * k_dim + col) >> 1];
      const uint32_t nib = (col & 1) ? (byte >> 4) : (byte & 0x0Fu);
      const int64_t g = row * groups_per_row + col / group_size;
      if (KIND == kQuantInt4) {
        value = static_cast<float>(nib) * to_f(scales[g]);
        if (zeros) {
          value += to_f(zeros[g]);
        }
      } else {
        value = fp4_e2m1_to_float(nib) * fp8_e4m3_to_float(scales_u8[g]) *
                (channel_scale ? channel_scale[row] : global_scale);
      }
    }
    out[idx] = static_cast<scalar_t>(value);
  }
}

struct QuantArgs {
  const uint8_t* qweight;
  const uint8_t* scales_u8;
  const float* channel_scale;
  float global_scale;
  int64_t m_rows;
  int64_t n_cols;
  int64_t k_dim;
  int64_t group_size;
  int64_t groups_per_row;
};

template <typename scalar_t, int KIND, int MMAX>
void launch_qgemv_rows(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& scales,
    const c10::optional<at::Tensor>& zeros,
    at::Tensor& out,
    const QuantArgs& a) {
  const scalar_t* scale_t = (KIND == kQuantInt4 && scales.has_value())
                                ? scales->data_ptr<scalar_t>()
                                : nullptr;
  const scalar_t* zero_t =
      (zeros.has_value() && zeros->defined()) ? zeros->data_ptr<scalar_t>() : nullptr;
  const int threads = 256;
  const int warps = threads / kWarpSize;
  const int blocks = static_cast<int>((a.n_cols + warps - 1) / warps);
  qgemv_kernel<scalar_t, KIND, MMAX><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<scalar_t>(), a.qweight, scale_t, a.scales_u8, zero_t, a.channel_scale,
      a.global_scale, out.data_ptr<scalar_t>(), static_cast<int>(a.m_rows),
      static_cast<int>(a.n_cols), static_cast<int>(a.k_dim),
      static_cast<int>(a.group_size), static_cast<int>(a.groups_per_row));
}

// The row cap is a register budget: MMAX accumulators plus kPerLane dequantized
// weights live in registers for the whole row, so a 32-row launch that only
// needs four would halve its occupancy for nothing.
template <typename scalar_t, int KIND>
void launch_qgemv(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& scales,
    const c10::optional<at::Tensor>& zeros,
    at::Tensor& out,
    const QuantArgs& a) {
  if (a.m_rows <= 1) {
    launch_qgemv_rows<scalar_t, KIND, 1>(x, scales, zeros, out, a);
  } else if (a.m_rows <= 4) {
    launch_qgemv_rows<scalar_t, KIND, 4>(x, scales, zeros, out, a);
  } else if (a.m_rows <= 8) {
    launch_qgemv_rows<scalar_t, KIND, 8>(x, scales, zeros, out, a);
  } else if (a.m_rows <= 16) {
    launch_qgemv_rows<scalar_t, KIND, 16>(x, scales, zeros, out, a);
  } else {
    launch_qgemv_rows<scalar_t, KIND, 32>(x, scales, zeros, out, a);
  }
}

template <typename scalar_t>
void launch_qgemv_kind(
    int64_t kind,
    const at::Tensor& x,
    const c10::optional<at::Tensor>& scales,
    const c10::optional<at::Tensor>& zeros,
    at::Tensor& out,
    const QuantArgs& a) {
  if (kind == kQuantInt4) {
    launch_qgemv<scalar_t, kQuantInt4>(x, scales, zeros, out, a);
  } else if (kind == kQuantNvfp4) {
    launch_qgemv<scalar_t, kQuantNvfp4>(x, scales, zeros, out, a);
  } else {
    launch_qgemv<scalar_t, kQuantFp8>(x, scales, zeros, out, a);
  }
}

template <typename scalar_t, int KIND>
void launch_qmoe(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& scales,
    const c10::optional<at::Tensor>& zeros,
    const at::Tensor& row_expert,
    const c10::optional<at::Tensor>& row_input,
    at::Tensor& out,
    const QuantArgs& a) {
  const scalar_t* scale_t = (KIND == kQuantInt4 && scales.has_value())
                                ? scales->data_ptr<scalar_t>()
                                : nullptr;
  const scalar_t* zero_t =
      (zeros.has_value() && zeros->defined()) ? zeros->data_ptr<scalar_t>() : nullptr;
  const int threads = 256;
  const int warps = threads / kWarpSize;
  const dim3 blocks(
      static_cast<unsigned>((a.n_cols + warps - 1) / warps),
      static_cast<unsigned>(a.m_rows));
  qmoe_gemv_kernel<scalar_t, KIND><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<scalar_t>(), a.qweight, scale_t, a.scales_u8, zero_t, a.channel_scale,
      row_expert.data_ptr<int32_t>(),
      (row_input.has_value() && row_input->defined()) ? row_input->data_ptr<int32_t>()
                                                      : nullptr,
      a.global_scale, out.data_ptr<scalar_t>(), static_cast<int>(a.n_cols),
      static_cast<int>(a.k_dim), static_cast<int>(a.group_size),
      static_cast<int>(a.groups_per_row));
}

template <typename scalar_t>
void launch_qmoe_kind(
    int64_t kind,
    const at::Tensor& x,
    const c10::optional<at::Tensor>& scales,
    const c10::optional<at::Tensor>& zeros,
    const at::Tensor& row_expert,
    const c10::optional<at::Tensor>& row_input,
    at::Tensor& out,
    const QuantArgs& a) {
  if (kind == kQuantInt4) {
    launch_qmoe<scalar_t, kQuantInt4>(x, scales, zeros, row_expert, row_input, out, a);
  } else if (kind == kQuantNvfp4) {
    launch_qmoe<scalar_t, kQuantNvfp4>(x, scales, zeros, row_expert, row_input, out, a);
  } else {
    launch_qmoe<scalar_t, kQuantFp8>(x, scales, zeros, row_expert, row_input, out, a);
  }
}

template <typename scalar_t, int KIND>
void launch_dequant(
    const c10::optional<at::Tensor>& scales,
    const c10::optional<at::Tensor>& zeros,
    at::Tensor& out,
    const QuantArgs& a) {
  const scalar_t* scale_t = (KIND == kQuantInt4 && scales.has_value())
                                ? scales->data_ptr<scalar_t>()
                                : nullptr;
  const scalar_t* zero_t =
      (zeros.has_value() && zeros->defined()) ? zeros->data_ptr<scalar_t>() : nullptr;
  const int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();

  // Vector path: eight columns per thread, which needs a width that divides by
  // eight and groups that do not split a vector.
  const bool use_vec =
      a.k_dim % 8 == 0 && (KIND == kQuantFp8 || a.group_size % 8 == 0);
  if (use_vec) {
    int shift = -1;
    for (int bit = 0; bit < 31; ++bit) {
      if ((int64_t{1} << bit) == a.group_size) {
        shift = bit;
        break;
      }
    }
    const int64_t chunks = a.k_dim / 8;
    const dim3 grid(
        static_cast<unsigned>((chunks + threads - 1) / threads),
        static_cast<unsigned>(std::min<int64_t>(a.n_cols, 32768)));
    dequant_vec_kernel<scalar_t, KIND><<<grid, threads, 0, stream>>>(
        a.qweight, scale_t, a.scales_u8, zero_t, a.channel_scale, a.global_scale,
        out.data_ptr<scalar_t>(), static_cast<int>(a.n_cols), static_cast<int>(a.k_dim),
        static_cast<int>(a.group_size), shift, static_cast<int>(a.groups_per_row));
    return;
  }

  const int64_t total = a.n_cols * a.k_dim;
  const int blocks =
      static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 8192));
  dequant_kernel<scalar_t, KIND><<<blocks, threads, 0, stream>>>(
      a.qweight, scale_t, a.scales_u8, zero_t, a.channel_scale, a.global_scale,
      out.data_ptr<scalar_t>(), static_cast<int>(a.n_cols), static_cast<int>(a.k_dim),
      static_cast<int>(a.group_size), static_cast<int>(a.groups_per_row));
}

template <typename scalar_t>
void launch_dequant_kind(
    int64_t kind,
    const c10::optional<at::Tensor>& scales,
    const c10::optional<at::Tensor>& zeros,
    at::Tensor& out,
    const QuantArgs& a) {
  if (kind == kQuantInt4) {
    launch_dequant<scalar_t, kQuantInt4>(scales, zeros, out, a);
  } else if (kind == kQuantNvfp4) {
    launch_dequant<scalar_t, kQuantNvfp4>(scales, zeros, out, a);
  } else {
    launch_dequant<scalar_t, kQuantFp8>(scales, zeros, out, a);
  }
}

}  // namespace

at::Tensor qgemv_cuda(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const c10::optional<at::Tensor>& scales,
    const c10::optional<at::Tensor>& zeros,
    const c10::optional<at::Tensor>& channel_scale,
    int64_t kind,
    int64_t group_size,
    int64_t n_cols,
    int64_t k_dim,
    double global_scale) {
  TORCH_CHECK(x.is_cuda() && qweight.is_cuda(), "qgemv expects CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
              "qgemv activations must be bf16 or fp16");
  TORCH_CHECK(x.size(-1) == k_dim, "activation width does not match k");
  TORCH_CHECK(k_dim % 64 == 0, "qgemv needs k divisible by 64");

  const at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  auto x_c = x.contiguous().view({-1, k_dim});
  const int64_t m_rows = x_c.size(0);
  TORCH_CHECK(m_rows <= 32, "qgemv fast path handles at most 32 rows");

  auto out_shape = x.sizes().vec();
  out_shape.back() = n_cols;
  auto out = at::empty({m_rows, n_cols}, x_c.options());

  QuantArgs args;
  args.qweight = reinterpret_cast<const uint8_t*>(qweight.data_ptr());
  args.scales_u8 = nullptr;
  if (kind == kQuantNvfp4) {
    TORCH_CHECK(scales.has_value(), "nvfp4 needs block scales");
    // The reinterpret_cast below bypasses the dtype check data_ptr<uint8_t>()
    // would have made, so the bytes have to be checked here.
    TORCH_CHECK(scales->scalar_type() == at::kByte, "nvfp4 block scales must be uint8");
    args.scales_u8 = reinterpret_cast<const uint8_t*>(scales->data_ptr());
  }
  args.channel_scale = (channel_scale.has_value() && channel_scale->defined())
                           ? channel_scale->data_ptr<float>()
                           : nullptr;
  args.global_scale = static_cast<float>(global_scale);
  args.m_rows = m_rows;
  args.n_cols = n_cols;
  args.k_dim = k_dim;
  args.group_size = group_size;
  args.groups_per_row =
      (kind == kQuantFp8) ? 1 : (k_dim / (group_size > 0 ? group_size : k_dim));

  if (x.scalar_type() == at::kBFloat16) {
    launch_qgemv_kind<at::BFloat16>(kind, x_c, scales, zeros, out, args);
  } else {
    launch_qgemv_kind<at::Half>(kind, x_c, scales, zeros, out, args);
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return out.view(out_shape);
}

at::Tensor qmoe_gemv_cuda(
    const at::Tensor& x,
    const at::Tensor& qweight,
    const c10::optional<at::Tensor>& scales,
    const c10::optional<at::Tensor>& zeros,
    const c10::optional<at::Tensor>& channel_scale,
    const at::Tensor& row_expert,
    const c10::optional<at::Tensor>& row_input,
    int64_t kind,
    int64_t group_size,
    int64_t expert_cols,
    int64_t k_dim,
    double global_scale) {
  TORCH_CHECK(x.is_cuda() && qweight.is_cuda(), "qmoe_gemv expects CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == k_dim, "x must be [T, K]");
  TORCH_CHECK(k_dim % 64 == 0, "qmoe_gemv needs k divisible by 64");
  TORCH_CHECK(row_expert.scalar_type() == at::kInt, "row_expert must be int32");

  const at::cuda::OptionalCUDAGuard guard(at::device_of(x));
  auto x_c = x.contiguous();
  const int64_t m_rows = row_expert.numel();
  // One routed row per gridDim.y, which the hardware caps at 65535.
  TORCH_CHECK(m_rows <= 65535, "qmoe_gemv handles at most 65535 routed rows, got ", m_rows);
  auto out = at::empty({m_rows, expert_cols}, x_c.options());
  if (m_rows == 0) {
    return out;
  }

  QuantArgs args;
  args.qweight = reinterpret_cast<const uint8_t*>(qweight.data_ptr());
  args.scales_u8 = nullptr;
  if (kind == kQuantNvfp4) {
    TORCH_CHECK(scales.has_value(), "nvfp4 needs block scales");
    TORCH_CHECK(scales->scalar_type() == at::kByte, "nvfp4 block scales must be uint8");
    args.scales_u8 = reinterpret_cast<const uint8_t*>(scales->data_ptr());
  }
  args.channel_scale = (channel_scale.has_value() && channel_scale->defined())
                           ? channel_scale->data_ptr<float>()
                           : nullptr;
  args.global_scale = static_cast<float>(global_scale);
  args.m_rows = m_rows;
  args.n_cols = expert_cols;
  args.k_dim = k_dim;
  args.group_size = group_size;
  args.groups_per_row =
      (kind == kQuantFp8) ? 1 : (k_dim / (group_size > 0 ? group_size : k_dim));

  if (x.scalar_type() == at::kBFloat16) {
    launch_qmoe_kind<at::BFloat16>(
        kind, x_c, scales, zeros, row_expert, row_input, out, args);
  } else {
    launch_qmoe_kind<at::Half>(kind, x_c, scales, zeros, row_expert, row_input, out, args);
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

at::Tensor dequant_cuda(
    const at::Tensor& qweight,
    const c10::optional<at::Tensor>& scales,
    const c10::optional<at::Tensor>& zeros,
    const c10::optional<at::Tensor>& channel_scale,
    int64_t kind,
    int64_t group_size,
    int64_t n_cols,
    int64_t k_dim,
    double global_scale,
    at::ScalarType dtype) {
  TORCH_CHECK(qweight.is_cuda(), "dequant expects CUDA tensors");
  const at::cuda::OptionalCUDAGuard guard(at::device_of(qweight));
  auto out = at::empty({n_cols, k_dim}, qweight.options().dtype(dtype));

  QuantArgs args;
  args.qweight = reinterpret_cast<const uint8_t*>(qweight.data_ptr());
  args.scales_u8 = nullptr;
  if (kind == kQuantNvfp4) {
    TORCH_CHECK(scales.has_value(), "nvfp4 needs block scales");
    TORCH_CHECK(scales->scalar_type() == at::kByte, "nvfp4 block scales must be uint8");
    args.scales_u8 = reinterpret_cast<const uint8_t*>(scales->data_ptr());
  }
  args.channel_scale = (channel_scale.has_value() && channel_scale->defined())
                           ? channel_scale->data_ptr<float>()
                           : nullptr;
  args.global_scale = static_cast<float>(global_scale);
  args.m_rows = 1;
  args.n_cols = n_cols;
  args.k_dim = k_dim;
  args.group_size = group_size;
  args.groups_per_row =
      (kind == kQuantFp8) ? 1 : (k_dim / (group_size > 0 ? group_size : k_dim));

  if (dtype == at::kBFloat16) {
    launch_dequant_kind<at::BFloat16>(kind, scales, zeros, out, args);
  } else {
    launch_dequant_kind<at::Half>(kind, scales, zeros, out, args);
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return out;
}

}  // namespace infer
