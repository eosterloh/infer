// Op registration for the infer kernels.
//
// Everything lands under `torch.ops.infer.*` with a CPU implementation and,
// when nvcc built this extension, a CUDA one. The CPU side stays the readable
// reference; the CUDA side is what runs on the Spark.
#include <torch/extension.h>

#include <limits>
#include <string>

namespace infer {

#ifdef WITH_CUDA
at::Tensor rms_norm_cuda(
    const at::Tensor& x, const at::Tensor& weight, double eps, double weight_offset);
void fused_add_rms_norm_cuda(
    at::Tensor& x,
    at::Tensor& residual,
    const at::Tensor& weight,
    double eps,
    double weight_offset);
at::Tensor gated_rms_norm_cuda(
    const at::Tensor& x,
    const at::Tensor& gate,
    const at::Tensor& weight,
    double eps,
    int64_t group,
    bool gate_first);
at::Tensor act_mul_cuda(
    const at::Tensor& gate, const at::Tensor& up, const std::string& act);
at::Tensor act_and_mul_cuda(const at::Tensor& gate_up, const std::string& act);
void rope_inplace_cuda(
    at::Tensor& q,
    at::Tensor& k,
    const at::Tensor& cos,
    const at::Tensor& sin,
    bool interleaved);
at::Tensor gemv_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias);
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
    double global_scale);
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
    double global_scale);
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
    at::ScalarType dtype);

at::Tensor moe_gemv_cuda(
    const at::Tensor& x,
    const at::Tensor& w,
    const at::Tensor& row_expert,
    const c10::optional<at::Tensor>& row_input,
    const c10::optional<at::Tensor>& bias);
at::Tensor moe_combine_cuda(
    const at::Tensor& expert_out, const at::Tensor& weights, int64_t topk);

at::Tensor gdn_decode_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& g_log,
    const at::Tensor& beta,
    at::Tensor state);
at::Tensor attn_decode_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& kv_mask,
    const c10::optional<at::Tensor>& sinks,
    double scale,
    int64_t window,
    double softcap);
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
    double dt_hi);

// The registered op takes a dtype code because torch.ScalarType is not a
// schema type; 0 = bfloat16, 1 = float16.
static at::Tensor dequant_cuda_shim(
    const at::Tensor& qweight,
    const c10::optional<at::Tensor>& scales,
    const c10::optional<at::Tensor>& zeros,
    const c10::optional<at::Tensor>& channel_scale,
    int64_t kind,
    int64_t group_size,
    int64_t n_cols,
    int64_t k_dim,
    double global_scale,
    int64_t dtype_code) {
  return dequant_cuda(
      qweight, scales, zeros, channel_scale, kind, group_size, n_cols, k_dim,
      global_scale, dtype_code == 1 ? at::kHalf : at::kBFloat16);
}
#endif

static at::Tensor activate(const at::Tensor& x, const std::string& act) {
  if (act == "silu" || act == "swiglu") {
    return at::silu(x);
  }
  if (act == "gelu_tanh" || act == "gelu_pytorch_tanh" || act == "gelu_new") {
    return at::gelu(x, "tanh");
  }
  if (act == "gelu" || act == "gelu_erf") {
    return at::gelu(x);
  }
  if (act == "relu2" || act == "relu_squared" || act == "squared_relu") {
    return at::relu(x).square();
  }
  TORCH_CHECK(false, "unsupported fused activation: ", act);
}

at::Tensor rms_norm_cpu(
    const at::Tensor& x, const at::Tensor& weight, double eps, double weight_offset) {
  TORCH_CHECK(x.is_floating_point(), "rms_norm expects floating x");
  TORCH_CHECK(x.size(-1) == weight.numel(), "rms_norm: weight must match hidden dim");
  auto xf = x.to(at::kFloat);
  auto inv = at::rsqrt(xf.pow(2).mean(-1, /*keepdim=*/true) + eps);
  auto w = weight.to(at::kFloat).view(-1) + weight_offset;
  return (xf * inv * w).to(x.scalar_type());
}

void fused_add_rms_norm_cpu(
    at::Tensor& x,
    at::Tensor& residual,
    const at::Tensor& weight,
    double eps,
    double weight_offset) {
  TORCH_CHECK(x.sizes() == residual.sizes(), "fused_add_rms_norm shape mismatch");
  residual.add_(x);
  x.copy_(rms_norm_cpu(residual, weight, eps, weight_offset));
}

at::Tensor gated_rms_norm_cpu(
    const at::Tensor& x,
    const at::Tensor& gate,
    const at::Tensor& weight,
    double eps,
    int64_t group,
    bool gate_first) {
  TORCH_CHECK(x.sizes() == gate.sizes(), "gated_rms_norm: gate shape mismatch");
  TORCH_CHECK(group > 0 && x.numel() % group == 0, "gated_rms_norm: bad group");
  auto shape = x.sizes().vec();
  auto xf = x.to(at::kFloat).reshape({-1, group});
  auto gf = at::silu(gate.to(at::kFloat).reshape({-1, group}));
  auto value = gate_first ? xf * gf : xf;
  auto inv = at::rsqrt(value.pow(2).mean(-1, /*keepdim=*/true) + eps);
  // The variance is per group but the scale is per channel, so a weight that
  // spans several groups lines up with the consecutive rows of one token.
  TORCH_CHECK(weight.numel() % group == 0, "gated_rms_norm: weight must be whole groups");
  auto wv = weight.to(at::kFloat).view({-1, group});
  auto out = ((value * inv).view({-1, wv.size(0), group}) * wv).view({-1, group});
  if (!gate_first) {
    out = out * gf;
  }
  return out.reshape(shape).to(x.scalar_type());
}

at::Tensor act_mul_cpu(
    const at::Tensor& gate, const at::Tensor& up, const std::string& act) {
  TORCH_CHECK(gate.sizes() == up.sizes(), "act_mul shape mismatch");
  auto out = activate(gate.to(at::kFloat), act) * up.to(at::kFloat);
  return out.to(gate.scalar_type());
}

at::Tensor act_and_mul_cpu(const at::Tensor& gate_up, const std::string& act) {
  const int64_t width = gate_up.size(-1);
  TORCH_CHECK(width % 2 == 0, "act_and_mul needs an even last dim");
  auto parts = gate_up.chunk(2, -1);
  return act_mul_cpu(parts[0].contiguous(), parts[1].contiguous(), act);
}

void rope_inplace_cpu(
    at::Tensor& q,
    at::Tensor& k,
    const at::Tensor& cos,
    const at::Tensor& sin,
    bool interleaved) {
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "rope_inplace expects [B, H, S, D]");
  const int64_t rotary_dim = cos.size(-1);
  auto c = cos.unsqueeze(1);
  auto s = sin.unsqueeze(1);
  if (interleaved) {
    const int64_t half = rotary_dim / 2;
    c = c.narrow(-1, 0, half).repeat_interleave(2, -1);
    s = s.narrow(-1, 0, half).repeat_interleave(2, -1);
  }
  for (at::Tensor* t : {&q, &k}) {
    auto slice = t->narrow(-1, 0, rotary_dim);
    at::Tensor rotated;
    if (interleaved) {
      auto even = slice.slice(-1, 0, rotary_dim, 2);
      auto odd = slice.slice(-1, 1, rotary_dim, 2);
      rotated = at::stack({-odd, even}, -1).flatten(-2);
    } else {
      const int64_t half = rotary_dim / 2;
      auto x1 = slice.narrow(-1, 0, half);
      auto x2 = slice.narrow(-1, half, rotary_dim - half);
      rotated = at::cat({-x2, x1}, -1);
    }
    slice.copy_(slice * c + rotated * s);
  }
}

at::Tensor gemv_cpu(
    const at::Tensor& x,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias) {
  return at::linear(x, weight, bias.has_value() ? bias.value() : at::Tensor());
}

// --- quantized weights -------------------------------------------------
// FP4 E2M1 nibble codes, then FP8 E4M3 bytes, decoded with ATen so the CPU
// reference reads the same bits as the kernel.
static at::Tensor fp4_table(const at::TensorOptions& options) {
  static const float values[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,
                                   4.0f,  6.0f,  -0.0f, -0.5f, -1.0f, -1.5f,
                                   -2.0f, -3.0f, -4.0f, -6.0f};
  return at::from_blob(const_cast<float*>(values), {16}, at::kFloat)
      .to(options.dtype(at::kFloat));
}

static at::Tensor fp8_e4m3_decode(const at::Tensor& bytes) {
  auto x = bytes.to(at::kInt);
  auto sign = at::where((x.bitwise_and(0x80)).ne(0), -1.0f, 1.0f);
  auto exp = x.bitwise_right_shift(3).bitwise_and(0x0F).to(at::kFloat);
  auto man = x.bitwise_and(0x07).to(at::kFloat);
  auto sub = man * 0.001953125f;
  auto norm = (1.0f + man / 8.0f) * at::pow(2.0f, exp - 7.0f);
  return sign * at::where(exp.eq(0.0f), sub, norm);
}

static at::Tensor unpack_nibbles(const at::Tensor& packed, int64_t n, int64_t k) {
  auto flat = packed.reshape({n, k / 2}).to(at::kInt);
  auto low = flat.bitwise_and(0x0F);
  auto high = flat.bitwise_right_shift(4).bitwise_and(0x0F);
  return at::stack({low, high}, -1).reshape({n, k});
}

at::Tensor dequant_cpu(
    const at::Tensor& qweight,
    const c10::optional<at::Tensor>& scales,
    const c10::optional<at::Tensor>& zeros,
    const c10::optional<at::Tensor>& channel_scale,
    int64_t kind,
    int64_t group_size,
    int64_t n_cols,
    int64_t k_dim,
    double global_scale,
    int64_t dtype_code) {
  const auto dtype = dtype_code == 1 ? at::kHalf : at::kBFloat16;
  at::Tensor out;
  if (kind == 2) {  // fp8
    out = fp8_e4m3_decode(qweight.reshape({n_cols, k_dim}));
    if (channel_scale.has_value() && channel_scale->defined()) {
      out = out * channel_scale->to(at::kFloat).reshape({n_cols, 1});
    } else {
      out = out * static_cast<float>(global_scale);
    }
  } else if (kind == 0) {  // int4 group-wise, w = q * scale + zero
    auto codes = unpack_nibbles(qweight, n_cols, k_dim).to(at::kFloat);
    auto grouped = codes.reshape({n_cols, k_dim / group_size, group_size});
    auto s = scales->to(at::kFloat).reshape({n_cols, k_dim / group_size, 1});
    auto value = grouped * s;
    if (zeros.has_value() && zeros->defined()) {
      value = value + zeros->to(at::kFloat).reshape({n_cols, k_dim / group_size, 1});
    }
    out = value.reshape({n_cols, k_dim});
  } else {  // nvfp4
    auto codes = unpack_nibbles(qweight, n_cols, k_dim);
    auto table = fp4_table(qweight.options());
    auto value = table.index_select(0, codes.reshape(-1).to(at::kLong))
                     .reshape({n_cols, k_dim / group_size, group_size});
    auto s = fp8_e4m3_decode(scales->reshape({n_cols, k_dim / group_size})).unsqueeze(-1);
    if (channel_scale.has_value() && channel_scale->defined()) {
      // A stack of concatenated experts carries one global scale per row.
      s = s * channel_scale->to(at::kFloat).reshape({n_cols, 1, 1});
    } else {
      s = s * static_cast<float>(global_scale);
    }
    out = (value * s).reshape({n_cols, k_dim});
  }
  return out.to(dtype);
}

at::Tensor moe_gemv_cpu(
    const at::Tensor& x,
    const at::Tensor& w,
    const at::Tensor& row_expert,
    const c10::optional<at::Tensor>& row_input,
    const c10::optional<at::Tensor>& bias) {
  TORCH_CHECK(w.dim() == 3, "expert weight must be [E, N, K]");
  const int64_t m_rows = row_expert.numel();
  auto experts = row_expert.to(at::kLong);
  auto inputs = (row_input.has_value() && row_input->defined())
                    ? row_input->to(at::kLong)
                    : at::arange(m_rows, experts.options());
  auto rows = x.index_select(0, inputs);            // [M, K]
  auto weight = w.index_select(0, experts);         // [M, N, K]
  auto out = at::bmm(weight, rows.unsqueeze(-1)).squeeze(-1);
  if (bias.has_value() && bias->defined()) {
    out = out + bias->index_select(0, experts);
  }
  return out;
}

at::Tensor moe_combine_cpu(
    const at::Tensor& expert_out, const at::Tensor& weights, int64_t topk) {
  const int64_t n_cols = expert_out.size(1);
  const int64_t tokens = expert_out.size(0) / topk;
  auto grouped = expert_out.reshape({tokens, topk, n_cols}).to(at::kFloat);
  auto w = weights.reshape({tokens, topk, 1}).to(at::kFloat);
  return (grouped * w).sum(1).to(expert_out.scalar_type());
}

at::Tensor qmoe_gemv_cpu(
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
  const int64_t code = x.scalar_type() == at::kHalf ? 1 : 0;
  const int64_t rows = qweight.numel() * (kind == 2 ? 1 : 2) / k_dim;
  auto w = dequant_cpu(
      qweight, scales, zeros, channel_scale, kind, group_size, rows, k_dim,
      global_scale, code);
  auto stacked = w.reshape({-1, expert_cols, k_dim});
  return moe_gemv_cpu(
      x, stacked.to(x.scalar_type()), row_expert, row_input, c10::nullopt);
}

// Reference selective scan: the same recurrence the kernel runs, written with
// ATen ops so parity tests have something to compare against. `state` is
// updated in place.
at::Tensor mamba2_scan_cpu(
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
  const int64_t batch = x.size(0);
  const int64_t seq = x.size(1);
  const int64_t n_heads = x.size(2);
  const int64_t head_dim = x.size(3);
  const int64_t n_groups = b_mat.size(2);
  const int64_t state_size = b_mat.size(3);
  const int64_t reps = n_heads / n_groups;

  auto a = -at::exp(a_log.to(at::kFloat));                         // [H]
  auto dt = at::softplus(dt_raw.to(at::kFloat) + dt_bias.to(at::kFloat));
  dt = at::clamp(dt, dt_lo, dt_hi);                                // [B, S, H]
  auto xf = x.to(at::kFloat);
  auto bf = b_mat.to(at::kFloat).repeat_interleave(reps, 2);       // [B, S, H, N]
  auto cf = c_mat.to(at::kFloat).repeat_interleave(reps, 2);
  auto acc = has_state ? state.clone()
                       : at::zeros({batch, n_heads, head_dim, state_size}, dt.options());
  auto y = at::empty({batch, seq, n_heads, head_dim}, dt.options());
  for (int64_t t = 0; t < seq; ++t) {
    auto dt_t = dt.select(1, t).unsqueeze(-1);                     // [B, H, 1]
    auto da = at::exp(dt_t.unsqueeze(-1) * a.view({1, n_heads, 1, 1}));
    auto x_t = xf.select(1, t);                                    // [B, H, D]
    auto db = (dt_t * x_t).unsqueeze(-1) * bf.select(1, t).unsqueeze(2);
    acc = acc * da + db;
    auto y_t = at::einsum("bhdn,bhn->bhd", {acc, cf.select(1, t)});
    y.select(1, t).copy_(y_t + x_t * d_skip.to(at::kFloat).view({1, n_heads, 1}));
  }
  state.copy_(acc);
  return y.to(x.scalar_type());
}

// Reference gated delta step, straight off the recurrence in the paper.
// ``state`` is updated in place, as in the kernel.
at::Tensor gdn_decode_cpu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& g_log,
    const at::Tensor& beta,
    at::Tensor state) {
  auto decay = at::exp(g_log).unsqueeze(-1).unsqueeze(-1);       // [B, H, 1, 1]
  auto rec = state * decay;                                      // [B, H, Dk, Dv]
  auto kv_mem = (rec * k.unsqueeze(-1)).sum(-2);                 // [B, H, Dv]
  auto delta = (v - kv_mem) * beta.unsqueeze(-1);
  rec = rec + k.unsqueeze(-1) * delta.unsqueeze(-2);
  state.copy_(rec);
  return (rec * q.unsqueeze(-1)).sum(-2);
}

// Reference single-query attention. Same masking rules as the CUDA kernel:
// every cached key is visible to the newest query, minus the window, minus the
// validity mask, with the sink as a denominator-only logit.
at::Tensor attn_decode_cpu(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& kv_mask,
    const c10::optional<at::Tensor>& sinks,
    double scale,
    int64_t window,
    double softcap) {
  const int64_t heads = q.size(1);
  const int64_t kv_heads = k.size(1);
  const int64_t kv_len = k.size(2);
  const int64_t group = heads / kv_heads;

  auto kf = k.to(at::kFloat).repeat_interleave(group, 1);          // [B, H, L, D]
  auto vf = v.to(at::kFloat).repeat_interleave(group, 1);
  auto qf = q.to(at::kFloat).unsqueeze(2);                         // [B, H, 1, D]
  auto scores = at::matmul(qf, kf.transpose(-2, -1)) * scale;      // [B, H, 1, L]
  if (softcap > 0.0) {
    scores = at::tanh(scores / softcap) * softcap;
  }
  auto keep = at::ones({1, 1, 1, kv_len}, q.options().dtype(at::kBool));
  if (window > 0 && window < kv_len) {
    auto pos = at::arange(kv_len, q.options().dtype(at::kLong));
    keep = keep.logical_and((pos >= (kv_len - window)).view({1, 1, 1, kv_len}));
  }
  if (kv_mask.has_value() && kv_mask->defined()) {
    keep = keep.logical_and(kv_mask->to(at::kBool).view({-1, 1, 1, kv_len}));
  }
  scores = scores.masked_fill(keep.logical_not(), -std::numeric_limits<float>::infinity());
  if (sinks.has_value() && sinks->defined()) {
    auto sink = sinks->to(at::kFloat).view({1, heads, 1, 1}).expand({q.size(0), heads, 1, 1});
    scores = at::cat({scores, sink}, -1);
    auto w = at::softmax(scores, -1).slice(-1, 0, kv_len);
    return at::nan_to_num(at::matmul(w, vf)).squeeze(2).to(q.scalar_type());
  }
  auto w = at::nan_to_num(at::softmax(scores, -1));
  return at::matmul(w, vf).squeeze(2).to(q.scalar_type());
}

at::Tensor qgemv_cpu(
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
  const int64_t code = x.scalar_type() == at::kHalf ? 1 : 0;
  auto w = dequant_cpu(
      qweight, scales, zeros, channel_scale, kind, group_size, n_cols, k_dim,
      global_scale, code);
  return at::linear(x, w.to(x.scalar_type()), at::Tensor());
}

}  // namespace infer

TORCH_LIBRARY(infer, m) {
  m.def("rms_norm(Tensor x, Tensor weight, float eps, float weight_offset) -> Tensor");
  m.def(
      "fused_add_rms_norm(Tensor(a!) x, Tensor(b!) residual, Tensor weight, "
      "float eps, float weight_offset) -> ()");
  m.def("act_mul(Tensor gate, Tensor up, str act) -> Tensor");
  m.def(
      "gated_rms_norm(Tensor x, Tensor gate, Tensor weight, float eps, int group, "
      "bool gate_first) -> Tensor");
  m.def("act_and_mul(Tensor gate_up, str act) -> Tensor");
  m.def(
      "rope_inplace(Tensor(a!) q, Tensor(b!) k, Tensor cos, Tensor sin, "
      "bool interleaved) -> ()");
  m.def("gemv(Tensor x, Tensor weight, Tensor? bias) -> Tensor");
  m.def(
      "qgemv(Tensor x, Tensor qweight, Tensor? scales, Tensor? zeros, "
      "Tensor? channel_scale, int kind, int group_size, int n_cols, int k_dim, "
      "float global_scale) -> Tensor");
  m.def(
      "dequant(Tensor qweight, Tensor? scales, Tensor? zeros, "
      "Tensor? channel_scale, int kind, int group_size, int n_cols, int k_dim, "
      "float global_scale, int dtype_code) -> Tensor");
  m.def(
      "moe_gemv(Tensor x, Tensor w, Tensor row_expert, Tensor? row_input, "
      "Tensor? bias) -> Tensor");
  m.def("moe_combine(Tensor expert_out, Tensor weights, int topk) -> Tensor");
  m.def(
      "qmoe_gemv(Tensor x, Tensor qweight, Tensor? scales, Tensor? zeros, "
      "Tensor? channel_scale, Tensor row_expert, Tensor? row_input, int kind, "
      "int group_size, int expert_cols, int k_dim, float global_scale) -> Tensor");
  m.def(
      "mamba2_scan(Tensor x, Tensor dt_raw, Tensor dt_bias, Tensor a_log, Tensor b, "
      "Tensor c, Tensor d, Tensor(a!) state, bool has_state, float dt_lo, "
      "float dt_hi) -> Tensor");
  m.def(
      "attn_decode(Tensor q, Tensor k, Tensor v, Tensor? kv_mask, Tensor? sinks, "
      "float scale, int window, float softcap) -> Tensor");
  m.def(
      "gdn_decode(Tensor q, Tensor k, Tensor v, Tensor g_log, Tensor beta, "
      "Tensor(a!) state) -> Tensor");
}

TORCH_LIBRARY_IMPL(infer, CPU, m) {
  m.impl("rms_norm", TORCH_FN(infer::rms_norm_cpu));
  m.impl("fused_add_rms_norm", TORCH_FN(infer::fused_add_rms_norm_cpu));
  m.impl("act_mul", TORCH_FN(infer::act_mul_cpu));
  m.impl("gated_rms_norm", TORCH_FN(infer::gated_rms_norm_cpu));
  m.impl("act_and_mul", TORCH_FN(infer::act_and_mul_cpu));
  m.impl("rope_inplace", TORCH_FN(infer::rope_inplace_cpu));
  m.impl("gemv", TORCH_FN(infer::gemv_cpu));
  m.impl("qgemv", TORCH_FN(infer::qgemv_cpu));
  m.impl("dequant", TORCH_FN(infer::dequant_cpu));
  m.impl("moe_gemv", TORCH_FN(infer::moe_gemv_cpu));
  m.impl("moe_combine", TORCH_FN(infer::moe_combine_cpu));
  m.impl("qmoe_gemv", TORCH_FN(infer::qmoe_gemv_cpu));
  m.impl("mamba2_scan", TORCH_FN(infer::mamba2_scan_cpu));
  m.impl("attn_decode", TORCH_FN(infer::attn_decode_cpu));
  m.impl("gdn_decode", TORCH_FN(infer::gdn_decode_cpu));
}

#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(infer, CUDA, m) {
  m.impl("rms_norm", TORCH_FN(infer::rms_norm_cuda));
  m.impl("fused_add_rms_norm", TORCH_FN(infer::fused_add_rms_norm_cuda));
  m.impl("act_mul", TORCH_FN(infer::act_mul_cuda));
  m.impl("gated_rms_norm", TORCH_FN(infer::gated_rms_norm_cuda));
  m.impl("act_and_mul", TORCH_FN(infer::act_and_mul_cuda));
  m.impl("rope_inplace", TORCH_FN(infer::rope_inplace_cuda));
  m.impl("gemv", TORCH_FN(infer::gemv_cuda));
  m.impl("qgemv", TORCH_FN(infer::qgemv_cuda));
  m.impl("dequant", TORCH_FN(infer::dequant_cuda_shim));
  m.impl("moe_gemv", TORCH_FN(infer::moe_gemv_cuda));
  m.impl("moe_combine", TORCH_FN(infer::moe_combine_cuda));
  m.impl("qmoe_gemv", TORCH_FN(infer::qmoe_gemv_cuda));
  m.impl("mamba2_scan", TORCH_FN(infer::mamba2_scan_cuda));
  m.impl("attn_decode", TORCH_FN(infer::attn_decode_cuda));
  m.impl("gdn_decode", TORCH_FN(infer::gdn_decode_cuda));
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "infer kernels; prefer torch.ops.infer.*";
  m.def("has_cuda", []() {
#ifdef WITH_CUDA
    return true;
#else
    return false;
#endif
  });
}
