// Op registration for the infer kernels.
//
// Everything lands under `torch.ops.infer.*` with a CPU implementation and,
// when nvcc built this extension, a CUDA one. The CPU side stays the readable
// reference; the CUDA side is what runs on the Spark.
#include <torch/extension.h>

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

}  // namespace infer

TORCH_LIBRARY(infer, m) {
  m.def("rms_norm(Tensor x, Tensor weight, float eps, float weight_offset) -> Tensor");
  m.def(
      "fused_add_rms_norm(Tensor(a!) x, Tensor(b!) residual, Tensor weight, "
      "float eps, float weight_offset) -> ()");
  m.def("act_mul(Tensor gate, Tensor up, str act) -> Tensor");
  m.def("act_and_mul(Tensor gate_up, str act) -> Tensor");
  m.def(
      "rope_inplace(Tensor(a!) q, Tensor(b!) k, Tensor cos, Tensor sin, "
      "bool interleaved) -> ()");
  m.def("gemv(Tensor x, Tensor weight, Tensor? bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(infer, CPU, m) {
  m.impl("rms_norm", TORCH_FN(infer::rms_norm_cpu));
  m.impl("fused_add_rms_norm", TORCH_FN(infer::fused_add_rms_norm_cpu));
  m.impl("act_mul", TORCH_FN(infer::act_mul_cpu));
  m.impl("act_and_mul", TORCH_FN(infer::act_and_mul_cpu));
  m.impl("rope_inplace", TORCH_FN(infer::rope_inplace_cpu));
  m.impl("gemv", TORCH_FN(infer::gemv_cpu));
}

#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(infer, CUDA, m) {
  m.impl("rms_norm", TORCH_FN(infer::rms_norm_cuda));
  m.impl("fused_add_rms_norm", TORCH_FN(infer::fused_add_rms_norm_cuda));
  m.impl("act_mul", TORCH_FN(infer::act_mul_cuda));
  m.impl("act_and_mul", TORCH_FN(infer::act_and_mul_cuda));
  m.impl("rope_inplace", TORCH_FN(infer::rope_inplace_cuda));
  m.impl("gemv", TORCH_FN(infer::gemv_cuda));
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
