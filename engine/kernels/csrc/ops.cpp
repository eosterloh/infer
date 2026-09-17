// CPU reference kernels for infer. CUDA lives in ops.cu when compiled with nvcc.
#include <torch/extension.h>
#include <cmath>

namespace infer_kernels {

torch::Tensor rms_norm(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_floating_point(), "rms_norm expects floating x");
  TORCH_CHECK(weight.is_floating_point(), "rms_norm expects floating weight");
  TORCH_CHECK(x.size(-1) == weight.numel(), "hidden dim must match weight");
  auto x_c = x.contiguous().to(torch::kFloat32).cpu();
  auto w_c = weight.contiguous().to(torch::kFloat32).cpu().view(-1);
  int64_t hidden = x_c.size(-1);
  int64_t rows = x_c.numel() / hidden;
  auto out = torch::empty_like(x_c);
  const float* xp = x_c.data_ptr<float>();
  const float* wp = w_c.data_ptr<float>();
  float* op = out.data_ptr<float>();
  float eps_f = static_cast<float>(eps);
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = xp + r * hidden;
    float acc = 0.f;
    for (int64_t i = 0; i < hidden; ++i) {
      acc += row[i] * row[i];
    }
    float inv = 1.f / std::sqrt(acc / static_cast<float>(hidden) + eps_f);
    float* o = op + r * hidden;
    for (int64_t i = 0; i < hidden; ++i) {
      o[i] = row[i] * inv * wp[i];
    }
  }
  return out.to(x.scalar_type()).to(x.device());
}

torch::Tensor silu_mul(torch::Tensor gate, torch::Tensor up) {
  TORCH_CHECK(gate.sizes() == up.sizes(), "silu_mul shape mismatch");
  auto g = gate.contiguous().to(torch::kFloat32).cpu();
  auto u = up.contiguous().to(torch::kFloat32).cpu();
  auto out = torch::empty_like(g);
  const float* gp = g.data_ptr<float>();
  const float* uptr = u.data_ptr<float>();
  float* op = out.data_ptr<float>();
  int64_t n = g.numel();
  for (int64_t i = 0; i < n; ++i) {
    float v = gp[i];
    op[i] = (v / (1.f + std::exp(-v))) * uptr[i];
  }
  return out.to(gate.scalar_type()).to(gate.device());
}

}  // namespace infer_kernels

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm", &infer_kernels::rms_norm, "RMSNorm (C++)");
  m.def("silu_mul", &infer_kernels::silu_mul, "SiLU(gate) * up (C++)");
}
