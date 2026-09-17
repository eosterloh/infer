// CUDA RMSNorm / silu_mul. Compiled only when nvcc is available.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

namespace infer_kernels {

template <typename T>
__global__ void rms_norm_kernel(
    const T* x, const T* weight, T* out, int64_t rows, int64_t hidden, float eps) {
  int row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const T* row_x = x + row * hidden;
  T* row_o = out + row * hidden;
  float acc = 0.f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float v = static_cast<float>(row_x[i]);
    acc += v * v;
  }
  __shared__ float warp_acc[32];
  // naive block reduce
  __shared__ float shared[256];
  int tid = threadIdx.x;
  shared[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  float inv = rsqrtf(shared[0] / static_cast<float>(hidden) + eps);
  for (int i = tid; i < hidden; i += blockDim.x) {
    float v = static_cast<float>(row_x[i]);
    float w = static_cast<float>(weight[i]);
    row_o[i] = static_cast<T>(v * inv * w);
  }
}

torch::Tensor rms_norm_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda(), "rms_norm_cuda requires CUDA tensor");
  auto x_c = x.contiguous();
  auto w_c = weight.contiguous().view(-1);
  int64_t hidden = x_c.size(-1);
  int64_t rows = x_c.numel() / hidden;
  auto out = torch::empty_like(x_c);
  dim3 block(256);
  dim3 grid(rows);
  if (x_c.scalar_type() == torch::kFloat32) {
    rms_norm_kernel<float><<<grid, block>>>(
        x_c.data_ptr<float>(),
        w_c.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        hidden,
        static_cast<float>(eps));
  } else {
    auto xf = x_c.to(torch::kFloat32);
    auto wf = w_c.to(torch::kFloat32);
    auto of = torch::empty_like(xf);
    rms_norm_kernel<float><<<grid, block>>>(
        xf.data_ptr<float>(),
        wf.data_ptr<float>(),
        of.data_ptr<float>(),
        rows,
        hidden,
        static_cast<float>(eps));
    out = of.to(x.scalar_type());
  }
  return out;
}

}  // namespace infer_kernels
