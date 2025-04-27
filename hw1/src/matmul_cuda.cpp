#include <torch/extension.h>

// Declare the CUDA kernel wrapper
torch::Tensor matmul_cuda_forward(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda_forward, "Matrix multiplication forward (CUDA)");
}
