// src/matmul_cuda.cpp
#include <torch/extension.h>
#include <vector>

// Declaration of the CUDA kernel launcher
torch::Tensor matmul_cuda_forward(torch::Tensor a1, torch::Tensor a2);

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda_forward, "Matrix multiplication forward (CUDA)");
}
