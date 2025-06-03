#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 32
#define N 32

// CUDA declaration
void matmul_cuda_half(at::Half* A, at::Half* B, at::Half* C, int batch_size);

torch::Tensor matmul_cpp(torch::Tensor A, torch::Tensor B) {
    int batch_size = 10000;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);

    torch::Tensor C = torch::empty({batch_size, N, N}, options);

    at::Half* A_ptr = A.data_ptr<at::Half>();
    at::Half* B_ptr = B.data_ptr<at::Half>();   
    at::Half* C_ptr = C.data_ptr<at::Half>();

    matmul_cuda_half(A_ptr, B_ptr, C_ptr, batch_size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cpp", &matmul_cpp, "cpp matrix mul (float16)");
}
