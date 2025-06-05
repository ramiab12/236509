#include <torch/extension.h>
#include <cuda_runtime.h>

#define N 32

// CUDA kernel declaration
void matmul_bool_cuda(const bool* A, const bool* B, bool* C, int batch_size);

torch::Tensor matmul_cpp(torch::Tensor A, torch::Tensor B) {
    int batch_size = A.size(0);  // Should be 10000
    
    // Ensure contiguous memory layout
    A = A.contiguous();
    B = B.contiguous();
    
    // Create output tensor
    torch::Tensor C = torch::empty_like(A);
    
    // Launch CUDA kernel
    matmul_bool_cuda(
        A.data_ptr<bool>(),
        B.data_ptr<bool>(),
        C.data_ptr<bool>(),
        batch_size
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cpp", &matmul_cpp, "Boolean matrix multiplication");
}