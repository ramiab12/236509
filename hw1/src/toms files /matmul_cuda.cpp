#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 32
#define N 32

//cuda declaration
void matmul_cuda(float *A, float *B, float *C, int batch_size);

torch::Tensor matmul_cpp(torch::Tensor A, torch::Tensor B) {
    int batch_size = 10000;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    torch::Tensor C = torch::empty({batch_size, N, N}, options);

    float* A_ptr = A.data_ptr<float>();
    float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();
    //float* C_ptr = nullptr;

    matmul_cuda(A_ptr, B_ptr, C_ptr, batch_size);

    //torch::Tensor C = torch::from_blob(C_ptr, {batch_size, N, N}, torch::kFloat32).to(torch::kCUDA);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cpp", &matmul_cpp, "cpp matrix mul");
}