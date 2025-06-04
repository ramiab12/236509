// matmul_cuda.cpp
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>            // for __half
#include <vector>

#define N 32

// Declaration of the device function in matmul_cuda_kernel.cu
extern void matmul_cuda(const __half* A, const __half* B, __half* C, int batch_size);

torch::Tensor matmul_cpp(torch::Tensor A, torch::Tensor B) {
    // A and B are expected to be torch.float16 tensors of shape [batch_size, 32, 32]
    const int batch_size = A.size(0);

    // Allocate output in FP16
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat16)
                       .device(torch::kCUDA);
    torch::Tensor C = torch::empty({batch_size, N, N}, options);

    // Reinterpret the contiguous data pointers as __half*
    const __half* A_data = reinterpret_cast<const __half*>(A.contiguous().data_ptr<at::Half>());
    const __half* B_data = reinterpret_cast<const __half*>(B.contiguous().data_ptr<at::Half>());
    __half*       C_data = reinterpret_cast<__half*>(C.data_ptr<at::Half>());

    matmul_cuda(A_data, B_data, C_data, batch_size);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cpp", &matmul_cpp, "FP16 matrix multiply (32×32 batched)");
}
