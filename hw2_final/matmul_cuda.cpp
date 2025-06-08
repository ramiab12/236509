
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <vector>


extern  void matmul_bool_cuda(const bool* A,const bool* B,bool* C,int batch_size);

extern  void matmul_fp16_cuda(const __half* A, const __half* B,__half* C,int batch_size);


torch::Tensor matmul_cpp(torch::Tensor A, torch::Tensor B)
{
    const int batch = A.size(0);
    static torch::Tensor C = torch::empty_like(A);

    if (A.scalar_type() == torch::kBool) {
        matmul_bool_cuda(A.data_ptr<bool>(),
                         B.data_ptr<bool>(),
                         C.data_ptr<bool>(),
                         batch);
    } else if (A.scalar_type() == torch::kHalf) {
        matmul_fp16_cuda(reinterpret_cast<__half*>(A.data_ptr<at::Half>()),
                         reinterpret_cast<__half*>(B.data_ptr<at::Half>()),
                         reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
                         batch);}
    return C;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cpp", &matmul_cpp, "32×32 batched MatMul (bool / fp16)");
}
