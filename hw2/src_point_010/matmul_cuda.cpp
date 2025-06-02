// ============================================================================
// PyTorch binding for the bool mat-mul kernel.
// Optional  -DPROFILE  prints CSV “batch,whole_ms,kernel_ms” to stderr.
// ============================================================================

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

// Forward decl. of the CUDA wrapper
void matmul_bool_cuda(const bool* A,
                       const bool* B,
                       bool*       C,
                       int         batch_size);

// ----------------------------------------------------------------------------
torch::Tensor matmul_cpp(torch::Tensor A, torch::Tensor B)
{
    // ---- sanity checks ------------------------------------------------------
    TORCH_CHECK(A.dim()==3 && B.dim()==3, "inputs must be [batch,32,32]");
    TORCH_CHECK(A.size(1)==32 && A.size(2)==32 &&
                B.size(1)==32 && B.size(2)==32, "must be 32×32");
    TORCH_CHECK(A.scalar_type()==torch::kBool &&
                B.scalar_type()==torch::kBool, "dtype must be bool");

    int batch = A.size(0);
    auto opts = torch::TensorOptions().dtype(torch::kBool).device(A.device());
    auto C    = torch::empty({batch,32,32}, opts);

    matmul_bool_cuda(A.data_ptr<bool>(),
                                       B.data_ptr<bool>(),
                                       C.data_ptr<bool>(),
                                       batch);
    return C;
}

// ----------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul_cpp", &matmul_cpp, "32×32 bool matmul (CUDA, 2-warp)");
}
