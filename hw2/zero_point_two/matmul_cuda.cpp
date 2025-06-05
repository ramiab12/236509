// -----------------------------------------------------------------------------
// matmul_cuda.cpp  –  PyTorch binding with end-to-end timing
// -----------------------------------------------------------------------------
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>

// Forward declaration (returns kernel-only time)
float matmul_bool_cuda(const bool* A,
                       const bool* B,
                       bool*       C,
                       int         batch_size);

// -----------------------------------------------------------------------------
// C++ → CUDA bridge
// -----------------------------------------------------------------------------
torch::Tensor matmul_cpp(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3,
                "Input tensors must be [batch, 32, 32]");
    TORCH_CHECK(A.size(1) == 32 && A.size(2) == 32, "A must be 32×32");
    TORCH_CHECK(B.size(1) == 32 && B.size(2) == 32, "B must be 32×32");
    TORCH_CHECK(A.scalar_type() == torch::kBool &&
                B.scalar_type() == torch::kBool,
                "dtype must be bool");

    int batch = A.size(0);

    auto options = torch::TensorOptions()
                       .dtype(torch::kBool)
                       .device(A.device());
    torch::Tensor C = torch::empty({batch, 32, 32}, options);

    // ---------- whole-call CPU timer ----------
    auto t0 = std::chrono::steady_clock::now();
    float kernel_ms = matmul_bool_cuda(A.data_ptr<bool>(),
                                       B.data_ptr<bool>(),
                                       C.data_ptr<bool>(),
                                       batch);
    cudaDeviceSynchronize();                       // ensure GPU work is done
    auto t1 = std::chrono::steady_clock::now();
    double whole_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---------- CSV to stderr ----------
    static bool header_printed = false;
    if (!header_printed) {
        std::fprintf(stderr, "batch,whole_ms,kernel_ms\n");
        header_printed = true;
    }
    std::fprintf(stderr, "%d,%.6f,%.6f\n", batch, whole_ms, kernel_ms);

    return C;
}

// -----------------------------------------------------------------------------
// pybind11 module
// -----------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul_cpp", &matmul_cpp,
          "32×32 bool matmul (timed: whole + kernel)");
}
