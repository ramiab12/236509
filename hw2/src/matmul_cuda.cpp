//   matmul_cuda.cpp
//   PyBind11 glue exposing two kernels: float16 and bool (bit-packed)
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Forward declarations of the two CUDA kernels (defined in .cu file)
// ---------------------------------------------------------------------------
void launch_fp16_kernel(const __half* A, const __half* B, __half* C,
                        int batch, cudaStream_t stream);
void launch_bool_kernel(const uint8_t* A, const uint8_t* B, uint8_t* C,
                        int batch, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Helper to ensure contiguous CUDA tensor of given dtype
// ---------------------------------------------------------------------------
static at::Tensor to_cuda_contig(const at::Tensor& cpu, at::ScalarType dtype) {
    return cpu.to(dtype).to(at::kCUDA, /*non_blocking=*/true).contiguous();
}

// ---------------------------------------------------------------------------
// matmul_fp16: A,B float16 on *CPU* ? returns float16 on CPU
// ---------------------------------------------------------------------------
static at::Tensor matmul_fp16(const at::Tensor& A_cpu, const at::Tensor& B_cpu) {
    TORCH_CHECK(A_cpu.scalar_type() == at::kHalf, "dtype must be float16");
    TORCH_CHECK(A_cpu.sizes() == B_cpu.sizes() && A_cpu.dim() == 3 &&
                A_cpu.size(1) == 32 && A_cpu.size(2) == 32,
                "Input shape must be (N,32,32)");

    int batch = A_cpu.size(0);
    auto A = to_cuda_contig(A_cpu, at::kHalf);
    auto B = to_cuda_contig(B_cpu, at::kHalf);
    auto C = at::empty_like(A);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_fp16_kernel(reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
                       reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
                       reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
                       batch, stream);
    return C.to(at::kCPU, /*non_blocking=*/false);
}

// ---------------------------------------------------------------------------
// matmul_bool: A,B bool on *CPU* ? returns bool on CPU
// ---------------------------------------------------------------------------
static at::Tensor matmul_bool(const at::Tensor& A_cpu, const at::Tensor& B_cpu) {
    TORCH_CHECK(A_cpu.scalar_type() == at::kBool, "dtype must be bool");
    TORCH_CHECK(A_cpu.sizes() == B_cpu.sizes() && A_cpu.dim() == 3 &&
                A_cpu.size(1) == 32 && A_cpu.size(2) == 32,
                "Input shape must be (N,32,32)");

    int batch = A_cpu.size(0);
    // move to CUDA as uint8 (0/1)
    auto A = to_cuda_contig(A_cpu.to(at::kByte), at::kByte);
    auto B = to_cuda_contig(B_cpu.to(at::kByte), at::kByte);
    auto C = at::empty_like(A);  // uint8 on CUDA

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_bool_kernel(A.data_ptr<uint8_t>(), B.data_ptr<uint8_t>(),
                       C.data_ptr<uint8_t>(), batch, stream);
    // Cast back to bool on CPU
    return C.to(at::kCPU, /*non_blocking=*/false).to(at::kBool);
}

// ---------------------------------------------------------------------------
PYBIND11_MODULE(matmul_cuda, m) {
    m.def("matmul_fp16", &matmul_fp16, "Matrix multiply float16 (CUDA)");
    m.def("matmul_bool", &matmul_bool, "Matrix multiply bool (CUDA, bitpacked)");
}