#include <cuda_runtime.h>
#include <torch/extension.h>

// CUDA kernel for matrix multiplication (32x32 matrices)
__global__ void matmul_cuda_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K) {
    // Calculate row and column indices for the current thread
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Perform the dot product for the row and column
    float value = 0.0f;
    for (int k = 0; k < K; ++k) {
        value += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = value;
}

// Host function to launch the kernel
void matmul_cuda(
    at::Tensor A, at::Tensor B, at::Tensor C,
    int M, int N, int K) {
    // Ensure tensors are on the GPU and contiguous
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(C.is_contiguous(), "C must be contiguous");

    // Define block and grid dimensions for 32x32 matrix
    dim3 blockDim(32, 32);  // 32x32 threads per block (one thread per element)
    dim3 gridDim(1, 1);     // Single block for the entire matrix

    // Launch the kernel
    matmul_cuda_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    // Synchronize the device
    cudaDeviceSynchronize();
}