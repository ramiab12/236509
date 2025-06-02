#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <cuda.h>
#define N 32
#define THREADS_PER_BLOCK 256

// Single ultra-optimized kernel for 32x32 boolean matrix multiplication
__global__ void matmul_bool_kernel_optimized(
    const bool* __restrict__ A, 
    const bool* __restrict__ B, 
    bool* __restrict__ C, 
    int batch_size
) {
    // Load entire matrices into shared memory for maximum reuse
    __shared__ bool As[N][N];
    __shared__ bool Bs[N][N];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int matrix_offset = batch_idx * N * N;
    
    // Coalesced loading - each thread loads multiple elements
    // 256 threads loading 1024 elements (32x32) = 4 elements per thread
    for (int i = tid; i < N * N; i += THREADS_PER_BLOCK) {
        int row = i / N;
        int col = i % N;
        As[row][col] = A[matrix_offset + i];
        Bs[row][col] = B[matrix_offset + i];
    }
    
    __syncthreads();
    
    // Each thread computes 4 output elements for perfect load balancing
    // 1024 output elements / 256 threads = 4 elements per thread
    for (int elem_idx = tid; elem_idx < N * N; elem_idx += THREADS_PER_BLOCK) {
        int row = elem_idx / N;
        int col = elem_idx % N;
        
        // Compute C[row][col] = (A[row] * B[:][col]) > 0
        bool result = false;
        
        // Fully unrolled inner loop for maximum performance
        // No early exit to maintain consistent timing and avoid divergence
        #pragma unroll 32
        for (int k = 0; k < N; k++) {
            result = result || (As[row][k] && Bs[k][col]);
        }
        
        C[matrix_offset + elem_idx] = result;
    }
}

void matmul_bool_cuda(const bool* A, const bool* B, bool* C, int batch_size) {
    // One block per matrix in the batch
    // 256 threads per block for optimal occupancy
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(batch_size);
    
    matmul_bool_kernel_optimized<<<grid, block>>>(A, B, C, batch_size);
    
    cudaDeviceSynchronize();
}