#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 32
#define TILE_SIZE 16

__global__ void matmul_bool_kernel(
    const bool* __restrict__ A, 
    const bool* __restrict__ B, 
    bool* __restrict__ C, 
    int batch_size
) {
    // Shared memory for tiling - reduces global memory access
    __shared__ bool As[TILE_SIZE][TILE_SIZE];
    __shared__ bool Bs[TILE_SIZE][TILE_SIZE];
    
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    int result = 0;
    int matrix_offset = batch_idx * N * N;
    
    // Tile across the K dimension
    #pragma unroll
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load A tile
        int a_col = tile * TILE_SIZE + threadIdx.x;
        if (row < N && a_col < N) {
            As[threadIdx.y][threadIdx.x] = A[matrix_offset + row * N + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = false;
        }
        
        // Load B tile  
        int b_row = tile * TILE_SIZE + threadIdx.y;
        if (b_row < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[matrix_offset + b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = false;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            // Boolean AND + accumulate count
            if (As[threadIdx.y][k] && Bs[k][threadIdx.x]) {
                result++;
            }
        }
        
        __syncthreads();
    }
    
    // Store result (any non-zero count becomes true)
    if (row < N && col < N) {
        C[matrix_offset + row * N + col] = (result > 0);
    }
}

void matmul_bool_cuda(const bool* A, const bool* B, bool* C, int batch_size) {
    // Grid configuration for 32x32 matrices with 16x16 tiles
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,  // 2 blocks per matrix dimension  
        (N + TILE_SIZE - 1) / TILE_SIZE,  // 2 blocks per matrix dimension
        batch_size                         // 10000 matrices
    );
    
    matmul_bool_kernel<<<grid, block>>>(A, B, C, batch_size);
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}