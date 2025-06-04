#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>  // For at::Half

#define MATRIX_SIZE 32
#define MATRIX_AREA (MATRIX_SIZE * MATRIX_SIZE)
#define CYCLE_SIZE 4

__global__ void matMul32x32_half(const __half* A, const __half* B, __half* C) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int cycle_idx = blockIdx.x;
    int base_idx = cycle_idx * CYCLE_SIZE;

    __shared__ __half sA[CYCLE_SIZE][MATRIX_SIZE][MATRIX_SIZE];
    __shared__ __half sB[CYCLE_SIZE][MATRIX_SIZE][MATRIX_SIZE];

    #pragma unroll
    for (int i = 0; i < CYCLE_SIZE; ++i) {
        int batch_idx = base_idx + i;
        if (batch_idx < gridDim.x * CYCLE_SIZE) {
            const __half* Aptr = A + batch_idx * MATRIX_AREA;
            const __half* Bptr = B + batch_idx * MATRIX_AREA;
            sA[i][ty][tx] = Aptr[ty * MATRIX_SIZE + tx];
            sB[i][ty][tx] = Bptr[ty * MATRIX_SIZE + tx];
        }
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < CYCLE_SIZE; ++i) {
        int batch_idx = base_idx + i;
        if (batch_idx < gridDim.x * CYCLE_SIZE) {
            __half sum = __float2half(0.0f);
            #pragma unroll
            for (int k = 0; k < MATRIX_SIZE; ++k) {
                sum = __hadd(sum, __hmul(sA[i][ty][k], sB[i][k][tx]));
            }
            __half* Cptr = C + batch_idx * MATRIX_AREA;
            Cptr[ty * MATRIX_SIZE + tx] = sum;
        }
    }
}

void matmul_cuda_half(at::Half* A, at::Half* B, at::Half* C, int batch_size) {
    __half* A_half = reinterpret_cast<__half*>(A);
    __half* B_half = reinterpret_cast<__half*>(B);
    __half* C_half = reinterpret_cast<__half*>(C);

    dim3 threads(MATRIX_SIZE, MATRIX_SIZE);
    int grid_size = (batch_size + CYCLE_SIZE - 1) / CYCLE_SIZE;

    matMul32x32_half<<<grid_size, threads>>>(A_half, B_half, C_half);
}
