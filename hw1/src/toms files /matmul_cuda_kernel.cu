#include <cuda_runtime.h>
#include <cstdio>

#define MATRIX_SIZE 32
#define MATRIX_AREA (MATRIX_SIZE * MATRIX_SIZE)
#define CYCLE_SIZE 4


__global__ void matMul32x32(float* A, float* B, float* C) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int cycle_idx = blockIdx.x;
    int base_idx = cycle_idx * CYCLE_SIZE;

    //shared memory for all 4 matrix
    __shared__ float sA[CYCLE_SIZE][MATRIX_SIZE][MATRIX_SIZE];
    __shared__ float sB[CYCLE_SIZE][MATRIX_SIZE][MATRIX_SIZE];

    // Prefetch: loading 4 matrix to shared memory
    #pragma unroll
    for (int i = 0; i < CYCLE_SIZE; ++i) {
        int batch_idx = base_idx + i;
        if (batch_idx < gridDim.x * CYCLE_SIZE) {
            const float* Aptr = A + batch_idx * MATRIX_AREA;
            const float* Bptr = B + batch_idx * MATRIX_AREA;
            sA[i][ty][tx] = Aptr[ty * MATRIX_SIZE + tx];
            sB[i][ty][tx] = Bptr[ty * MATRIX_SIZE + tx];
        }
    }

    __syncthreads();

    //calculate result
    #pragma unroll
    for (int i = 0; i < CYCLE_SIZE; ++i) {
        int batch_idx = base_idx + i;
        if (batch_idx < gridDim.x * CYCLE_SIZE) {
            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < MATRIX_SIZE; ++k) {
                sum += sA[i][ty][k] * sB[i][k][tx];
            }
            float* Cptr = C + batch_idx * MATRIX_AREA;
            Cptr[ty * MATRIX_SIZE + tx] = sum;
        }
    }
}


void matmul_cuda(float* A, float* B, float* C, int batch_size) {
    dim3 threads(MATRIX_SIZE, MATRIX_SIZE);  // 1024 threads
    int grid_size = (batch_size + CYCLE_SIZE - 1) / CYCLE_SIZE;

    matMul32x32<<<grid_size, threads>>>(A, B, C);
}
