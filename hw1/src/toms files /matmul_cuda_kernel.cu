#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16
#define N 32

__global__ void inerProd(float *A, float *B, float *C, int size){

    __shared__ float A_sub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_sub[BLOCK_SIZE][BLOCK_SIZE];

    // Calculate row and column indices for the thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bpg = gridDim.x;


    // Declare a variable to accumulate the sum
    float sum = 0;

    // Check if the thread's indices are within the matrix dimensions
    if(row < size && col < size) {

        for(int i = 0; i < bpg; ++i) {
            // Load submatrices into shared memory
            A_sub[tx][ty] = A[row + (i * BLOCK_SIZE + tx)];
            B_sub[tx][ty] = B[(i * BLOCK_SIZE + ty) * N + col];
            __syncthreads();
            for(int k = 0; k < BLOCK_SIZE; ++k) {
                sum += A_sub[tx][k] * B_sub[ty][k];
            }
            __syncthreads();
        }
        // Store the sum in the corresponding element of matrix C
        C[row * size + col] = sum;
    }
}


void matmul_cuda(float *A, float *B, float *C){

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float *d_A,* d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));


    cudaMemcpy(d_A, A, N * N * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float),cudaMemcpyHostToDevice);

    // Launch the kernel
    inerProd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, N * N * sizeof(float),cudaMemcpyDeviceToHost);

    // Free allocated memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    //return 0;
}