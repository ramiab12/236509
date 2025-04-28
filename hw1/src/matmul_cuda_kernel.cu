#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 32

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;  // Row index of the output matrix
    int col = threadIdx.x + blockIdx.x * blockDim.x;  // Column index of the output matrix

    __shared__ float A_shared[TILE][TILE];
    __shared__ float B_shared[TILE][TILE];

    float sum = 0.0f;

    for (int tile_idx = 0; tile_idx < (K + TILE - 1) / TILE; ++tile_idx) {
        // Load A and B into shared memory
        if (row < M && tile_idx * TILE + threadIdx.x < K) {
            A_shared[threadIdx.y][threadIdx.x] = A[blockIdx.z * M * K + row * K + tile_idx * TILE + threadIdx.x];
        } else {
            A_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && tile_idx * TILE + threadIdx.y < K) {
            B_shared[threadIdx.y][threadIdx.x] = B[blockIdx.z * K * N + (tile_idx * TILE + threadIdx.y) * N + col];
        } else {
            B_shared[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Perform the computation for this tile
        for (int k = 0; k < TILE; ++k) {
            sum += A_shared[threadIdx.y][k] * B_shared[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to the output matrix
    if (row < M && col < N) {
        C[blockIdx.z * M * N + row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda_forward(torch::Tensor a, torch::Tensor b) {
    const auto batch_size = a.size(0);
    const auto M = a.size(1);
    const auto K = a.size(2);
    const auto N = b.size(2);

    auto output = torch::zeros({batch_size, M, N}, a.options());

    const dim3 threads(TILE, TILE);  // 32x32 threads per block
    const dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch_size);  // N blocks for each matrix in the batch

    matmul_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        M,
        N,
        K
    );

    return output;
}