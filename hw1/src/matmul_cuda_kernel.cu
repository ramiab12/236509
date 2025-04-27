#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 32

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int batch_size) {
    int batch = blockIdx.z;
    int row = threadIdx.y;
    int col = threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < TILE; ++k) {
        sum += A[batch * TILE * TILE + row * TILE + k] * B[batch * TILE * TILE + k * TILE + col];
    }

    C[batch * TILE * TILE + row * TILE + col] = sum;
}

torch::Tensor matmul_cuda_forward(torch::Tensor a, torch::Tensor b) {
    const auto batch_size = a.size(0);
    auto output = torch::zeros_like(a);

    const dim3 threads(TILE, TILE);
    const dim3 blocks(1, 1, batch_size);

    matmul_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size
    );

    return output;
}
