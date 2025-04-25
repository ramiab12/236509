#include <cuda.h>
#include <cuda_runtime.h>

// Naive kernel
__global__ void matmul_naive_kernel(
    const float* a1, const float* a2, float* out, int n, int N
) {
    int batch = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int offset = batch * n * n;

    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
        sum += a1[offset + row * n + k] * a2[offset + k * n + col];
    }
    out[offset + row * n + col] = sum;
}

// Optimized kernel (shared memory + transposed access)
__global__ void matmul_optimized_kernel(
    const float* a1, const float* a2, float* out, int n, int N
) {
    int batch = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int offset = batch * n * n;

    extern __shared__ float shared_mem[];
    float* a1_shared = shared_mem;
    float* a2_shared = shared_mem + n * n;

    // Load matrices into shared memory (a2 is transposed)
    a1_shared[row * n + col] = a1[offset + row * n + col];
    a2_shared[col * n + row] = a2[offset + row * n + col];  // Transpose during load
    __syncthreads();

    // Compute dot product
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < n; k++) {
        sum += a1_shared[row * n + k] * a2_shared[col * n + k];
    }
    out[offset + row * n + col] = sum;
}

// Kernel dispatcher
void launch_matmul_kernel(
    const float* a1, const float* a2, float* out, int n, int N, bool use_optimized
) {
    dim3 grid(N);  // One block per batch
    dim3 block(n, n);  // One thread per matrix element
    if (use_optimized) {
        size_t shared_size = 2 * n * n * sizeof(float);
        matmul_optimized_kernel<<<grid, block, shared_size>>>(a1, a2, out, n, N);
    } else {
        matmul_naive_kernel<<<grid, block>>>(a1, a2, out, n, N);
    }
}