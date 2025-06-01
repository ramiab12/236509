//   matmul_cuda_kernel.cu
//   Two kernels:
//     1. fp16 WMMA tiling 32×32 (accumulate fp32)
//     2. bool bit-packed using popcount
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>
#include <cuda.h>

using namespace nvcuda;
constexpr int N = 32;      // matrix dimension
constexpr int TILE = 16;   // WMMA tile size

// ---------------------------------------------------------------------------
// 1.  FP16 kernel (Tensor-Core, WMMA)
// ---------------------------------------------------------------------------
// ---------------- fp16 WMMA kernel -----------------------------
__global__ void fp16_kernel(const __half* __restrict__ A,
                                        const __half* __restrict__ B,
                                        __half* __restrict__ C,
                                        int batch) {
    const int bid = blockIdx.x;
    if (bid >= batch) return;

    // Base pointers for this matrix
    const __half* A0 = A + bid * 1024;
    const __half* B0 = B + bid * 1024;
          __half* C0 = C + bid * 1024;

    // Shared memory tiles (fp16 for load; fp32 for accum store)
    __shared__ __half sA[32][32 + 1];   // +1 to avoid bank conflicts
    __shared__ __half sB[32][32 + 1];

    // Accumulator fragment (fp32)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc00, acc01, acc10, acc11;
    wmma::fill_fragment(acc00, 0.f);
    wmma::fill_fragment(acc01, 0.f);
    wmma::fill_fragment(acc10, 0.f);
    wmma::fill_fragment(acc11, 0.f);

    // Loop over K tiles (32 elements / 16 = 2 iterations)
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        // cooperative load 32×32 half tile into shared
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        if (ty < 32 && tx < 32) {
            sA[ty][tx] = A0[ty * 32 + k * 16 + tx];
            sB[ty][tx] = B0[(k * 16 + ty) * 32 + tx];
        }
        __syncthreads();

        // each warp (32 threads) handles 2×2 WMMA tiles (16×16)
        int warpId = (threadIdx.y >> 4) * 2 + (threadIdx.x >> 4);
        int lane   = threadIdx.x & 15;

        const __half* tileA = &sA[(warpId >> 1) * 16][k * 16];
        const __half* tileB = &sB[k * 16][(warpId & 1) * 16];

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>  aFrag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>  bFrag;

        wmma::load_matrix_sync(aFrag, tileA, 32);
        wmma::load_matrix_sync(bFrag, tileB, 32);
        wmma::mma_sync(acc00, aFrag, bFrag, acc00);
    }

    // Store fp32 acc fragments to shared as fp32 first
    __shared__ float sAcc[32][32 + 1];
    wmma::store_matrix_sync(&sAcc[0][0], acc00, 32, wmma::mem_row_major);

    __syncthreads();

    // Each thread converts its element to half and writes global
    int gx = threadIdx.x;
    int gy = threadIdx.y;
    if (gx < 32 && gy < 32) {
        C0[gy * 32 + gx] = __float2half_rn(sAcc[gy][gx]);
    }
}

// Launcher
void launch_fp16_kernel(const __half* A, const __half* B, __half* C,
                                    int batch, cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(batch, 1, 1);
    fp16_kernel<<<grid, block, 0, stream>>>(A, B, C, batch);
}

// ---------------------------------------------------------------------------
// 2.  BOOL kernel – bit-packed popcount
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t ld_u8(const uint8_t* p, int offs) {
    return p[offs];
}

__global__ void bool_kernel(const uint8_t* __restrict__ A,
                            const uint8_t* __restrict__ B,
                            uint8_t* __restrict__ C,
                            int batch) {
    int bi = blockIdx.x;  // which matrix
    if (bi >= batch) return;

    const uint8_t* A_mat = A + bi * N * N;
    const uint8_t* B_mat = B + bi * N * N;
    uint8_t* C_mat       = C + bi * N * N;

    int row = threadIdx.y; // 0..31
    int col = threadIdx.x; // 0..31

    // Pack row bits of A into one uint32
    uint32_t a_pack = 0u;
    #pragma unroll
    for (int k = 0; k < N; ++k) {
        uint8_t bit = ld_u8(A_mat, row * N + k) & 1u;
        a_pack |= (bit << k);
    }

    // Pack column bits of B into one uint32
    uint32_t b_pack = 0u;
    #pragma unroll
    for (int k = 0; k < N; ++k) {
        uint8_t bit = ld_u8(B_mat, k * N + col) & 1u;
        b_pack |= (bit << k);
    }

    // Dot product = popcount of AND
    int dot = __popc(a_pack & b_pack);
    C_mat[row * N + col] = static_cast<uint8_t>(dot != 0);
}

// Launcher
void launch_bool_kernel(const uint8_t* A, const uint8_t* B, uint8_t* C,
                                    int batch, cudaStream_t stream) {
    dim3 block(32, 32, 1);
    dim3 grid(batch, 1, 1);
    bool_kernel<<<grid, block, 0, stream>>>(A, B, C, batch);
}