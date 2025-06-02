// ============================================================================
// 32×32 Boolean MatMul — 2-warp ballot kernel
//
// • Each warp (32 threads) handles one 32×32 matrix.
// • Two warps share a block ⇒ launch latency amortised across 2 matrices.
// • Pure bit-mask logic → exact results (error = 0).
// • Optional –DPROFILE: wraps kernel in CUDA events and returns kernel time.
// ============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#define N                 32
#define WARP_SIZE         32
#define WARPS_PER_BLOCK    2          // —— new
#define FULL_MASK   0xFFFFFFFFu

// ----------------------------------------------------------------------------
// Kernel: one warp  ↔  one matrix
// ----------------------------------------------------------------------------
__global__ void matmul_bool_kernel_2warp(const bool* __restrict__ A,
                                         const bool* __restrict__ B,
                                         bool*       __restrict__ C,
                                         int                       batch_size)
{
    // Which warp inside this block (0 or 1) and my lane (0-31)?
    const int warp_idx = threadIdx.x >> 5;         // /32
    const int lane     = threadIdx.x & 31;         // %32

    // Global matrix handled by this warp
    const int matrix_id = blockIdx.x * WARPS_PER_BLOCK + warp_idx;
    if (matrix_id >= batch_size) return;           // extra warp exits

    const int col      = lane;                     // output column
    const int base     = matrix_id * N * N;        // offset to matrix

    // ---- Phase 1: pack *my* column of B into one uint32_t -------------------
    uint32_t col_mask = 0u;
    #pragma unroll
    for (int row = 0; row < N; ++row) {
        bool bit   = B[base + row * N + col];
        col_mask  |= (uint32_t)bit << row;
    }

    // ---- Phase 2: loop over rows of A, ballot packs row mask ----------------
    #pragma unroll
    for (int row = 0; row < N; ++row) {
        bool     a_bit   = A[base + row * N + col];
        uint32_t rowmask = __ballot_sync(FULL_MASK, a_bit);   // 32-bit row
        bool     val     = (rowmask & col_mask) != 0u;
        C[base + row * N + col] = val;
    }
}

// ----------------------------------------------------------------------------
// Host wrapper: same signature as before.
// • Computes grid/block dims for 2-warp blocks
// • With -DPROFILE: returns kernel time (ms); otherwise returns 0.
// ----------------------------------------------------------------------------

void matmul_bool_cuda(const bool* A,
                       const bool* B,
                       bool*       C,
                       int         batch_size)
{
    dim3 block(WARP_SIZE * WARPS_PER_BLOCK);                 // 64 threads
    dim3 grid((batch_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    matmul_bool_kernel_2warp<<<grid, block>>>(A, B, C, batch_size);
}
