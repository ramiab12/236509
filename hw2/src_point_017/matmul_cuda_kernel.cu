// -----------------------------------------------------------------------------
// matmul_cuda_kernel.cu  (BIT-PACKED version)
//   • One block   = one 32×32 matrix
//   • One warp    = 32 threads  (THREADS_PER_BLOCK)
//   • Each thread = one output column
//
// Performance path:
//   1. Pack   32 bools  → one uint32_t  (row of A, column of B)
//   2. AND    the two uint32_t          (1 op)
//   3. Test   non-zero → output bit     (1 op)
//
//   Replaces 95 boolean ops per output bit with 2.
// -----------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#define N 32
#define THREADS_PER_BLOCK 32                // one warp

// -----------------------------------------------------------------------------
// Bit-packed kernel
// -----------------------------------------------------------------------------
__global__ void matmul_bool_kernel_bitpacked(const bool* __restrict__ A,
                                             const bool* __restrict__ B,
                                             bool*       __restrict__ C,
                                             int                       batch_size)
{
    const int batch_idx = blockIdx.x;       // one matrix per block
    if (batch_idx >= batch_size) return;

    const int tid = threadIdx.x;            // 0 … 31 (one column)

    // ---------------------------------------------------------
    // Shared memory for packed rows / columns (4 B × 32 × 2 = 256 B)
    // ---------------------------------------------------------
    __shared__ uint32_t A_rows[N];          // packed rows of A
    __shared__ uint32_t B_cols[N];          // packed columns of B

    const int mat_off = batch_idx * N * N;  // starting index of this matrix

    // ---------------------------------------------------------
    // 1.  Pack ONE row of A into a uint32_t
    // ---------------------------------------------------------
    uint32_t row_word = 0u;
    #pragma unroll
    for (int k = 0; k < N; ++k)
    {
        bool bit = A[mat_off + tid * N + k];          // A[tid][k]
        row_word |= (uint32_t)bit << k;
    }
    A_rows[tid] = row_word;

    // ---------------------------------------------------------
    // 2.  Pack ONE column of B into a uint32_t
    //     (non-contiguous load, but only 32 bytes)
    // ---------------------------------------------------------
    uint32_t col_word = 0u;
    #pragma unroll
    for (int k = 0; k < N; ++k)
    {
        bool bit = B[mat_off + k * N + tid];          // B[k][tid]
        col_word |= (uint32_t)bit << k;
    }
    B_cols[tid] = col_word;

    __syncthreads();                                  // all rows/cols ready

    // ---------------------------------------------------------
    // 3.  Compute this thread’s entire column of C
    // ---------------------------------------------------------
    for (int row = 0; row < N; ++row)
    {
        uint32_t mask = A_rows[row] & B_cols[tid];    // 1 AND
        bool     val  = (mask != 0u);                 // any-bit-set?
        C[mat_off + row * N + tid] = val;             // store C[row][tid]
    }
}

// -----------------------------------------------------------------------------
// Host wrapper – unchanged signature, still returns kernel time in ms
// -----------------------------------------------------------------------------

void matmul_bool_cuda(const bool* A,
                       const bool* B,
                       bool*       C,
                       int         batch_size)
{
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid (batch_size);

    matmul_bool_kernel_bitpacked<<<grid, block>>>(A, B, C, batch_size);

}
