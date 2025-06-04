// matmul_cuda_kernel.cu
#include <cuda_runtime.h>
#include <cuda_fp16.h>    // for __half, __float2half
#include <mma.h>          // for nvcuda::wmma

using namespace nvcuda::wmma;

static constexpr int MATRIX_SIZE = 32;  // dimension of each matrix
static constexpr int TILE        = 16;  // Tensor‐Core tile is 16×16
static constexpr int WARPSIZE    = 32;  // number of threads in one warp

// Shared‐memory scratchpad: 16×16 floats per block
__shared__ float Ctemp_shared[TILE * TILE];

extern "C" __global__ void matmul_kernel_32x32_wmma_fp16_fp32(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C)
{
    // Determine which 32×32 matrix and which 16×16 subtile this block handles
    int global_block_id = blockIdx.x;          // ranges 0..(4*batch_size−1)
    int matrix_id      = global_block_id >> 2; // block_id / 4
    int subtile_id     = global_block_id & 3;  // block_id % 4

    // Compute sub_i, sub_j from subtile_id:
    //   subtile_id: 0→(0,0), 1→(0,1), 2→(1,0), 3→(1,1)
    int sub_i = (subtile_id >> 1) & 1;  // bit 1
    int sub_j = (subtile_id >> 0) & 1;  // bit 0

    // Base pointers into A, B, C for this specific 32×32 matrix in the batch:
    const __half* Aptr = A + matrix_id * (MATRIX_SIZE * MATRIX_SIZE);
    const __half* Bptr = B + matrix_id * (MATRIX_SIZE * MATRIX_SIZE);
    __half*       Cptr = C + matrix_id * (MATRIX_SIZE * MATRIX_SIZE);

    // Each block is exactly one warp: 32 threads
    int laneId = threadIdx.x;  // 0..31

    // 1) Create and zero‐initialize a 16×16 FP32 accumulator fragment
    fragment<accumulator, TILE, TILE, TILE, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

    // ----------------------------------------------------------------------------
    // 2) First half of K: multiply
    //    A[ sub_i*16 .. sub_i*16+15,       0 .. 15 ]
    //    B[       0 .. 15,             sub_j*16 .. sub_j*16+15 ]
    //    Both are loaded row_major because our data is row-major.
    // ----------------------------------------------------------------------------
    {
        // a_frag holds the A‐tile; b_frag holds the B‐tile
        fragment<matrix_a, TILE, TILE, TILE, __half, row_major> a_frag;
        fragment<matrix_b, TILE, TILE, TILE, __half, row_major> b_frag;

        // Pointer to A[sub_i*16, 0]
        const __half* Atile0 = Aptr + (sub_i * TILE) * MATRIX_SIZE + 0;
        // Pointer to B[0, sub_j*16]
        const __half* Btile0 = Bptr + 0 * MATRIX_SIZE + (sub_j * TILE);

        // Load 16×16 from global memory (row_major)
        load_matrix_sync(a_frag, Atile0, MATRIX_SIZE);
        load_matrix_sync(b_frag, Btile0, MATRIX_SIZE);

        // MMA: acc_frag += a_frag * b_frag  (FP16 inputs, FP32 accumulation)
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // ----------------------------------------------------------------------------
    // 3) Second half of K: multiply
    //    A[ sub_i*16 .. sub_i*16+15,      16 .. 31 ]
    //    B[      16 .. 31,            sub_j*16 .. sub_j*16+15 ]
    // ----------------------------------------------------------------------------
    {
        fragment<matrix_a, TILE, TILE, TILE, __half, row_major> a_frag;
        fragment<matrix_b, TILE, TILE, TILE, __half, row_major> b_frag;

        // Pointer to A[sub_i*16, 16]
        const __half* Atile1 = Aptr + (sub_i * TILE) * MATRIX_SIZE + TILE;
        // Pointer to B[16, sub_j*16]
        const __half* Btile1 = Bptr + TILE * MATRIX_SIZE + (sub_j * TILE);

        load_matrix_sync(a_frag, Atile1, MATRIX_SIZE);
        load_matrix_sync(b_frag, Btile1, MATRIX_SIZE);

        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // ----------------------------------------------------------------------------
    // 4) Store the 16×16 FP32 accumulator into shared memory (Ctemp_shared).
    //    All 32 threads in this warp must call the store.
    // ----------------------------------------------------------------------------
    store_matrix_sync(
        Ctemp_shared,    // destination in shared memory
        acc_frag,        // FP32 accumulator fragment
        TILE,            // leading dimension = 16 floats
        mem_row_major    // row‐major order
    );

    // Make sure all 32 threads have finished writing to shared memory
    __syncthreads();

    // ----------------------------------------------------------------------------
    // 5) Scatter from Ctemp_shared (FP32) → global C (FP16)
    //    Each of the 32 threads writes 8 consecutive elements:
    // ----------------------------------------------------------------------------
    {
        // Each thread in [0..31] handles 8 elements
        for (int e = 0; e < 8; ++e) {
            int localIndex = laneId + e * WARPSIZE;   // 0..255
            int r_local    = localIndex / TILE;       // 0..15
            int c_local    = localIndex % TILE;       // 0..15

            // Global row/col within the 32×32 matrix:
            int r_global = sub_i * TILE + r_local;    // in [0..31]
            int c_global = sub_j * TILE + c_local;    // in [0..31]

            int write_idx = r_global * MATRIX_SIZE + c_global;
            float val32 = Ctemp_shared[r_local * TILE + c_local];
            Cptr[write_idx] = __float2half(val32);
        }
    }

    // Wait for all 32 threads to finish scattering before ending
    __syncthreads();
}

// ------------------------------------------------------------
// Host wrapper (UNCHANGED from before)
// ------------------------------------------------------------
void matmul_cuda(
    const __half* A,
    const __half* B,
    __half*       C,
    int           batch_size)
{
    // **4 blocks per matrix**; each block has exactly 32 threads (one warp).
    // blockIdx.x runs from 0 .. (4*batch_size − 1).
    // threadIdx.x runs from 0 .. 31.
    dim3 blockDim(WARPSIZE, 1, 1);
    dim3 gridDim(batch_size * 4, 1, 1);
    matmul_kernel_32x32_wmma_fp16_fp32<<<gridDim, blockDim>>>(A, B, C);
}
