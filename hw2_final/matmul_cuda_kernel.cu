#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda::wmma;

// ---------------------------------------------------------------------------
//-------------------------  1.  BOOL  KERNEL  ---------------------------------
// ---------------------------------------------------------------------------
#define N                 32
#define WARP_SIZE         32
#define WARPS_PER_BLOCK    2
#define FULL_MASK   0xFFFFFFFFu

__global__ void matmul_bool_kernel_2warp(const bool* __restrict__ A,const bool* __restrict__ B,bool* __restrict__ C,int batch_size)
{
    const int warp_idx = threadIdx.x >> 5;       // 0 or 1
    const int lane     = threadIdx.x & 31;       // 0-31
    const int matrix_id = blockIdx.x * WARPS_PER_BLOCK + warp_idx;
    if (matrix_id >= batch_size) return;

    const int col  = lane;
    const int base = matrix_id * N * N;

    // ---- pack one column of B into 32-bit mask
    uint32_t col_mask = 0u;
    #pragma unroll
    for (int row = 0; row < N; ++row)
        col_mask |= (uint32_t)B[base + row * N + col] << row;

    // ---- iterate rows of A and ballot
    #pragma unroll
    for (int row = 0; row < N; ++row) {
        bool     abit   = A[base + row * N + col];
        uint32_t rowmask = __ballot_sync(FULL_MASK, abit);
        C[base + row * N + col] = (rowmask & col_mask) != 0u;
    }
}

// host wrapper
void matmul_bool_cuda(const bool* A,const bool* B,bool* C,int batch_size)
{
    dim3 block(WARP_SIZE * WARPS_PER_BLOCK);          // 64 thr
    dim3 grid((batch_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    matmul_bool_kernel_2warp<<<grid, block>>>(A, B, C, batch_size);
}

// ---------------------------------------------------------------------------
//-------------------------  2.  FP16  KERNEL  ---------------------------------
// ---------------------------------------------------------------------------
static constexpr int MATRIX_SIZE = 32;
static constexpr int TILE        = 16;
static constexpr int WARPSIZE    = 32;

__shared__ float Ctemp_shared[TILE * TILE];

extern "C" __global__ void matmul_kernel_32x32_wmma_fp16_fp32(const __half* __restrict__ A,const __half* __restrict__ B,__half* __restrict__ C)
{
    int global_block_id = blockIdx.x;          // 0 .. 4*batch-1
    int matrix_id       = global_block_id >> 2;
    int subtile_id      = global_block_id & 3;
    int sub_i = (subtile_id >> 1) & 1;
    int sub_j = (subtile_id      ) & 1;

    const __half* Aptr = A + matrix_id * MATRIX_SIZE * MATRIX_SIZE;
    const __half* Bptr = B + matrix_id * MATRIX_SIZE * MATRIX_SIZE;
    __half*       Cptr = C + matrix_id * MATRIX_SIZE * MATRIX_SIZE;

    int laneId = threadIdx.x;

    fragment<accumulator, TILE, TILE, TILE, float> acc;
    fill_fragment(acc, 0.0f);

    // ---- K = 0..15 ---------------------------------------------------------
    {
        fragment<matrix_a, TILE, TILE, TILE, __half, row_major> a_frag;
        fragment<matrix_b, TILE, TILE, TILE, __half, row_major> b_frag;
        const __half* Atile = Aptr + (sub_i * TILE) * MATRIX_SIZE;
        const __half* Btile = Bptr + (sub_j * TILE);
        load_matrix_sync(a_frag, Atile, MATRIX_SIZE);
        load_matrix_sync(b_frag, Btile, MATRIX_SIZE);
        mma_sync(acc, a_frag, b_frag, acc);
    }
    // ---- K = 16..31 --------------------------------------------------------
    {
        fragment<matrix_a, TILE, TILE, TILE, __half, row_major> a_frag;
        fragment<matrix_b, TILE, TILE, TILE, __half, row_major> b_frag;
        const __half* Atile = Aptr + (sub_i * TILE) * MATRIX_SIZE + TILE;
        const __half* Btile = Bptr + TILE * MATRIX_SIZE + (sub_j * TILE);
        load_matrix_sync(a_frag, Atile, MATRIX_SIZE);
        load_matrix_sync(b_frag, Btile, MATRIX_SIZE);
        mma_sync(acc, a_frag, b_frag, acc);
    }

    // ---- store FP32 tile to shared ----------------------------------------
    store_matrix_sync(Ctemp_shared, acc, TILE, mem_row_major);
    __syncthreads();

    // ---- scatter shared→global (FP16) -------------------------------------
    for (int e = 0; e < 8; ++e) {
        int localIdx = laneId + e * WARPSIZE;        // 0..255
        int r_local  = localIdx / TILE;
        int c_local  = localIdx % TILE;
        int r_global = sub_i * TILE + r_local;
        int c_global = sub_j * TILE + c_local;
        float val32  = Ctemp_shared[r_local * TILE + c_local];
        Cptr[r_global * MATRIX_SIZE + c_global] = __float2half(val32);
    }
    __syncthreads();
}

// host wrapper
void matmul_fp16_cuda(const __half* A,const __half* B,__half* C,int batch_size)
{
    dim3 blockDim(WARPSIZE);
    dim3 gridDim(batch_size * 4);
    matmul_kernel_32x32_wmma_fp16_fp32<<<gridDim, blockDim>>>(A, B, C);
}
