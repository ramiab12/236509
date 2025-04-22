#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include <iostream>

using std::cout;
using std::endl;

#define BLOCK_SIZE 16

namespace {

  //   template <typename scalar_t>
  // __global__ void parallel_matmul_cuda_kernel(
  //     torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> A,
  //     torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> B,
  //     torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> C, 
  //     unsigned int outDimY, unsigned int outDimX, unsigned int sharedDim, unsigned int batchX, unsigned int batchY
  //     ) {
  //   unsigned int row = batchY * blockDim.y * gridDim.y + blockIdx.y * blockDim.y + threadIdx.y;
  //   unsigned int col = batchX * blockDim.x * gridDim.x + blockIdx.x * blockDim.x + threadIdx.x;

  //   if(row < outDimY && col < outDimX){
  //     scalar_t tmpSum = 0;
      
  //     for(int k = 0; k < sharedDim; k++){
  //       tmpSum = tmpSum + A[row][k] * B[k][col];
  //     }
  //     C[row][col] = tmpSum;
  //   }
  // }
  template <typename scalar_t>
__global__ void parallel_matmul_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> A,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> B,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> C, 
    unsigned int outDimY, unsigned int outDimX, unsigned int sharedDim,
    unsigned int batchX, unsigned int batchY
    ) {
  __shared__ scalar_t sA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ scalar_t sB[BLOCK_SIZE][BLOCK_SIZE];

  unsigned int bx = blockIdx.x + batchX * gridDim.x;
  unsigned int by = blockIdx.y + batchY * gridDim.y;
  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;

  unsigned int row = by * BLOCK_SIZE + ty;
  unsigned int col = bx * BLOCK_SIZE + tx;

  scalar_t tmpSum = 0;

  for(int t = 0; t < (sharedDim - 1) / BLOCK_SIZE + 1; t++){

    if(row < outDimY && t*BLOCK_SIZE + tx < sharedDim)
      sA[ty][tx] = A[row][t*BLOCK_SIZE + tx];
    else
      sA[ty][tx] = 0.0;

    if(t*BLOCK_SIZE + ty < sharedDim && col < outDimX)
      sB[ty][tx] = B[t*BLOCK_SIZE + ty][col];
    else
      sB[ty][tx] = 0.0;

    __syncthreads();

    for(int k = 0; k < BLOCK_SIZE; k++){
      tmpSum += sA[ty][k] * sB[k][tx];
    }

    __syncthreads();
  }

  if(row < outDimY && col < outDimX){
    C[row][col] = tmpSum;
  }
}
} // namespace

torch::Tensor parallel_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    const int threads,
    const int threadBlocks) 
{

  int blockSize = ceil(sqrt(threads)); // in our case always 16
  int gridSize = ceil(sqrt(threadBlocks));
  dim3 dimGrid(gridSize, gridSize);
  dim3 dimBlock(blockSize, blockSize);
  using std::cout;
  using std::endl;
  torch::Tensor C_cuda = C.to(torch::kCUDA, true);

  int iterations_y = ceil((float)C.size(0) / blockSize);
  int iterations_x = ceil((float)C.size(1) / blockSize);
  if(iterations_x == 0){
    iterations_x = 1;
  }
  if(iterations_y == 0){
    iterations_y = 1;
  }
  for(int batchY = 0; batchY < iterations_y; batchY++){
    for(int batchX = 0; batchX < iterations_x; batchX++){
      AT_DISPATCH_FLOATING_TYPES(A.type(), "parallel_matmul_cuda", ([&] {
        parallel_matmul_cuda_kernel<scalar_t><<<dimGrid, dimBlock>>>(
          A.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          B.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          C_cuda.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
          C.size(0), C.size(1), A.size(1), batchX, batchY
        );
      }));
    }
  }
  cudaDeviceSynchronize();
  return C_cuda;
}
