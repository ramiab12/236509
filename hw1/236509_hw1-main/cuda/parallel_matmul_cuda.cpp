#include <torch/extension.h>
#include <stdio.h>
#include <iostream>

class WrongDimensions: std::exception{};

torch::Tensor parallel_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    const int threads,
    const int thread_blocks);



// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor parallel_matmul(
    torch::Tensor A,
    torch::Tensor B,
    const int threads,
    const int thread_blocks) {

  using std::cout;
  using std::endl;

  if(A.dim() == 1 && B.dim() == 1 && A.size(0) == B.size(0)){
    // only 1 row and 1 col => vector dot => one thread => cpu can be faster since no GPU memory management overhead and no dat amigration overhead.
    torch::Tensor C = torch::matmul(A, B);
    return C;
  }else{
    if(A.dim() == 1){
      A = A.reshape({1, A.size(0)});
    }
    if(B.dim() == 1){
      B = B.reshape({B.size(0), 1});
    }
    torch::Tensor C = torch::zeros({A.size(0), B.size(1)});
    if(A.size(1) == B.size(0)){
      C = parallel_matmul_cuda(A, B, C, threads, thread_blocks);
      if(C.size(0) == 1){
        C = C.reshape({C.size(1)});
      }else if(C.size(1) == 1){
        C = C.reshape({C.size(0)});
      }
      return C;
    }   
  }
  throw WrongDimensions();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ParallelMatMul", &parallel_matmul, "Parallel MatMul (CUDA)");
}
