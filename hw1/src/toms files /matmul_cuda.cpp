#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16
#define N 32

//cuda declaration
void matmul_cuda(float *A, float *B, float *C);

void matmul_cpp(float *A, float *B, float *C){

    matmul_cuda(A, B, C);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cpp", &matmul_cpp, "cpp matrix mul");
}