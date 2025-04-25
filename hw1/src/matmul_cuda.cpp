#include <torch/extension.h>

void launch_matmul_kernel(
    const float* a1, const float* a2, float* out, int n, int N, bool use_optimized
);

torch::Tensor matmul_forward(torch::Tensor a1, torch::Tensor a2, bool use_optimized) {
    int N = a1.size(0);
    int n = a1.size(1);
    auto out = torch::zeros_like(a1);
    launch_matmul_kernel(
        a1.data_ptr<float>(), 
        a2.data_ptr<float>(), 
        out.data_ptr<float>(), 
        n, 
        N, 
        use_optimized
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_forward", &matmul_forward, "CUDA-optimized matrix multiplication");
}