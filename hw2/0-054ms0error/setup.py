from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="matmul_cuda",
    ext_modules=[
        CUDAExtension(
            "matmul_cuda",
            ["matmul_cuda.cpp", "matmul_cuda_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++14"],
                "nvcc": ["-O3", "-arch=sm_75", "-use_fast_math"]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
