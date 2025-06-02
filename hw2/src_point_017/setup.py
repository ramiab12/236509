from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='matmul_cuda',
    ext_modules=[
        CUDAExtension(
            name='matmul_cuda',
            sources=[
                'matmul_cuda.cpp',
                'matmul_cuda_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)