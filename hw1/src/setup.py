from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='matmul_cuda',
    ext_modules=[
        CUDAExtension(
            name='matmul_cuda',
            sources=['src/matmul_cuda.cpp', 'src/matmul_cuda_kernel.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)