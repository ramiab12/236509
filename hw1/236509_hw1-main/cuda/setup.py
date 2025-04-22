from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='parallel_matmul_cuda',
    ext_modules=[
        CUDAExtension(name='parallel_matmul_cuda', sources=[
            'parallel_matmul_cuda.cpp',
            'parallel_matmul_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
