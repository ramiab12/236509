from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name='matmul_cuda',
    ext_modules=[CUDAExtension('matmul_cuda', [
        'matmul_cuda.cpp',
        'matmul_cuda_kernel.cu',
    ])],
    cmdclass={'build_ext': BuildExtension}
)

# from setuptools import setup
# from setuptools import Extension
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension
#
# setup(
#     name='matmul_cuda',
#     ext_modules=[
#         CUDAExtension(
#             name='matmul_cuda',
#             sources=[
#                 'matmul_cuda.cpp',
#                 'matmul_cuda_kernel.cu',
#             ],
#             extra_compile_args={
#                 'cxx': ['-O2', '-std=c++14'],
#                 'nvcc': ['-O2', '-arch=sm_75']
#             }
#         ),
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )
