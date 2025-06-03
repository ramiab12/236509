import numpy as np
import torch
import matmul_cuda
# from numba import cuda


gridsize = (1024, 1024)
blocksize = (32, 32, 32)


def matmul(a1_tensor, a2_tensor):
    return matmul_cuda.matmul_cpp(a1_tensor, a2_tensor)
