import torch
import matmul_cuda

def matmul(a1_tensor, a2_tensor):
    """
    Optimized boolean matrix multiplication for benchmark
    Input: [10000, 32, 32] boolean tensors on GPU
    Output: [10000, 32, 32] boolean tensor on GPU
    """
    return matmul_cuda.matmul_cpp(a1_tensor, a2_tensor)