import torch
import matmul_cuda

def matmul(a1_tensor, a2_tensor):
    a1=a1_tensor.to("cuda")
    a2=a2_tensor.to("cuda")
    return matmul_cuda.forward(a1,a2)