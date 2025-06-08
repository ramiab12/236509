import torch
import matmul_cuda

def matmul(a1_tensor: torch.Tensor, a2_tensor: torch.Tensor) -> torch.Tensor:
    return matmul_cuda.matmul_cpp(a1_tensor.contiguous(), a2_tensor.contiguous())
