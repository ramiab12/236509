import torch
import matmul_cuda   # الامتداد المُبنى بالكود أعلاه

def matmul(a1_tensor: torch.Tensor, a2_tensor: torch.Tensor) -> torch.Tensor:
    """
    يستدعي نواة CUDA لِضرب دفعة من مصفوفات 32×32 بالـ float32.
    """


    return matmul_cuda.matmul_cpp(a1_tensor.contiguous(), a2_tensor.contiguous())
