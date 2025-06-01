

import torch
import matmul_cuda as matmul_cuda # the compiled C++/CUDA extension




def matmul(a1, a2):
    if a1.dtype == torch.float16:
        out = matmul_cuda.matmul_fp16(a1, a2)
    elif a1.dtype == torch.bool:
        out = matmul_cuda.matmul_bool(a1, a2)
    else:
        raise TypeError("Only float16 and bool are supported now")

    return out