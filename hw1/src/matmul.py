import torch
import matmul_cuda

def matmul(a1_tensor, a2_tensor):
    if isinstance(a1_tensor, (list, tuple)):
        results = []
        for a1, a2 in zip(a1_tensor, a2_tensor):
            out = matmul_cuda.forward(a1.to("cuda"), a2.to("cuda"))
            results.append(out.cpu())
        return results
    else:
        return matmul_cuda.forward(a1_tensor.to("cuda"), a2_tensor.to("cuda")).cpu()
