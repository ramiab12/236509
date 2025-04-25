import torch
from torch.utils.cpp_extension import load

# Load compiled CUDA extension
matmul_cuda = load(
    name='matmul_cuda',
    sources=['src/matmul_cuda.cpp', 'src/matmul_cuda_kernel.cu'],
    verbose=True
)

def matmul(a1_tensor, a2_tensor):
    # Handle list inputs (convert to batched tensor)
    if isinstance(a1_tensor, list):
        a1_tensor = torch.stack(a1_tensor, dim=0)
    if isinstance(a2_tensor, list):
        a2_tensor = torch.stack(a2_tensor, dim=0)
    
    # Ensure tensors are on CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a1_cuda = a1_tensor.to(device)
    a2_cuda = a2_tensor.to(device)
    
    # Compute and return result on CPU
    out_cuda = matmul_cuda.matmul_forward(a1_cuda, a2_cuda, use_optimized=True)
    return out_cuda.cpu()