import numpy as np
import torch
import matmul_cuda
# from numba import cuda


gridsize = (1024, 1024)
blocksize = (32, 32, 32)


# @cuda.jit
# def matmul(a1_tensor, a2_tensor):
#     # print "tuple"
#     print("type(a1_list):", type(a1_tensor))
#     out = np.zeros((a1_tensor.shape[0], a2_tensor.shape[1]), dtype=np.float32)
#     matmul_cuda.matmul_cpp(a1_tensor, a2_tensor, out)
#     return out

# def matmul(a1, a2):
#     # print "tensor"
#     print("type(a1_list):", type(a1))
#     # print("type(a1_list[0]):", type(a1[0]))
#     if isinstance(a1, list) and isinstance(a2, list):
#         results = []
#         for x, y in zip(a1, a2):
#             out = torch.zeros_like(x)
#             matmul_cuda.matmul_cuda(x, y, out)
#             results.append(out)
#         return results
#     else:
#         print("not suppose to be here")

# def matmul(a1, a2):
#     # print "tensor"
#     print("type(a1_list):", type(a1))
#     return 0

def matmul(a1_tensor, a2_tensor):
    # print "tuple"
    print("type(a1_list):", type(a1_tensor))
    out = np.zeros((32, 32), dtype=np.float32)
    matmul_cuda.matmul_cpp(a1_tensor, a2_tensor, out)
    return out
