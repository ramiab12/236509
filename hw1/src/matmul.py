def matmul(a1_tensor, a2_tensor):
    assert a1_tensor.shape == a2_tensor.shape
    assert a1_tensor.shape[1:] == (32, 32)
    assert a1_tensor.device.type == 'cpu'

    # Move to GPU for CUDA operation
    a1_cuda = a1_tensor.to('cuda')
    a2_cuda = a2_tensor.to('cuda')

    # Call the CUDA extension
    out_cuda = matmul_cuda.forward(a1_cuda, a2_cuda)

    # Return result to CPU
    return out_cuda.to('cpu')
