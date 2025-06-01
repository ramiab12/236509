# 236509
Advanced topics in hardware accelerators for deep learning

# 🚀 Fast Batched Matrix Multiplication with CUDA

This project implements a high-performance batched matrix multiplication kernel using CUDA. It is optimized for small fixed-size matrices (e.g., 32×32) and supports large batch sizes (e.g., 1000+ matrices). The kernel uses shared memory tiling and double buffering to overlap memory access and computation, achieving significant speedup over naive implementations.

---

## 📌 Features

- ⚡ High-performance GPU kernel for matrix multiplication
- 🧠 Shared memory tiling
- 🔁 Double-buffered memory (ping-pong buffering)
- 📦 Batch support (`[batch_size, 32, 32]` input tensors)
- ✅ Compatible with PyTorch and integrates as a C++/CUDA extension
- 🧪 Tested using a fixed benchmark harness (`benchmark.py`)

---

## 🛠️ Implementation Highlights

- Each matrix multiplication is handled by a single CUDA block.
- Shared memory is used to tile and load `A` and `B` matrices.
- Two shared memory buffers are used to allow asynchronous loading and computing.
- Each thread computes a single output element `C[row][col]`.
- The kernel is written in C++/CUDA and integrated with Python via PyTorch’s `torch.utils.cpp_extension`.

---

## 📁 Project Structure

├── matmul.py # Python wrapper to call the CUDA kernel
├── matmul_cuda.cpp # C++ binding for PyTorch extension
├── matmul_cuda_kernel.cu # The optimized CUDA kernel
├── setup.py # Build script for PyTorch extension
├── benchmark.py # Fixed benchmarking script (provided)
└── README.md # This file


---

## 🚀 Getting Started

### 1. Install requirements
Make sure you have:
- PyTorch with CUDA support
- A compatible GPU (e.g., RTX 2080 Ti or later)
- A working CUDA compiler (nvcc)

### 2. Build the extension
```bash
python setup.py install
python benchmark.py --num_examples 10000 --num_runs 10

📊 Performance
On an NVIDIA RTX 2080 Ti, the final implementation achieved:

✅ ~0.07 ms per matrix multiplication (32×32)

✅ Very low L1 and L2 error (float32 precision)

✅ Efficient batching and memory reuse

Compared to naive shared memory implementations (~0.5 ms) and WMMA-based kernels (>1.0 ms with high error), this kernel is both fast and accurate.


---

Let me know if you'd like to include:
- Performance plots
- Visual diagrams of tiling/double buffering
- A citation/reference section if used in a course or paper

I can generate those too.
