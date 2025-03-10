# PyTorch Introduction

This README provides a basic introduction to PyTorch, a popular open-source machine learning framework.

## What is PyTorch?

PyTorch is a Python-based scientific computing package serving two broad purposes:

* **A replacement for NumPy to use the power of GPUs:** PyTorch leverages GPUs for accelerated tensor computations, significantly speeding up machine learning tasks.
* **A deep learning research platform that provides maximum flexibility and speed:** Its dynamic computation graph allows for easy experimentation and debugging, making it a favorite among researchers.

## Key Features

* **Tensors:** PyTorch's fundamental data structure, similar to NumPy arrays but with GPU acceleration.
* **Dynamic Computation Graph:** Defines the computational graph on the fly, enabling flexible model architectures.
* **Automatic Differentiation (Autograd):** Automatically computes gradients for backpropagation, simplifying the training process.
* **GPU Acceleration:** Seamlessly utilizes GPUs for faster computations.
* **Extensive Library:** Offers a rich library of modules for building and training neural networks.
* **Pythonic:** Integrates smoothly with the Python ecosystem.
* **Community and Ecosystem:** Large and active community with numerous pre-trained models and tools.

## Installation

You can install PyTorch using `pip` or `conda`. For specific installation instructions, refer to the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

**Example (using pip):**

```bash
pip install torch torchvision torchaudio

import torch

# Create a tensor
x = torch.tensor([[1, 2], [3, 4]])
print("Tensor x:", x)

# Perform operations
y = x + 2
print("Tensor y (x + 2):", y)

# GPU usage (if available)
if torch.cuda.is_available():
    device = torch.device("cuda")
    x_gpu = x.to(device)
    y_gpu = x_gpu * 2
    print("Tensor y_gpu (x * 2 on GPU):", y_gpu)
else:
    print("CUDA is not available.")

# Automatic differentiation
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 2*x + 1
y.backward() #computes the derivative
print("Derivative of y with respect to x:", x.grad)

