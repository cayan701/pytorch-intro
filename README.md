# <span style="color: #FF5733;">PyTorch</span> <span style="color: #3498DB;">Introduction</span>

This README provides a basic introduction to <span style="color: #FF5733;">PyTorch</span>, a popular open-source machine learning framework.

## <span style="color: #2ECC71;">What is PyTorch?</span>

<span style="color: #FF5733;">PyTorch</span> is a Python-based scientific computing package serving two broad purposes:

* <span style="color: #E67E22;">**A replacement for NumPy to use the power of GPUs:**</span> <span style="color: #FF5733;">PyTorch</span> leverages GPUs for accelerated tensor computations, significantly speeding up machine learning tasks.
* <span style="color: #E67E22;">**A deep learning research platform that provides maximum flexibility and speed:**</span> Its dynamic computation graph allows for easy experimentation and debugging, making it a favorite among researchers.

## <span style="color: #9B59B6;">Key Features</span>

* <span style="color: #3498DB;">**Tensors:**</span> <span style="color: #808080;">PyTorch's fundamental data structure, similar to NumPy arrays but with GPU acceleration.</span>
* <span style="color: #3498DB;">**Dynamic Computation Graph:**</span> <span style="color: #808080;">Defines the computational graph on the fly, enabling flexible model architectures.</span>
* <span style="color: #3498DB;">**Automatic Differentiation (Autograd):**</span> <span style="color: #808080;">Automatically computes gradients for backpropagation, simplifying the training process.</span>
* <span style="color: #3498DB;">**GPU Acceleration:**</span> <span style="color: #808080;">Seamlessly utilizes GPUs for faster computations.</span>
* <span style="color: #3498DB;">**Extensive Library:**</span> <span style="color: #808080;">Offers a rich library of modules for building and training neural networks.</span>
* <span style="color: #3498DB;">**Pythonic:**</span> <span style="color: #808080;">Integrates smoothly with the Python ecosystem.</span>
* <span style="color: #3498DB;">**Community and Ecosystem:**</span> <span style="color: #808080;">Large and active community with numerous pre-trained models and tools.</span>

## <span style="color: #F39C12;">Installation</span>

You can install <span style="color: #FF5733;">PyTorch</span> using `pip` or `conda`. For specific installation instructions, refer to the official <span style="color: #FF5733;">PyTorch</span> website: [<span style="color: #2980B9;">https://pytorch.org/get-started/locally/</span>](<span style="color: #2980B9;">https://pytorch.org/get-started/locally/</span>)

**<span style="color: #E74C3C;">Example (using pip):</span>**

```bash
<span style="color: #1ABC9C;">pip install torch torchvision torchaudio</span>
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

import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Instantiate the network
net = SimpleNet()

# Define loss function and optimizer
criterion = nn.BCELoss() #Binary Cross Entropy Loss
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Example training data
inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
targets = torch.tensor([[0.0], [1.0], [1.0]])

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Trained model outputs:", net(inputs))
