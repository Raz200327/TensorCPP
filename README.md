# TensorCPP

> **Lightweight C++ tensor library built for PyTorch compatibility**

[![C++](https://img.shields.io/badge/C++-11-blue.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build](https://img.shields.io/badge/Build-Makefile-orange.svg)](Makefile)

## 🚀 Features

### 🔢 Core Tensor Operations
- **Matrix Multiplication**: Efficient `matMul()` operation with dimension validation
- **Element-wise Addition**: Support for tensor addition with broadcasting
- **Random Initialization**: `randInit()` for weight initialization
- **Memory Management**: Smart pointer-based memory handling with automatic cleanup

### 🧠 Neural Network Activation Functions
- **ReLU Activation**: `ReLU()` function for non-linear transformations
- **Softmax**: `Softmax()` for probability distribution outputs
- **Broadcasting Support**: Automatic handling of different tensor dimensions

### 📁 Data I/O Capabilities
- **CSV Loading**: Direct tensor creation from CSV files
- **Weight Loading**: Seamless integration with PyTorch-trained models
- **Flexible Dimensions**: Support for arbitrary 2D tensor shapes

### 🔧 Development Tools
- **Makefile Build System**: Simple compilation with `make run`

## 📦 Project Structure

```
TensorCPP/
├── 📁 linalg/
│   └── tensor.h          # Core tensor class definition
├── source.cpp            # Tensor implementation
├── main.cpp              # Example neural network inference
└── Makefile              # Build configuration
```

## 🛠️ Quick Start

### Building the Project
```bash
# Compile and run the neural network inference
make run

# Clean build artifacts
make clean
```

### Basic Usage Example
```cpp
#include "linalg/tensor.h"

// Create tensors
Tensor input("./weights/test_data.csv", 10000, 784);
Tensor weights("./weights/net_1_weight.csv", 784, 128);
Tensor bias("./weights/net_1_bias.csv", 1, 128);

// Neural network forward pass
Tensor layer = input.matMul(weights);
layer = layer + bias;
layer.ReLU();
layer.Softmax();

std::cout << "Output: " << layer << std::endl;
```

## 🧪 Neural Network Example

The library includes a complete MNIST neural network implementation:

- **Architecture**: 784 → 128 → 64 → 10 (fully connected)
- **Activations**: ReLU for hidden layers, Softmax for output
- **Pre-trained**: Includes weights trained on MNIST dataset
- **Inference**: Ready-to-run classification on test data

## 🔗 PyTorch Integration

The `test.py` script demonstrates:
- PyTorch model training on MNIST
- Weight export to CSV format
- Data preprocessing and normalization
- Model evaluation and accuracy reporting

## 📊 Performance

- **Memory Efficient**: Shared pointer-based tensor storage
- **Type Safe**: Strong typing with dimension validation
- **Extensible**: Easy to add new operations and functions
- **Compatible**: Seamless integration with PyTorch workflows

## 🤝 Contributing

This is a lightweight educational implementation. Feel free to:
- Add new tensor operations
- Implement additional activation functions
- Optimize performance
- Add GPU support

---

**Built with ❤️ for learning and experimentation**
