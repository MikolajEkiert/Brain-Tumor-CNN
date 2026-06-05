# Brain Tumor Detection CNN from Scratch

A Convolutional Neural Network (CNN) built entirely from scratch using only **pure mathematics and NumPy** (no PyTorch, TensorFlow, or Keras) to classify brain MRI scans into two categories: `yes` (tumor present) and `no` (tumor absent).

---

## Project Overview

Deep learning frameworks often obscure the underlying calculus and linear algebra. This project is a scratch implementation designed to showcase the complete forward and backward propagation mathematics behind 2D convolutions, max-pooling, dense layers, dropout regularization, and optimization algorithms.

---

## Features

* **Pure NumPy Engine**: Custom mathematical implementations of CNN layers, activations, and backpropagation.
* **Test-Time Augmentation (TTA)**: Average predictions over original, flipped, and dynamically augmented views during inference for increased test robustness.
* **Regularization & Optimization**:
  * **Adam Optimizer**: Custom implementation with bias correction, first moment (momentum), and second moment (RMSprop).
  * **Dropout**: Inverted dropout layer to combat overfitting.
  * **L2 Regularization**: Weight decay added directly to the loss function gradient.
* **Custom Data Processing**:
  * Gray-scale loading, image resizing ($48 \times 48$), histogram equalization, and custom mini-batch creation.
  * **Dynamic Image Augmentation**: Built-in horizontal flips, random rotations, random scaling, translations, contrast/brightness variance, reduced Gaussian blurring (10% probability), and reduced Gaussian pixel noise ($\sigma = 0.005$).
* **Weighted Binary Cross Entropy**: Configurable weights for positive and negative classes to counter dataset imbalance.

---

## Model Architecture

The network processes grey-scale MRI inputs of size $(48 \times 48 \times 1)$:

1. **Input**: $(m, 48, 48, 1)$
2. **Convolutional Layer 1**: $10$ filters of size $3 \times 3$, stride $2$, padding $1 \rightarrow$ Output: $(m, 24, 24, 10)$
3. **Activation**: ReLU
4. **Max Pooling 1**: Pool size $2 \times 2$, stride $2 \rightarrow$ Output: $(m, 12, 12, 10)$
5. **Convolutional Layer 2**: $20$ filters of size $3 \times 3$, stride $1$, padding $1 \rightarrow$ Output: $(m, 12, 12, 20)$
6. **Activation**: ReLU
7. **Max Pooling 2**: Pool size $2 \times 2$, stride $2 \rightarrow$ Output: $(m, 6, 6, 20)$
8. **Flatten**: Reshape to $(m, 720)$
9. **Dense Layer 1 (Hidden)**: $128$ hidden units $\rightarrow$ Output: $(128, m)$
10. **Activation**: ReLU
11. **Regularization**: Dropout (keep probability = $0.65$)
12. **Dense Layer 2 (Output)**: $1$ output unit $\rightarrow$ Output: $(1, m)$
13. **Activation**: Sigmoid (probability prediction)

## Getting Started

### Prerequisites

Create a virtual environment and install the required dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

