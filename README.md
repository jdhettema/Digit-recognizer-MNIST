# MNIST Digital Recognizer

## A Neural Network Implementation from Scratch

This repository contains a simple two-layer neural network implementation for recognizing handwritten digits from the MNIST dataset. Built from scratch using only NumPy, this project provides an educational look at the core mathematics and principles behind neural networks.

## Overview

This neural network classifier:
- Processes 28×28 pixel images of handwritten digits (0-9)
- Uses a simple architecture with one hidden layer
- Implements forward and backward propagation algorithms manually
- Achieves reasonable accuracy with minimal complexity

## Architecture

The network consists of:
- Input layer: 784 units (28×28 pixels)
- Hidden layer: 10 units with ReLU activation
- Output layer: 10 units with softmax activation (one for each digit class)

## Mathematical Foundation

### Forward Propagation
- Z¹ = W¹X + b¹
- A¹ = ReLU(Z¹)
- Z² = W²A¹ + b²
- A² = softmax(Z²)

### Backward Propagation
- dZ² = A² - Y (one-hot encoded)
- dW² = (1/m) × dZ² × A¹ᵀ
- db² = (1/m) × sum(dZ²)
- dZ¹ = W²ᵀ × dZ² * ReLU'(Z¹)
- dW¹ = (1/m) × dZ¹ × Xᵀ
- db¹ = (1/m) × sum(dZ¹)

### Parameter Updates
W and b parameters are updated using gradient descent:
- W := W - α × dW
- b := b - α × db

## Key Functions

- `init_params()`: Initializes weights and biases
- `ReLU()` and `ReLU_deriv()`: ReLU activation and its derivative
- `softmax()`: Converts raw outputs to probabilities
- `forward_prop()`: Performs forward propagation
- `one_hot()`: Converts labels to one-hot encoded vectors
- `backward_prop()`: Computes gradients
- `update_params()`: Updates weights and biases
- `gradient_descent()`: Trains the model

## Dataset

The program uses the MNIST dataset, which contains 70,000 grayscale images of handwritten digits. The code:
1. Loads the dataset using pandas
2. Splits it into training and development sets
3. Normalizes pixel values (0-255) to range 0-1

## Usage

1. Ensure you have NumPy, pandas, and matplotlib installed
2. Download the MNIST dataset from Kaggle (train.csv)
3. Run the Jupyter notebook to train and test the model

## Educational Value

This implementation is designed to be educational rather than optimized for performance. It allows you to:
- Understand the mathematical foundations of neural networks
- See how gradient descent optimizes a model
- Observe how different activations (ReLU, softmax) work
- Learn backpropagation through a clear implementation

## Video Tutorial

A detailed explanation of the mathematics and implementation is available in this tutorial video:
[Neural Network from Scratch | Mathematics of Neural Networks](https://youtu.be/w8yWXqWQYmU)

## Performance

With the provided configuration (learning rate = 0.10, 500 iterations), the model achieves reasonable accuracy on the MNIST dataset, demonstrating the power of even simple neural network architectures for image classification tasks.

## Extensions

Possible ways to extend this project:
- Add additional hidden layers
- Experiment with different activation functions
- Implement mini-batch gradient descent
- Add regularization techniques
- Visualize the learning process

## License

MIT License
