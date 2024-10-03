# MNIST Classification with NumPy and TensorFlow

This project demonstrates how to train a neural network for classifying handwritten digits from the MNIST dataset using **NumPy** for key operations and **TensorFlow** for data handling. The project avoids using high-level APIs, offering an educational perspective on implementing machine learning algorithms from scratch.

## Project Overview

- **Dataset**: MNIST dataset, which contains 70,000 28x28 grayscale images of handwritten digits (0-9).
- **Goal**: Build and train a neural network to classify these digits.
- **Technologies**: NumPy, TensorFlow, Matplotlib (for visualizations).

## Features

1. **Custom One-Hot Encoding**: Implementation of one-hot encoding using NumPy.
2. **CrossEntropy Loss**: Custom implementation of the cross-entropy loss function.
3. **Activation Functions**:
   - **LeakyReLU**: Rectified linear activation allowing for a small gradient when the unit is not active.
   - **Softmax**: Output activation function used for multi-class classification.
4. **Accuracy Function**: Manual computation of accuracy based on predictions.
5. **Training Loop**: Step-by-step implementation of the forward pass, loss calculation, backpropagation, and gradient descent.

## Project Architecture

The architecture of the neural network consists of:

1. **Input Layer**: Flattening the 28x28 images into a 1D vector of size 784.
2. **Hidden Layers**: Two dense layers with **LeakyReLU** activations.
3. **Output Layer**: A dense layer with **Softmax** activation for multi-class classification.
4. **Loss Function**: Custom **CrossEntropy** for measuring the prediction error.
5. **Optimization**: Gradient descent with manually computed gradients.

## Machine Learning Techniques

- **Forward Propagation**: The input data is passed through the network, with activations computed at each layer.
- **Backpropagation**: Gradients are computed layer by layer, and weights are updated using gradient descent.
- **Mini-Batch Gradient Descent**: Instead of processing the entire dataset at once, training is done in smaller batches to improve convergence and reduce memory usage.
- **Weight Initialization**: Weights are initialized using random values from a standard normal distribution to break symmetry.

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install numpy tensorflow matplotlib
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook mnist_with_numpy_and_TF.ipynb
   ```

## Results

- **Training Accuracy**: After training, the model achieves competitive accuracy on the MNIST dataset.
- **Visualization**: The notebook provides visualization of training progress and final accuracy on test data.
