# scratch-nn

This project implements a neural network from scratch, utilizing only Numpy, to classify the MNIST dataset. 

## Network Architecture
- **4 Layers**:
  - **Input Layer**: 784 neurons (28x28 pixels flattened)
  - **First Hidden Layer**: 128 neurons with ReLU activation
  - **Second Hidden Layer**: 64 neurons with ReLU activation
  - **Output Layer**: 10 neurons (one for each digit class 0-9) with SoftMax activation
- **Activation Functions**:
  - **ReLU (Rectified Linear Unit)**: Applied to the hidden layers to introduce non-linearity
  - **Softmax**: Applied to the output layer to produce a probability distribution over the 10 classes
- **Backpropagation**: The network uses backpropagation to adjust weights based on the error between predicted and actual labels
- **Loss Function**:
  - **Cross-Entropy Loss**: Used to measure the performance of the classification model whose output is a probability value between 0 and 1.

### Libraries
- `numpy`
- `matplotlib`
- `click`
- `torchvision`  # for MNIST
- `torch`        # for dataloaders in train.py
- `scikit-learn` # for a confusion matrix
