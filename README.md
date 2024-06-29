# Neural Network from Scratch (scratch-nn)

This project implements a neural network from scratch using only NumPy to classify the MNIST dataset. The network employs a feedforward architecture with ReLU activation functions in the hidden layers and a Softmax activation function in the output layer. Training is performed using backpropagation and cross-entropy loss.

## Best Result
The 4-layer MLP, given 50 epochs and a learning rate of 0.1. These figures are available under [figs](scratch-nn/figs/4_layers/).

### Confusion Matrix
![Confusion Matrix](https://github.com/IsaiahHarvi/scratch-nn/blob/main/scratch-nn/figs/4_layers/best_50e/ConfusionMatrix.png)

### Loss Plot GIF
![Loss Plot gif](https://github.com/IsaiahHarvi/scratch-nn/blob/main/scratch-nn/figs/4_layers/training_loss.gif)

*The GIF above shows the training loss over 100 epochs.*

---

## Network Architecture

The 4-layer MLP consists of the following layers:

- **Input Layer**: 784 neurons (one for each pixel in a 28x28 image, flattened)
- **First Hidden Layer**: 128 neurons with ReLU activation
- **Second Hidden Layer**: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons (one for each digit class 0-9) with Softmax activation

### Activation Functions

- **ReLU (Rectified Linear Unit)**: Applied to the hidden layers to introduce non-linearity, allowing the network to learn more complex patterns.
- **Softmax**: Applied to the output layer to produce a probability distribution over the 10 digit classes.

### Training Details

- **Backpropagation**: The network uses backpropagation to adjust weights and biases based on the error between predicted and actual labels.
- **Loss Function**: Cross-Entropy Loss is used to measure the performance of the classification model, providing a probability value between 0 and 1.

---

## Used Libraries

The only library utilized in the construction of the model is NumPy. Other libraries are used for training and plotting:
- `numpy`
- `matplotlib`
- `click` (for command line arguments in `train.py`)
- `torchvision` (for downloading the MNIST dataset)
- `torch` (for data loaders in `train.py`)
- `scikit-learn` (for confusion matrix plot)

