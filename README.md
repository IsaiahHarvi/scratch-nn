# Neural Network from Scratch (scratch-nn)

This project implements a neural network from scratch using only NumPy to classify the MNIST dataset. The network employs a feedforward architecture with ReLU activation functions in the hidden layers and a Softmax activation function in the output layer. Training is performed using backpropagation and cross-entropy loss.

## Best Result
The 5-layer MLP is given 20 epochs and a learning rate of 0.01. These figures are available under [figs](scratch-nn/figs/5_layers/20_epochs-0.01_lr).

### Confusion Matrix
![Confusion Matrix](https://github.com/IsaiahHarvi/scratch-nn/blob/77d235a391b6a3fb3378408e85f6812683bb09ab/scratch-nn/figs/5_layers/20_epochs-0.01_lr/ConfusionMatrix_val.png)

### Loss Plot GIF
![Loss Plot gif](https://github.com/IsaiahHarvi/scratch-nn/blob/77d235a391b6a3fb3378408e85f6812683bb09ab/scratch-nn/figs/5_layers/20_epochs-0.01_lr/training_loss.gif)


---

## Network Architecture
The 5-layer (excluding Activation Functions, of course) MLP consists of the following layers:

- **0 | Linear(784, 392)**: 784 neurons (one for each pixel in a 28x28 image, flattened)
- **1 | RelU**            : Introduces non-linearity
- **2 | Linear(392, 128)**
- **3 | RelU**
- **4 | Linear(128, 64)**
- **5 | RelU**
- **6 | Linear(64, 32)**
- **7 | RelU**
- **8 | Linear(32, 10)**
- **9 | SoftMax**

*Architectures are with all related figures as [model_info.txt](scratch-nn/figs/5_layers/20_epochs-0.01_lr/_model_info.txt)* 
### Activation Functions

- **ReLU (Rectified Linear Unit)**: Applied to the hidden layers to introduce non-linearity, allowing the network to learn more complex patterns.
- **Softmax**: Applied to the output layer to produce a probability distribution over the 10-digit classes.

### Training Details

- **Backpropagation**: The network uses backpropagation to adjust weights and biases based on the error between predicted and actual labels.
- **Loss Function**: Cross-entropy loss is used to measure the performance of the classification model, providing a probability value between 0 and 1.

---

## Used Libraries

The only library utilized in the construction of the model is NumPy. Other libraries are used for training and plotting:
- `numpy`
- `matplotlib`
- `click` (for command line arguments in `train.py`)
- `torchvision` (for downloading the MNIST dataset)
- `torch` (for data loaders in `train.py` and data transformations)
- `scikit-learn` (for confusion matrix plot)

