import numpy as np
import matplotlib.pyplot as plt
import click
from datetime import datetime
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from network import NeuralNetwork

def _load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # mean and std dev of MNIST dataset
    ])
    
    train_dataset = MNIST(root='data/', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='data/', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, val_loader

@click.command()
@click.option('--epochs', default=100, help='Number of epochs to train the model')
@click.option('--lr', default=0.01, help='Learning rate for the optimizer')
def train(epochs, lr):
    train_loader, val_loader = _load_data()

    model = NeuralNetwork(
        in_features=28*28,
        hidden_layers=[128, 64],
        out_features=10
    )

    x_train, y_train = [], []
    for x, y in train_loader:
        x_train.append(x.view(-1, 28*28).numpy())
        y_train.append(y.numpy())

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    losses = model.train(x_train, y_train, epochs, lr) 

    train_preds = model.predict(x_train)
    train_pred_cls = np.argmax(train_preds, axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{epochs = }, {lr = }')
    plt.savefig(f"figs/best/Loss.png")

    x_val, y_val = [], []
    for x, y in val_loader:
        x_val.extend(x.view(-1, 28*28).numpy())
        y_val.extend(y.numpy())

    x_val = np.array(x_val)
    y_val = np.array(y_val)

    preds = model.predict(x_val)
    pred_cls = np.argmax(preds, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, pred_cls)
    
    # true vs predicted labels for a subset of validation data
    for y, cls, name in [(y_val, pred_cls, "Val"), (y_train, train_pred_cls, "Train")]:
        plt.figure(figsize=(10, 5))
        plt.scatter(range(len(y[:100])), y[:100], label='True Labels')
        plt.scatter(range(len(cls[:100])), cls[:100], label='Predicted Labels', marker='x')
        plt.xlabel('Sample Index')
        plt.ylabel('Class')
        plt.title(f'True vs Predicted Labels ({name})')
        plt.legend()
        plt.savefig(f"figs/best/Preds_{name}.png")

    # confusion matrix
    cm = confusion_matrix(y_val, pred_cls)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))    
    plt.figure(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)')
    plt.savefig("figs/best/ConfusionMatrix.png")

if __name__ == '__main__':
    train()
