import numpy as np
import os
import click
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from plot_utils import plot_losses, plot_predictions, plot_confusion_matrix
from network import NeuralNetwork

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # mean and std dev of MNIST dataset
    ])
    
    train_dataset = MNIST(root='data/', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='data/', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    x_train, y_train = [], []
    for x, y in train_loader:
        x_train.extend(x.view(-1, 28*28).numpy())
        y_train.extend(y.numpy())

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_val, y_val = [], []
    for x, y in val_loader:
        x_val.extend(x.view(-1, 28*28).numpy())
        y_val.extend(y.numpy())

    x_val = np.array(x_val)
    y_val = np.array(y_val)

    return x_train, y_train, x_val, y_val

@click.command()
@click.option('--epochs', default=100)
@click.option('--lr', default=0.05)
def train(epochs, lr):
    x_train, y_train, x_val, y_val = load_data()

    model = NeuralNetwork( # v1
        in_features=28*28,
        hidden_layers=[128, 64],
        out_features=10
    )

    # model = NeuralNetwork( # v2
    #     in_features=28*28,
    #     hidden_layers=[256, 128, 64, 32],
    #     out_features=10
    # )

    dir_ = f"figs/{model.id}_layers/best_{epochs}e"
    os.makedirs(dir_, exist_ok=True)

    losses = model.train(x_train, y_train, epochs, lr)
    train_preds, _ = model.predict(x_train)
    train_pred_cls = np.argmax(train_preds, axis=1)

    plot_losses(dir_, losses, epochs, lr)

    preds, _ = model.predict(x_val)
    pred_cls = np.argmax(preds, axis=1)
    
    plot_predictions(dir_, y_val, pred_cls, y_train, train_pred_cls)
    plot_confusion_matrix(dir_, y_val, pred_cls)


if __name__ == '__main__':
    train()
