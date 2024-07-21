import numpy as np
import os
import click
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from plot_utils import *
from network import NeuralNetwork
from layers import Linear
from activation_functions import RelU, SoftMax


@click.command()
@click.option('--epochs', default=20)
@click.option('--lr', default=0.01)
def train(epochs, lr):
    train_loader, val_loader = load_data()
    x_train, y_train, x_val, y_val = load_data_np(train_loader, val_loader)

    model = NeuralNetwork([
        Linear(28*28, 392), # 784 -> 784/2
        RelU(),
        Linear(392, 128),
        RelU(),
        Linear(128, 64),
        RelU(),
        Linear(64, 32),
        RelU(),
        Linear(32, 10),
        SoftMax()
    ])

    dir_ = f"figs/{model.id}_layers/{epochs}_epochs-{lr}_lr"
    os.makedirs(dir_, exist_ok=True)
    model.info(dir_, epochs, lr)

    losses = []

    for epoch in range(epochs):
        e_loss = 0.
        for x, y in train_loader:
            x = x.view(-1, 28*28).numpy()
            y = y.numpy()

            e_loss += model.backpropagation(x, y, lr)

        e_loss /= len(train_loader)
        losses.append(e_loss)
        print(f'Epoch {epoch+1}/{epochs} - {e_loss=:.5f}')
    
    # x_train, y_train = get_batch(train_loader)
    train_preds = model.predict(x_train)
    train_pred_cls = np.argmax(train_preds, axis=1)

    # x_val, y_val = get_batch(val_loader)
    val_preds = model.predict(x_val)
    val_pred_cls = np.argmax(val_preds, axis=1)
    
    plot_losses(dir_, losses, epochs, lr)
    plot_predictions(dir_, y_val, val_pred_cls)
    plot_confusion_matrix(dir_, y_val, val_pred_cls, "val")
    plot_confusion_matrix(dir_, y_train, train_pred_cls, "train")
    plot_loss_animation(losses)

    print(f"val_acc: {accuracy_score(y_val, val_pred_cls) * 100}")

def load_data() -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # mean and std dev of MNIST dataset
    ])
    
    train_dataset = MNIST(root='data/', train=True, download=True, transform=transform)
    val_dataset = MNIST(root='data/', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader

def load_data_np(train_loader, val_loader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, y_train = [], []
    for x, y in train_loader:
        x_train.extend(x.view(-1, 28*28).numpy())
        y_train.extend(y.numpy())

    x_val, y_val = [], []
    for x, y in val_loader:
        x_val.extend(x.view(-1, 28*28).numpy())
        y_val.extend(y.numpy())

    return x_train, y_train, x_val, y_val

def get_batch(loader):
    x, y = next(iter(loader))
    x = x.view(-1, 28*28).numpy()
    y = y.numpy()
    
    return x, y

if __name__ == '__main__':
    train()
