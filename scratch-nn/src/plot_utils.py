import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

matplotlib.use('TkAgg')  # Use Tkinter as the graphical backend for animations

def plot_losses(dir_: str, losses: list, epochs: int, lr: float) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{epochs = }, {lr = }')
    plt.savefig(f"{dir_}/Loss.png")

def plot_predictions(dir_: str, y_val, pred_cls) -> None:
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(y_val[:100])), y_val[:100], label='True Labels')
    plt.scatter(range(len(pred_cls[:100])), pred_cls[:100], label='Predicted Labels', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.title(f'True vs Predicted Labels (Validation Split)')
    plt.legend()
    plt.savefig(f"{dir_}/Preds_Val.png")

def plot_confusion_matrix(dir_: str, y: np.ndarray, pred_cls: np.ndarray, suffix: str) -> None:
    cm = confusion_matrix(y, pred_cls)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))    
    plt.figure(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy_score(y, pred_cls) * 100:.2f}%)')
    plt.savefig(f"{dir_}/ConfusionMatrix_{suffix}.png")

def plot_loss_animation(losses: list[float]) -> None:
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'r-')
    ax.set_xlim(0, len(losses))
    ax.set_ylim(0, max(losses) if losses else 1)
    ax.set_title('Training Loss Over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    xdata, ydata = [], []

    def init():
        line.set_data([], [])
        return line,

    def update(epoch):
        xdata.append(epoch)
        ydata.append(losses[epoch])
        line.set_data(xdata, ydata)
        ax.set_xlim(0, len(losses))
        ax.set_ylim(0, max(losses) if losses else 1)
        return line,

    ani = FuncAnimation(fig, update, frames=range(len(losses)), init_func=init, blit=True)
    ani.save(f"figs/training_loss.mp4", writer="ffmpeg", fps=10)
