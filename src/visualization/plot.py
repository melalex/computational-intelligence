from matplotlib import pyplot as plt
import numpy as np
from lib.ml.model.neural_net import TrainedNeuralNet
from lib.ml.util.types import ArrayLike


def plot_loss(model: TrainedNeuralNet, size=(12, 6)) -> None:
    plt.figure(figsize=size)
    plt.plot(model.history().loss, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_loss_and_val_loss(model: TrainedNeuralNet, size=(12, 6)) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(model.history().loss, label="Training Loss")
    plt.plot(model.history().validation_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_diff(y_true: ArrayLike, y_predicted: ArrayLike, size=(14, 7)):
    plt.figure(figsize=size)
    plt.plot(y_true, label="Actual", color="blue", alpha=0.7)
    plt.plot(y_predicted, label="Predicted", color="red", alpha=0.7)
    plt.title("Actual vs Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def plot_model_heat_map(model: TrainedNeuralNet):
    params = model.params().learned_params()

    for k, v in params.items():
        plot_params_heat_map(v, k)


def plot_params_heat_map(params: ArrayLike, name: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(params)

    ax.set_xticks(np.arange(params.shape[1]))
    ax.set_yticks(np.arange(params.shape[0]))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(params.shape[0]):
        for j in range(params.shape[1]):
            ax.text(j, i, round(params[i, j], 2), ha="center", va="center", color="w")

    ax.set_title(name)

    fig.tight_layout()
    plt.show()
