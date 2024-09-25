from abc import ABC
from dataclasses import dataclass
from pathlib import Path
import pickle
from lib.ml.layer.actual_layer import Layer
from lib.ml.util.loss_function import LossFunction
from lib.ml.optimizer.nn_optimizer import NeuralNetOptimizer
from lib.ml.util.progress_tracker import ProgressTracker
from lib.ml.util.types import ArrayLike


@dataclass
class NeuralNetHistory:
    loss: list[float]
    validation_loss: list[float]


@dataclass
class ValidationData:
    x: ArrayLike
    y: ArrayLike


class TrainedNeuralNet(ABC):

    def predict(self, x: ArrayLike) -> ArrayLike:
        pass

    def params(self) -> Layer:
        pass

    def history(self) -> NeuralNetHistory:
        pass

    def train_loss(self) -> float:
        return self.history().loss[-1]


class CompiledNeuralNet(ABC):

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        epochs: int,
        validation_data: ValidationData = None,
        batch_size: int = -1,
    ) -> TrainedNeuralNet:
        pass


class NeuralNet(ABC):

    def compile(
        self,
        optimizer: NeuralNetOptimizer,
        loss: LossFunction,
        progress_tracker: ProgressTracker,
    ) -> CompiledNeuralNet:
        pass


def read_model(path: Path) -> TrainedNeuralNet:
    with path.open("rb") as source:
        return pickle.load(source)


def write_model(model: TrainedNeuralNet, path: Path) -> None:
    with path.open("wb") as dest:
        pickle.dump(model, dest)
