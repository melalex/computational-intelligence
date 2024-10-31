from abc import ABC
from dataclasses import dataclass
from typing import Callable

from lib.layer.actual_layer import Layer
from lib.util.loss_function import LossFunction
from lib.util.types import ArrayLike


type LayerSupplier = Callable[[], Layer]


@dataclass
class OptimalResult:
    target: Layer
    loss: float


class NeuralNetOptimizer(ABC):

    def prepare(self, params_supplier: LayerSupplier) -> None:
        pass

    def optimize(
        self,
        epoch: int,
        x: ArrayLike,
        y_true: ArrayLike,
        loss: LossFunction,
    ) -> OptimalResult:
        pass
