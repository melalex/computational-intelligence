from abc import ABC
from dataclasses import dataclass
from typing import Callable

from lib.ml.layer.actual_layer import Layer
from lib.ml.util.loss_function import LossFunction
from lib.ml.util.types import ArrayLike


type ParamsSupplier = Callable[[], Layer]


@dataclass
class OptimalResult:
    params: Layer
    cost: float


class NeuralNetOptimizer(ABC):

    def prepare(self, params_supplier: ParamsSupplier) -> None:
        pass

    def optimize(
        self,
        epoch: int,
        x: ArrayLike,
        y_true: ArrayLike,
        loss: LossFunction,
    ) -> OptimalResult:
        pass
