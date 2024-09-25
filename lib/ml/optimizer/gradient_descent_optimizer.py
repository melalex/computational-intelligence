from lib.ml.layer.actual_layer import Layer
from lib.ml.optimizer.nn_optimizer import (
    LayerSupplier,
    NeuralNetOptimizer,
    OptimalResult,
)
from lib.ml.util.loss_function import LossFunction
from lib.ml.util.types import ArrayLike


class GradientDescentOptimizer(NeuralNetOptimizer):
    __learning_rate: float
    __momentum: float
    __target: Layer

    def __init__(self, learning_rate: float, momentum: float = 0) -> None:
        self.__learning_rate = learning_rate
        self.__momentum = momentum

    def prepare(self, params_supplier: LayerSupplier) -> None:
        self.__target = params_supplier()

    def optimize(
        self, epoch: int, x: ArrayLike, y_true: ArrayLike, loss: LossFunction
    ) -> OptimalResult:
        return super().optimize(epoch, x, y_true, loss)

    def __optimize_internal(self, target: Layer, x: ArrayLike, y_true: ArrayLike, loss: LossFunction):
        pass
