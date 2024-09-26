from abc import ABC
import numpy as np

from lib.ml.util.types import ArrayLike


class LossFunction(ABC):

    def apply(self, y_true: ArrayLike, y_predicted: ArrayLike) -> float:
        pass

    def apply_derivative(self, y_true: ArrayLike, y_predicted: ArrayLike) -> float:
        pass


class MeanSquaredError(LossFunction):

    def apply(self, y_true: ArrayLike, y_predicted: ArrayLike) -> float:
        return np.mean(np.square(y_true - y_predicted))

    def apply_derivative(self, y_true: ArrayLike, y_predicted: ArrayLike) -> float:
        return 2 * np.mean(y_predicted - y_true)


MEAN_SQUARED_ERROR = MeanSquaredError()
