from abc import ABC
import numpy as np

from lib.util.types import ArrayLike


class LossFunction(ABC):

    def apply(self, y_true: ArrayLike, y_predicted: ArrayLike) -> float:
        pass

    def apply_derivative(self, y_true: ArrayLike, y_predicted: ArrayLike) -> float:
        pass

    def apply_derivative_elem_wise(self, y_true: ArrayLike, y_predicted: ArrayLike) -> ArrayLike:
        pass



class MeanSquaredError(LossFunction):

    def apply(self, y_true: ArrayLike, y_predicted: ArrayLike) -> float:
        return np.mean(np.square(y_true - y_predicted))

    def apply_derivative(self, y_true: ArrayLike, y_predicted: ArrayLike) -> float:
        return 2 * np.mean(y_predicted - y_true)

    def apply_derivative_elem_wise(self, y_true: ArrayLike, y_predicted: ArrayLike) -> ArrayLike:
        return 2 * (y_predicted - y_true)


MEAN_SQUARED_ERROR = MeanSquaredError()
