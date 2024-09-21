from abc import ABC
from dataclasses import dataclass
import functools

import numpy as np

from lib.ml.activation.activation_function import ActivationFunction
from lib.ml.util.types import ArrayLike


class Params(ABC):

    def apply(self, x: ArrayLike) -> ArrayLike:
        pass


@dataclass
class CompositeParams(Params):
    values: list[Params]

    def apply(self, x: ArrayLike) -> ArrayLike:
        return functools.reduce(lambda acc, next: next.apply(acc), self.values, x)


@dataclass
class RegressionParams(Params):
    weight: ArrayLike
    bias: ArrayLike
    activation_fun: ActivationFunction

    def apply(self, x: ArrayLike) -> ArrayLike:
        return self.activation_fun.apply(np.dot(self.weight, x) + self.bias)
