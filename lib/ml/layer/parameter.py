from abc import ABC
from dataclasses import dataclass
import functools

import numpy as np

from lib.ml.util.activation_function import ActivationFunction
from lib.ml.util.types import ArrayLike


class Params(ABC):

    def apply(self, x: ArrayLike) -> ArrayLike:
        pass

    def as_array(self) -> list[ArrayLike]:
        pass

    def as_dict(self) -> dict[str, ArrayLike]:
        pass


@dataclass
class CompositeParams(Params):
    values: list[Params]

    def apply(self, x: ArrayLike) -> ArrayLike:
        return functools.reduce(lambda acc, next: next.apply(acc), self.values, x)

    def as_array(self) -> list[ArrayLike]:
        return functools.reduce(
            lambda acc, next: acc + next.as_array(), self.values, []
        )

    def as_dict(self) -> dict[str, ArrayLike]:
        result = {}

        for i in range(len(self.values)):
            for k, v in self.values[i].as_dict().items():
                result[k + "_" + str(i)] = v

        return result


@dataclass
class PerceptronParams(Params):
    weight: ArrayLike
    bias: ArrayLike
    activation_fun: ActivationFunction
    use_bias: bool

    def apply(self, x: ArrayLike) -> ArrayLike:
        return self.activation_fun.apply(
            np.dot(self.weight, x) + self.bias if self.use_bias else 0
        )

    def as_array(self) -> list[ArrayLike]:
        return [self.weight, self.bias] if self.use_bias else [self.weight]

    def as_dict(self) -> dict[str, ArrayLike]:
        return (
            {"W": self.weight, "b": self.bias} if self.use_bias else {"W": self.weight}
        )
