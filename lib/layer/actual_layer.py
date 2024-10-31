from abc import ABC
from dataclasses import dataclass
import functools

import numpy as np

from lib.util.activation_function import ActivationFunction
from lib.util.types import ArrayLike, ShapeLike


class Layer(ABC):

    def apply(self, x: ArrayLike) -> ArrayLike:
        pass

    def unapply(self, y: ArrayLike) -> ArrayLike:
        pass

    def learned_params(self) -> dict[str, ArrayLike]:
        return {}


@dataclass
class CompositeLayer(Layer):
    values: list[Layer]

    def apply(self, x: ArrayLike) -> ArrayLike:
        return functools.reduce(lambda acc, next: next.apply(acc), self.values, x)

    def learned_params(self) -> dict[str, ArrayLike]:
        result = {}

        for i in range(len(self.values)):
            for k, v in self.values[i].learned_params().items():
                result[k + "_" + str(i)] = v

        return result


@dataclass
class ReshapeLayer(Layer):
    prev_layer_shape: ShapeLike
    target_shape: ShapeLike

    def apply(self, x: ArrayLike) -> ArrayLike:
        # check whether input consist of single data point or not
        if self.prev_layer_shape == x.shape:
            return x.reshape(self.target_shape)
        else:
            return x.reshape(self.target_shape + (x.shape[-1],))

    def unapply(self, y: ArrayLike) -> ArrayLike:
        # check whether input consist of single data point or not
        if self.target_shape == y.shape:
            return y.reshape(self.prev_layer_shape)
        else:
            return y.reshape(self.prev_layer_shape + (y.shape[-1],))


@dataclass
class ActivationLayer(Layer):
    fun: ActivationFunction

    def apply(self, x: ArrayLike) -> ArrayLike:
        return self.fun.apply(x)


@dataclass
class BiasedWeightLayer(Layer):
    weight: ArrayLike
    bias: ArrayLike

    def apply(self, x: ArrayLike) -> ArrayLike:
        return np.dot(self.weight, x) + self.bias

    def learned_params(self) -> dict[str, ArrayLike]:
        return {"W": self.weight, "b": self.bias}


@dataclass
class WeightLayer(Layer):
    weight: ArrayLike

    def apply(self, x: ArrayLike) -> ArrayLike:
        return np.dot(self.weight, x)

    def learned_params(self) -> dict[str, ArrayLike]:
        return {"W": self.weight}
