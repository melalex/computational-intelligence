from abc import ABC
import math

import numpy as np

from lib.util.types import ArrayLike, ShapeLike


class ArrayInitializer(ABC):

    def of_shape(self, shape: ShapeLike) -> ArrayLike:
        pass


class UniformDistributionInitializer(ArrayInitializer):

    def of_shape(self, shape: ShapeLike) -> ArrayLike:
        return np.random.uniform(low=0, high=1, size=shape)


class ZeroInitializer(ArrayInitializer):

    def of_shape(self, shape: ShapeLike) -> ArrayLike:
        return np.zeros(shape)


class GlorotUniformInitializer(ArrayInitializer):

    def of_shape(self, shape: ShapeLike) -> ArrayLike:
        high = math.sqrt(6) / math.sqrt(shape[0] + shape[1])

        return np.random.uniform(low=-high, high=high, size=shape)


UNIFORM_DISTRIBUTION_INITIALIZER = UniformDistributionInitializer()
ZERO_INITIALIZER = ZeroInitializer()
GLOROT_INITIALIZER = GlorotUniformInitializer()
