from abc import ABC
from typing import Sequence, SupportsIndex

import numpy as np

from lib.ml.util.types import ArrayLike, ShapeLike


class ArrayInitializer(ABC):

    def of_shape(self, shape: ShapeLike) -> ArrayLike:
        pass


class UniformDistributionInitializer(ArrayInitializer):

    def of_shape(self, shape: ShapeLike) -> ArrayLike:
        return np.random.uniform(low=0, high=1, size=shape)

class ZeroInitializer(ArrayInitializer):

    def of_shape(self, shape: ShapeLike) -> ArrayLike:
        return np.zeros(shape)


UNIFORM_DISTRIBUTION_INITIALIZER = UniformDistributionInitializer()
ZERO_INITIALIZER = ZeroInitializer()
