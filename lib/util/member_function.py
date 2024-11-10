from abc import ABC
from dataclasses import dataclass

import numpy as np

from lib.util.types import ArrayLike


class MemberFunction(ABC):

    def apply(self, x: ArrayLike) -> ArrayLike:
        pass

    def backward(self, x: ArrayLike, dz: ArrayLike, lr: float) -> None:
        pass


@dataclass
class BellCurveMF(MemberFunction):
    mean: float
    std: float

    def apply(self, x: ArrayLike) -> ArrayLike:
        return (
            1
            / (self.std * np.sqrt(2 * np.pi))
            * np.exp(-((x - self.mean) ** 2) / (2 * self.std**2))
        )

    def backward(self, x: ArrayLike, dz: ArrayLike, lr: float) -> None:
        pass


@dataclass
class GaussianMF(MemberFunction):
    __mean: float
    __std: float

    def apply(self, x: ArrayLike) -> ArrayLike:
        return np.exp(-0.5 * ((x - self.__mean) / self.__std) ** 2)

    def backward(self, x: ArrayLike, dz: ArrayLike, lr: float) -> None:
        d_mean = dz * self.__apply_d_mean(x)
        d_std = dz * self.__apply_d_std(x)

        self.__mean -= lr * np.mean(d_mean)
        self.__std -= lr * np.mean(d_std)

    def __apply_d_mean(self, x: ArrayLike) -> ArrayLike:
        return ((x - self.__mean) / (self.__std**2)) * self.apply(x)

    def __apply_d_std(self, x: ArrayLike) -> ArrayLike:
        return ((x - self.__mean) ** 2 / (self.__std**3)) * self.apply(x)
