from abc import ABC
from dataclasses import dataclass

import numpy as np

from lib.util.types import ArrayLike


class MemberFunction(ABC):

    def apply(self, x: ArrayLike) -> ArrayLike:
        pass

    def get_mean(self) -> float:
        pass

    def get_std(self) -> float:
        pass

    def set_mean(self, mean: float) -> None:
        pass

    def set_std(self, std: float) -> None:
        pass

    def apply_d_mean(self, x: ArrayLike) -> float:
        pass

    def apply_d_std(self, x: ArrayLike) -> float:
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


@dataclass
class GaussianMF(MemberFunction):
    mean: float
    std: float

    def apply(self, x: ArrayLike) -> ArrayLike:
        return np.exp(-0.5 * ((x - self.mean) / self.std) ** 2)

    def get_mean(self) -> float:
        return self.mean

    def get_std(self) -> float:
        return self.std

    def set_mean(self, mean: float) -> None:
        self.mean = mean

    def set_std(self, std: float) -> None:
        self.std = std

    def apply_d_mean(self, x: ArrayLike) -> float:
        return ((x - self.mean) / (self.std**2)) * self.apply(x)

    def apply_d_std(self, x: ArrayLike) -> float:
        return ((x - self.mean) ** 2 / (self.std**3)) * self.apply(x)
