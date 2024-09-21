from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from math import exp
from typing import Tuple

import numpy as np
from lib.ml.layer.parameter import CompositeParams, Params, RegressionParams
from lib.ml.loss.loss_function import LossFunction
from lib.ml.util.types import ArrayLike, ShapeLike


@dataclass
class OptimalResult:
    params: Params
    cost: float


class NeuralNetOptimizer(ABC):

    def optimize(
        self,
        epoch: int,
        params: Params,
        x: ArrayLike,
        y_true: ArrayLike,
        loss: LossFunction,
    ) -> OptimalResult:
        pass
