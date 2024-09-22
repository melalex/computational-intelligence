from abc import ABC
from dataclasses import dataclass
from typing import Sequence, SupportsIndex

import numpy as np

from lib.ml.util.activation_function import LINEAR_ACTIVATION, ActivationFunction
from lib.ml.util.array_initializer import (
    UNIFORM_DISTRIBUTION_INITIALIZER,
    ZERO_INITIALIZER,
    ArrayInitializer,
)


class LayerDef(ABC):
    pass


@dataclass
class Input(LayerDef):
    units_count: int


@dataclass
class Dense(LayerDef):
    units_count: int
    use_bias: bool = True
    activation_fun: ActivationFunction = LINEAR_ACTIVATION
    weight_initializer: ArrayInitializer = UNIFORM_DISTRIBUTION_INITIALIZER
    bias_initializer: ArrayInitializer = ZERO_INITIALIZER
