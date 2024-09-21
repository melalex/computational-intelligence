from abc import ABC
from dataclasses import dataclass
from typing import Sequence, SupportsIndex

import numpy as np

from lib.ml.activation.activation_function import LINEAR_ACTIVATION, ActivationFunction
from lib.ml.initializer.array_initializer import (
    UNIFORM_DISTRIBUTION_INITIALIZER,
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
    activation_fun: ActivationFunction = LINEAR_ACTIVATION
    array_initializer: ArrayInitializer = UNIFORM_DISTRIBUTION_INITIALIZER
