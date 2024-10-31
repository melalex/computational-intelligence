from abc import ABC
from dataclasses import dataclass


from lib.util.activation_function import LINEAR_ACTIVATION, ActivationFunction
from lib.util.array_initializer import (
    GLOROT_INITIALIZER,
    ZERO_INITIALIZER,
    ArrayInitializer,
)
from lib.util.types import ShapeLike


class LayerDef(ABC):
    pass


@dataclass
class Input(LayerDef):
    shape: ShapeLike

    def __init__(self, units_count=None, shape=None):
        self.shape = shape if shape else (units_count,)


@dataclass
class Reshape(LayerDef):
    targe_shape: ShapeLike


@dataclass
class Dense(LayerDef):
    units_count: int
    use_bias: bool = True
    activation_fun: ActivationFunction = LINEAR_ACTIVATION
    weight_initializer: ArrayInitializer = GLOROT_INITIALIZER
    bias_initializer: ArrayInitializer = ZERO_INITIALIZER
