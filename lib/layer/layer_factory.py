import numpy as np
from lib.layer.layer_def import Dense, Input, LayerDef, Reshape
from lib.layer.actual_layer import (
    ActivationLayer,
    CompositeLayer,
    Layer,
    BiasedWeightLayer,
    ReshapeLayer,
    WeightLayer,
)


def create_layer_from_def(layers: list[LayerDef]) -> Layer:
    last_shape = (1,)
    result = []

    for layer in layers:
        match layer:
            case Input(shape):
                last_shape = shape
            case Reshape(target_shape):
                result.append(ReshapeLayer(last_shape, target_shape))

                last_shape = np.zeros(last_shape).reshape(target_shape).shape
            case Dense(units_count, use_bias, activation, w_init, b_init):
                shape = (units_count, last_shape[0])

                weight = w_init.of_shape((units_count, last_shape[0]))

                last_shape = shape

                if use_bias:
                    bias = b_init.of_shape((units_count, 1))
                    result.append(BiasedWeightLayer(weight, bias))
                else:
                    result.append(WeightLayer(weight))

                result.append(ActivationLayer(activation))

    return CompositeLayer(result)
