from lib.ml.layer.layer_def import Dense, Input, LayerDef
from lib.ml.layer.parameter import CompositeParams, Params, PerceptronParams


def params_from_layer_def(layers: list[LayerDef]) -> Params:
    last_dim = 1
    result = []

    for layer in layers:
        match layer:
            case Input(units_count):
                last_dim = units_count
            case Dense(units_count, use_bias, activation, w_init, b_init):
                weight = w_init.of_shape((units_count, last_dim))
                bias = b_init.of_shape((units_count, 1))

                last_dim = units_count

                result.append(PerceptronParams(weight, bias, activation, use_bias))

    return CompositeParams(result)
