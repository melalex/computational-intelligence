from dataclasses import dataclass
import numpy as np
from lib.ml.layer.layer_def import Dense, Input, LayerDef
from lib.ml.layer.parameter import CompositeParams, Params, RegressionParams
from lib.ml.loss.loss_function import LossFunction
from lib.ml.model.neural_net import (
    CompiledNeuralNet,
    NeuralNet,
    NeuralNetMetrics,
    TrainedNeuralNet,
)
from lib.ml.optimizer.nn_optimizer import NeuralNetOptimizer
from lib.ml.util.progress_tracker import NOOP_PROGRESS_TRACKER, ProgressTracker
from lib.ml.util.types import ArrayLike


class SeqNet(NeuralNet):
    __layers: list[LayerDef]

    def __init__(self, layers) -> None:
        self.__layers = layers

    def compile(
        self,
        optimizer: NeuralNetOptimizer,
        loss: LossFunction,
        progress_tracker: ProgressTracker = NOOP_PROGRESS_TRACKER,
    ) -> CompiledNeuralNet:
        last_dim = 1
        result = []

        for layer in self.__layers:
            match layer:
                case Input(units_count):
                    last_dim = units_count
                case Dense(units_count, activation, init):
                    weight = init.of_shape((units_count, last_dim))
                    bias = np.zeros((units_count, 1))

                    last_dim = units_count

                    result.append(RegressionParams(weight, bias, activation))

        return CompiledSeqNet(
            CompositeParams(result), loss, optimizer, progress_tracker
        )


class CompiledSeqNet(CompiledNeuralNet):
    __params: Params
    __loss: LossFunction
    __optimizer: NeuralNetOptimizer
    __progress_tracker: ProgressTracker

    def __init__(
        self,
        params: Params,
        loss: LossFunction,
        optimizer: NeuralNetOptimizer,
        progress_tracker: ProgressTracker,
    ) -> None:
        self.__params = params
        self.__loss = loss
        self.__optimizer = optimizer
        self.__progress_tracker = progress_tracker

    def fit(
        self, x: ArrayLike, y: ArrayLike, epochs: int, batch_size: int = -1
    ) -> TrainedNeuralNet:
        cost_avg = 0

        for epoch in range(epochs):
            batches = self.__divide_on_mini_batches(x, y, batch_size)
            cost_total = 0
            
            for batch_x, batch_y in batches:
                result = self.__optimizer.optimize(
                    epoch, self.__params, batch_x, batch_y, self.__loss
                )
                self.__params = result.params
                cost_total += result.cost

            cost_avg = cost_total / len(batches)

            self.__progress_tracker.track(epoch, cost_avg)

        return TrainedSeqNet(self.__params, NeuralNetMetrics(cost_avg))

    def __divide_on_mini_batches(
        self, x: ArrayLike, y: ArrayLike, batch_size: int
    ) -> list[tuple[ArrayLike, ArrayLike]]:
        if batch_size == -1:
            return [(x, y)]

        m = x.shape[1]
        result = []

        permutation = list(np.random.permutation(m))
        shuffled_x = x[:, permutation]
        shuffled_y = y[:, permutation]

        for k in range(0, m, batch_size):
            mini_batch_x = shuffled_x[:, k : min(batch_size * (k + 1), m)]
            mini_batch_y = shuffled_y[:, k : min(batch_size * (k + 1), m)]

            result.append((mini_batch_x, mini_batch_y))

        return result


class TrainedSeqNet(TrainedNeuralNet):
    __params: Params
    __metrics: NeuralNetMetrics

    def __init__(self, params, metrics) -> None:
        self.__params = params
        self.__metrics = metrics

    def predict(self, x: ArrayLike) -> ArrayLike:
        return self.__params.apply(x)

    def metrics(self):
        return self.__metrics
