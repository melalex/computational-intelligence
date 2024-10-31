import numpy as np
from lib.layer.layer_def import LayerDef
from lib.layer.actual_layer import CompositeLayer, Layer
from lib.layer.layer_factory import create_layer_from_def
from lib.util.loss_function import LossFunction
from lib.model.neural_net import (
    CompiledNeuralNet,
    NeuralNet,
    NeuralNetHistory,
    TrainedNeuralNet,
    ValidationData,
)
from lib.optimizer.nn_optimizer import NeuralNetOptimizer
from lib.util.lr_scheduler import NOOP_LR_SCHEDULER, LrScheduler
from lib.util.progress_tracker import NOOP_PROGRESS_TRACKER, ProgressTracker
from lib.util.types import ArrayLike


class SeqNet(NeuralNet):
    __layers: list[LayerDef]

    def __init__(self, layers) -> None:
        self.__layers = layers

    def compile(
        self,
        optimizer: NeuralNetOptimizer,
        loss: LossFunction,
        lr_scheduler: LrScheduler = NOOP_LR_SCHEDULER,
        progress_tracker: ProgressTracker = NOOP_PROGRESS_TRACKER,
    ) -> CompiledNeuralNet:
        optimizer.prepare(lambda: create_layer_from_def(self.__layers))

        return CompiledSeqNet(loss, optimizer, progress_tracker, lr_scheduler)


class CompiledSeqNet(CompiledNeuralNet):
    __loss: LossFunction
    __optimizer: NeuralNetOptimizer
    __progress_tracker: ProgressTracker
    __lr_scheduler: LrScheduler

    def __init__(
        self,
        loss: LossFunction,
        optimizer: NeuralNetOptimizer,
        progress_tracker: ProgressTracker,
        lr_scheduler: LrScheduler,
    ) -> None:
        self.__loss = loss
        self.__optimizer = optimizer
        self.__progress_tracker = progress_tracker
        self.__lr_scheduler = lr_scheduler

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        epochs: int,
        validation_data: ValidationData = None,
        batch_size: int = -1,
    ) -> TrainedNeuralNet:
        if epochs == 0:
            return None

        result = None
        loss_history = []
        val_loss_history = []

        with self.__progress_tracker.open(epochs) as tracker:
            for epoch in range(epochs):
                batches = self.__divide_on_mini_batches(x, y, batch_size)
                loss_total = 0

                for batch_x, batch_y in batches:
                    opt_result = self.__optimizer.optimize(
                        epoch, batch_x, batch_y, self.__loss
                    )
                    result = opt_result.target
                    loss_total += opt_result.loss

                loss_avg = loss_total / len(batches)

                loss_history.append(loss_avg)

                if validation_data:
                    y_predicted = result.apply(validation_data.x)
                    validation_loss = self.__loss.apply(validation_data.y, y_predicted)
                    val_loss_history.append(validation_loss)

                    if self.__lr_scheduler:
                        self.__lr_scheduler.step(validation_loss)

                    tracker.track_with_validation(
                        epoch, loss_avg, validation_loss, self.__lr_scheduler.get_lr()
                    )
                else:
                    tracker.track(epoch, loss_avg)

        return TrainedSeqNet(
            result,
            NeuralNetHistory(loss_history, val_loss_history),
        )

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
    __params: Layer
    __history: NeuralNetHistory

    def __init__(self, params, history) -> None:
        self.__params = params
        self.__history = history

    def predict(self, x: ArrayLike) -> ArrayLike:
        return self.__params.apply(x)

    def history(self) -> NeuralNetHistory:
        return self.__history

    def params(self) -> Layer:
        return self.__params
