from abc import ABC
import math


class LrScheduler(ABC):

    def step(self, val_loss: float) -> None:
        pass

    def get_lr(self) -> float:
        pass


class ConstantLrScheduler(LrScheduler):
    __lr: float

    def __init__(self, lr: float) -> None:
        self.__lr = lr

    def step(self, val_loss: float) -> None:
        pass

    def get_lr(self) -> float:
        return self.__lr


class NoopLrScheduler(LrScheduler):

    def step(self, val_loss: float) -> None:
        pass

    def get_lr(self) -> float:
        return -1


class ReduceLROnPlateau(LrScheduler):
    __former_best_loss: float
    __patience: int
    __intolerable_epochs: int
    __factor: float
    __lr: float
    __min_lr: float

    def __init__(
        self, lr: float, patience: float, factor: float, min_lr: float
    ) -> None:
        self.__former_best_loss = math.inf
        self.__patience = patience
        self.__factor = factor
        self.__lr = lr
        self.__min_lr = min_lr
        self.__intolerable_epochs = 0

    def step(self, val_loss: float) -> None:
        if self.__min_lr >= self.__lr:
            self.__lr = self.__min_lr
            return

        if val_loss >= self.__former_best_loss:
            self.__intolerable_epochs += 1
        else:
            self.__intolerable_epochs = 0
            self.__former_best_loss = val_loss

        if self.__intolerable_epochs >= self.__patience:
            self.__lr *= self.__factor
            self.__intolerable_epochs = 0

    def get_lr(self) -> float:
        return self.__lr


NOOP_LR_SCHEDULER = NoopLrScheduler()
