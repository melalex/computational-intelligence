from abc import ABC
from logging import Logger
import logging
from tqdm import tqdm_notebook
from tqdm.notebook import tqdm


class ProgressTracker(ABC):

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def open(self, epoch_count):
        return self

    def track(self, i: int, cost: float) -> None:
        pass


class LoggingProgressTracker(ProgressTracker):
    __print_period: int
    __logger: Logger

    def __init__(self, print_period) -> None:
        self.__print_period = print_period
        self.__logger = logging.getLogger("LoggingProgressTracker")

    def track(self, i: int, cost: float) -> None:
        if i % self.__print_period == 0:
            self.__logger.info("Iteration # [ %s ] cost is: %s", i, cost)


class PrintProgressTracker(ProgressTracker):
    __print_period: int

    def __init__(self, print_period) -> None:
        self.__print_period = print_period

    def track(self, i: int, cost: float) -> None:
        if i % self.__print_period == 0:
            print("Iteration # [ %s ] cost is: %s" % (i, cost))


class NoopProgressTracker(ProgressTracker):

    def track(self, i: int, cost: float) -> None:
        pass


class NotebookProgressTracker(ProgressTracker):
    __epoch_count: int
    __pbar: tqdm_notebook

    def __enter__(self):
        self.__pbar = tqdm(total=self.__epoch_count, desc="Progress")
        self.__pbar.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__pbar.__exit__(exc_type, exc_value, traceback)

    def open(self, epoch_count):
        self.__epoch_count = epoch_count
        return self

    def track(self, i: int, cost: float) -> None:
        self.__pbar.update()
        self.__pbar.set_postfix(loss=cost)


NOOP_PROGRESS_TRACKER = NoopProgressTracker()
