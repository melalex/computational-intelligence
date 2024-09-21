from abc import ABC
from logging import Logger
import logging
import math


class ProgressTracker(ABC):

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


NOOP_PROGRESS_TRACKER = NoopProgressTracker()
