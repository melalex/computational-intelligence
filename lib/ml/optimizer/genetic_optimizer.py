from copy import deepcopy
from math import exp
from typing import Tuple

import numpy as np
from lib.ml.layer.parameter import CompositeParams, Params, RegressionParams
from lib.ml.loss.loss_function import LossFunction
from lib.ml.optimizer.nn_optimizer import NeuralNetOptimizer, OptimalResult
from lib.ml.util.types import ArrayLike, ShapeLike


class GeneticAlgorithmNeuralNetOptimizer(NeuralNetOptimizer):
    __population_size: int
    __mutation_rate: float
    __alpha: float
    __population: list[Params]

    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        alpha: float,
    ):
        self.__population_size = population_size
        self.__mutation_rate = mutation_rate
        self.__alpha = alpha
        self.__population = []

    def optimize(
        self,
        epoch: int,
        params: Params,
        x: ArrayLike,
        y_true: ArrayLike,
        loss: LossFunction,
    ) -> OptimalResult:
        if not self.__population:
            self.__population = [
                deepcopy(params) for _ in range(self.__population_size)
            ]

        self.__breed(epoch, x, y_true, loss)
        self.__select(x, y_true, loss)

        candidate = self.__population[0]

        return OptimalResult(
            candidate, self.__calculate_cost(candidate, x, y_true, loss)
        )

    def __breed(
        self, epoch: int, x: ArrayLike, y_true: ArrayLike, loss: LossFunction
    ) -> None:
        fitnesses = [self.__fitness(p, x, y_true, loss) for p in self.__population]
        fitnesses_sum = np.sum(fitnesses)
        breed_prop = fitnesses / (fitnesses_sum + 1e-6)
        parents = np.random.choice(
            a=self.__population,
            size=self.__population_size,
            replace=False,
            p=breed_prop,
        )

        for i in range(0, self.__population_size, 2):
            if i + 1 < self.__population_size:
                child1, child2 = self.__crossover(parents[i], parents[i + 1])

                self.__population.append(self.__mutate(epoch, child1))
                self.__population.append(self.__mutate(epoch, child2))

    def __fitness(
        self, params: Params, x: ArrayLike, y_true: ArrayLike, loss: LossFunction
    ) -> float:
        return 1.0 - self.__calculate_cost(params, x, y_true, loss)

    def __mutate(self, epoch: int, params: Params) -> Params:
        match params:
            case CompositeParams(values):
                return CompositeParams([self.__mutate(epoch, it) for it in values])
            case RegressionParams(weight, bias, fun):
                exp_term = self.__mutation_rate * exp(-self.__alpha * epoch)
                weight_mutation = self.__mutation_of_shape(exp_term, weight.shape)
                bias_mutation = self.__mutation_of_shape(exp_term, bias.shape)
                return RegressionParams(
                    weight + weight_mutation,
                    bias + bias_mutation,
                    fun,
                )

    def __mutation_of_shape(self, exp_term: float, shape: ShapeLike) -> ArrayLike:
        return (
            np.random.uniform(
                low=-self.__mutation_rate, high=self.__mutation_rate, size=shape
            )
            * exp_term
        )

    def __crossover(self, father: Params, mother: Params) -> tuple[Params, Params]:
        match father, mother:
            case CompositeParams(father_values), CompositeParams(mother_values):
                children = [
                    self.__crossover(f, m) for f, m in zip(father_values, mother_values)
                ]

                brothers = CompositeParams([it[0] for it in children])
                sisters = CompositeParams([it[1] for it in children])

                return brothers, sisters
            case RegressionParams(father_weight, father_bias, fun), RegressionParams(
                mother_weight, mother_bias, _
            ):
                brother_weight, sister_weight = self.__crossover_arr(
                    father_weight, mother_weight
                )
                brother_bias, sister_bias = self.__crossover_arr(
                    father_bias, mother_bias
                )

                brother = RegressionParams(brother_weight, brother_bias, fun)
                sister = RegressionParams(sister_weight, sister_bias, fun)

                return brother, sister

    def __crossover_arr(
        self, father: ArrayLike, mother: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        brother = deepcopy(father)
        sister = deepcopy(mother)

        mask = self.__crossover_mask(father.shape)
        temp = deepcopy(brother[mask])
        brother[mask] = sister[mask]
        sister[mask] = temp

        return brother, sister

    def __crossover_mask(self, shape: ShapeLike) -> ArrayLike:
        return np.random.uniform(low=0, high=1, size=shape) > 0.5

    def __select(self, x: ArrayLike, y_true: ArrayLike, loss: LossFunction) -> Params:
        self.__population.sort(
            key=lambda it: self.__calculate_cost(it, x, y_true, loss)
        )
        self.__population = self.__population[: self.__population_size]

    def __calculate_cost(
        self, params: Params, x: ArrayLike, y_true: ArrayLike, loss: LossFunction
    ) -> float:
        return loss.apply(y_true, params.apply(x))
