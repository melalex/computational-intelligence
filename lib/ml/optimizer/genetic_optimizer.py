from copy import deepcopy
from math import exp

import numpy as np
from lib.ml.layer.actual_layer import (
    ActivationLayer,
    CompositeLayer,
    Layer,
    BiasedWeightLayer,
    ReshapeLayer,
    WeightLayer,
)
from lib.ml.util.loss_function import LossFunction
from lib.ml.optimizer.nn_optimizer import (
    NeuralNetOptimizer,
    OptimalResult,
    ParamsSupplier,
)
from lib.ml.util.types import ArrayLike, ShapeLike


class GeneticAlgorithmNeuralNetOptimizer(NeuralNetOptimizer):
    __population_size: int
    __mutation_rate: float
    __mutation_decay: float
    __population: list[Layer]
    __fitnesses: list[float]

    def __init__(
        self,
        population_size: int,
        mutation_rate: float,
        mutation_decay: float,
    ):
        self.__population_size = population_size
        self.__mutation_rate = mutation_rate
        self.__mutation_decay = mutation_decay

    def prepare(self, params_supplier: ParamsSupplier) -> None:
        self.__population = [params_supplier() for _ in range(self.__population_size)]
        self.__fitnesses = []

    def optimize(
        self,
        epoch: int,
        x: ArrayLike,
        y_true: ArrayLike,
        loss: LossFunction,
    ) -> OptimalResult:
        self.__calculate_fitnesses(x, y_true, loss)
        self.__breed(epoch, x, y_true, loss)
        self.__select()

        optimal = self.__population[0]
        accuracy = self.__calculate_cost(optimal, x, y_true, loss)

        return OptimalResult(optimal, accuracy)

    def __calculate_fitnesses(
        self, x: ArrayLike, y_true: ArrayLike, loss: LossFunction
    ) -> None:
        self.__fitnesses = [
            self.__fitness_idx(p, x, y_true, loss) for p in self.__population
        ]

    def __breed(
        self, epoch: int, x: ArrayLike, y_true: ArrayLike, loss: LossFunction
    ) -> None:
        fitnesses = np.array(self.__fitnesses)
        fitnesses_sum = np.sum(fitnesses)
        breed_prop = fitnesses / fitnesses_sum
        parents = np.random.choice(
            a=self.__population,
            size=self.__population_size,
            replace=False,
            p=breed_prop,
        )

        for i in range(0, self.__population_size, 2):
            if i + 1 < self.__population_size:
                child1, child2 = self.__crossover(parents[i], parents[i + 1])
                mutant1 = self.__mutate(epoch, child1)
                mutant2 = self.__mutate(epoch, child2)

                self.__population.append(mutant1)
                self.__population.append(mutant2)
                self.__fitnesses.append(self.__fitness_idx(mutant1, x, y_true, loss))
                self.__fitnesses.append(self.__fitness_idx(mutant2, x, y_true, loss))

    def __fitness_idx(
        self, params: Layer, x: ArrayLike, y_true: ArrayLike, loss: LossFunction
    ) -> float:
        return 1 / (self.__calculate_cost(params, x, y_true, loss) + 1e-9)

    def __mutate(self, epoch: int, params: Layer) -> Layer:
        match params:
            case CompositeLayer(values):
                return CompositeLayer([self.__mutate(epoch, it) for it in values])
            case BiasedWeightLayer(weight, bias):
                exp_term = self.__mutation_rate * exp(-self.__mutation_decay * epoch)
                weight_mutation = self.__mutation_of_shape(exp_term, weight.shape)
                bias_mutation = self.__mutation_of_shape(exp_term, bias.shape)

                return BiasedWeightLayer(weight + weight_mutation, bias + bias_mutation)
            case WeightLayer(weight):
                exp_term = self.__mutation_rate * exp(-self.__mutation_decay * epoch)
                weight_mutation = self.__mutation_of_shape(exp_term, weight.shape)

                return WeightLayer(weight + weight_mutation)
            case ReshapeLayer(_) | ActivationLayer(_) as it:
                return it

    def __mutation_of_shape(self, exp_term: float, shape: ShapeLike) -> ArrayLike:
        return (
            np.random.uniform(
                low=-self.__mutation_rate, high=self.__mutation_rate, size=shape
            )
            * exp_term
        )

    def __crossover(self, father: Layer, mother: Layer) -> tuple[Layer, Layer]:
        match father, mother:
            case CompositeLayer(father_values), CompositeLayer(mother_values):
                children = [
                    self.__crossover(f, m) for f, m in zip(father_values, mother_values)
                ]

                brothers = CompositeLayer([it[0] for it in children])
                sisters = CompositeLayer([it[1] for it in children])

                return brothers, sisters
            case BiasedWeightLayer(f_w, f_b), BiasedWeightLayer(m_w, m_b):
                brother_weight, sister_weight = self.__crossover_arr(f_w, m_w)
                brother_bias, sister_bias = self.__crossover_arr(f_b, m_b)

                brother = BiasedWeightLayer(brother_weight, brother_bias)
                sister = BiasedWeightLayer(sister_weight, sister_bias)

                return brother, sister
            case WeightLayer(father_weight), WeightLayer(mother_weight):
                brother_weight, sister_weight = self.__crossover_arr(
                    father_weight, mother_weight
                )

                brother = WeightLayer(brother_weight)
                sister = WeightLayer(sister_weight)

                return brother, sister
            case it:
                return it

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

    def __select(self) -> Layer:
        np_fitness = np.array(self.__fitnesses)
        np_population = np.array(self.__population)

        order = np_fitness.argsort()[::-1]

        self.__population = np_population[order][: self.__population_size].tolist()
        self.__fitnesses = np_fitness[order][: self.__population_size].tolist()

    def __calculate_cost(
        self, params: Layer, x: ArrayLike, y_true: ArrayLike, loss: LossFunction
    ) -> float:
        return loss.apply(y_true, params.apply(x))
