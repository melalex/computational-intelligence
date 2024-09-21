from pathlib import Path

import numpy as np
import pandas as pd
from lib.ml.layer.layer_def import Dense, Input
from lib.ml.loss.loss_function import MEAN_SQUARED_ERROR
from lib.ml.model.neural_net import TrainedNeuralNet
from lib.ml.model.seq_model import SeqNet
from lib.ml.optimizer.genetic_optimizer import GeneticAlgorithmNeuralNetOptimizer
from lib.ml.util.progress_tracker import LoggingProgressTracker
from src.data.economic.process_raw_economic_dataset import ECONOMIC_DATASET_FILENAME


def create_genetic_ipc_prediction_net(train_data_folder: Path) -> TrainedNeuralNet:
    model = SeqNet(layers=[Input(5), Dense(10), Dense(1)])
    opt = GeneticAlgorithmNeuralNetOptimizer(
        population_size=40, mutation_rate=320, alpha=0.05
    )

    compiled = model.compile(
        optimizer=opt,
        loss=MEAN_SQUARED_ERROR,
        progress_tracker=LoggingProgressTracker(100),
    )

    dataset = pd.read_csv(train_data_folder / ECONOMIC_DATASET_FILENAME)
    x, y = __split_in_x_and_y(dataset)

    return compiled.fit(x, y, 1000)


def test_ipc_prediction_net(
    model: TrainedNeuralNet, train_data_folder: Path, test_data_folder: Path
) -> tuple[float, float]:
    train_dataset = pd.read_csv(train_data_folder / ECONOMIC_DATASET_FILENAME)
    test_dataset = pd.read_csv(test_data_folder / ECONOMIC_DATASET_FILENAME)

    train_accuracy = __test_ipc_prediction_net_with(model, train_dataset)
    test_accuracy = __test_ipc_prediction_net_with(model, test_dataset)

    return train_accuracy, test_accuracy


def __test_ipc_prediction_net_with(
    model: TrainedNeuralNet, dataset: pd.DataFrame
) -> float:
    x, y = __split_in_x_and_y(dataset)

    y_predicted = model.predict(x)

    return MEAN_SQUARED_ERROR.apply(y, y_predicted)


def __split_in_x_and_y(dataset: pd.DataFrame) -> tuple[np.array, np.array]:
    return dataset.iloc[:, :-1].to_numpy().T, dataset.iloc[:, -1].to_numpy()
