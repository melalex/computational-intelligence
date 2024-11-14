import argparse
import logging
import logging.config
from pathlib import Path

from lib.model.neural_net import read_model, write_model
from src.definitions import (
    LOGGING_CONFIG_PATH,
    MODELS_FOLDER,
    TEST_DATA_FOLDER,
    TRAIN_DATA_FOLDER,
)
from src.model.genetic_ipc_prediction.genetic_ipc_prediction_def import (
    test_ipc_prediction_net,
)


MODEL_TO_ACTION_MAP = {"genetic_ipc_prediction": test_ipc_prediction_net}


def train_model(model_name: str, logger: logging.Logger) -> None:
    model_path = __create_model_path(model_name)
    action = MODEL_TO_ACTION_MAP.get(model_name)

    if not action:
        raise RuntimeError("Model with name [ %s ] is not found" % model_name)
    else:
        model = read_model(model_path)

        train_accuracy, test_accuracy = action(
            model, TRAIN_DATA_FOLDER, TEST_DATA_FOLDER
        )

        logger.info("Train accuracy: %s", train_accuracy)
        logger.info("Test accuracy: %s", test_accuracy)


def __create_model_path(name: str) -> Path:
    file_name = name + ".pickle"
    return MODELS_FOLDER / file_name


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)

    parser = argparse.ArgumentParser()

    parser.add_argument("model")

    args = parser.parse_args()
    model = args.model

    train_model(model, logging.getLogger(__name__))
