import argparse
import logging
import logging.config

from src.data.common.data_config import DataConfig
from src.data.economic.process_raw_economic_dataset import (
    process_economic_raw_dataset_and_save,
)
from src.definitions import (
    LOGGING_CONFIG_PATH,
    RAW_DATA_FOLDER,
    TEST_DATA_FOLDER,
    TEST_TRAIN_RATIO,
    TRAIN_DATA_FOLDER,
)


MODEL_TO_ACTION_MAP = {"genetic_ipc_prediction": process_economic_raw_dataset_and_save}


def process_dataset(model_name: str, logger: logging.Logger) -> None:
    action = MODEL_TO_ACTION_MAP.get(model_name)
    config = DataConfig(
        raw_data_path=RAW_DATA_FOLDER,
        train_data_path=TRAIN_DATA_FOLDER,
        test_data_path=TEST_DATA_FOLDER,
        test_train_ratio=TEST_TRAIN_RATIO,
    )

    config.train_data_path.mkdir(parents=True, exist_ok=True)
    config.test_data_path.mkdir(parents=True, exist_ok=True)

    if not action:
        raise RuntimeError("Model with name [ %s ] is not found" % model_name)
    else:
        action(config, logger)


if __name__ == "__main__":
    logging.config.fileConfig(LOGGING_CONFIG_PATH)

    parser = argparse.ArgumentParser()

    parser.add_argument("model")

    args = parser.parse_args()
    model = args.model

    process_dataset(model, logging.getLogger(__name__))
