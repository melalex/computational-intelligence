import logging
import logging.config
from pathlib import Path

import pandas as pd
from src.data.common.data_config import DataConfig
from src.data.common.dataset import split_with_ration

ECONOMIC_DATASET_FILENAME = "economic_1995_1997.csv"


def process_economic_raw_dataset_and_save(config: DataConfig, logger: logging.Logger):
    train_target_path = config.train_data_path / ECONOMIC_DATASET_FILENAME
    test_target_path = config.test_data_path / ECONOMIC_DATASET_FILENAME

    if train_target_path.exists() and test_target_path.exists():
        logger.info(
            "File [ %s ] and [ %s ] exist. Skipping dataset processing...",
            train_target_path,
            test_target_path,
        )
    else:
        dataset = process_economic_raw_dataset(
            pd.read_csv(config.raw_data_path / ECONOMIC_DATASET_FILENAME, sep="\t")
        )

        train_dataset, test_dataset = split_with_ration(
            dataset, config.test_train_ratio
        )

        train_dataset.to_csv(train_target_path, index=False)
        test_dataset.to_csv(test_target_path, index=False)

        logger.info(
            "Saved [ %s ] rows from train dataset to [ %s ]",
            len(train_dataset.index),
            train_target_path,
        )
        logger.info(
            "Saved [ %s ] rows from test dataset to [ %s ]",
            len(test_dataset.index),
            test_target_path,
        )


def process_economic_raw_dataset(input: pd.DataFrame) -> pd.DataFrame:
    dataset = input.rename(
        columns={
            "M0": "M0(-7)",
            "M2": "M2(-7)",
            "IOC": "IOC(0)",
            "IPC": "IPC(0)",
            "KVVE": "KVVE(-7)",
        }
    )

    dataset["IPC(+1)"] = dataset["IPC(0)"]

    dataset["M0(-7)"] = dataset["M0(-7)"].shift(periods=7)
    dataset["M2(-7)"] = dataset["M2(-7)"].shift(periods=7)
    dataset["KVVE(-7)"] = dataset["KVVE(-7)"].shift(periods=7)
    dataset["IPC(+1)"] = dataset["IPC(+1)"].shift(periods=-1)

    return dataset.dropna()
