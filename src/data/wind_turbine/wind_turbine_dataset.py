import logging
from pathlib import Path

import pandas as pd

from src.data.common.data_config import DataConfig
from src.data.common.dataset import download_dataset, split_with_ration, unzip_file


WIND_TURBINE_DATASET_FILE = "wind_turbine_2013_2016"


def download_wind_turbine_dataset(
    external_data_path: Path, logger: logging.Logger = logging.getLogger(__name__)
) -> Path:
    return download_dataset(
        "sudhanvahg",
        "wind-turbine-power-generation-forecasting",
        external_data_path,
        logger,
    )


def process_wind_turbine_dataset_and_save(
    archive: Path,
    config: DataConfig,
    logger: logging.Logger = logging.getLogger(__name__),
) -> tuple[Path, Path]:
    train, test = process_wind_turbine_dataset(archive, config.test_train_ratio, logger)

    filename = WIND_TURBINE_DATASET_FILE + ".csv"
    train_target_path = config.train_data_path / filename
    test_target_path = config.test_data_path / filename

    train.to_csv(train_target_path, index=False)
    test.to_csv(test_target_path, index=False)

    logger.info(
        "Saved [ %s ] rows from train dataset to [ %s ]",
        len(train.index),
        train_target_path,
    )
    logger.info(
        "Saved [ %s ] rows from test dataset to [ %s ]",
        len(test.index),
        test_target_path,
    )


def process_wind_turbine_dataset(
    archive: Path,
    test_train_ratio,
    take_rows: int = -1,
    logger: logging.Logger = logging.getLogger(__name__),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unzipped = unzip_file(archive, logger)
    dataset = pd.read_csv(unzipped / "Train.csv", index_col=0).drop(columns=["Time"])
    reduced_dataset = dataset if take_rows == -1 else dataset.iloc[:take_rows, :]

    return split_with_ration(reduced_dataset, test_train_ratio)
