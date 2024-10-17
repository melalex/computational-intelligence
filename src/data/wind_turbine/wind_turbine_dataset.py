import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

from lib.ml.util.data_tweaks import rolling_window
from src.data.common.dataset import download_dataset, unzip_file


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


def process_wind_turbine_dataset(
    archive: Path,
    scaler: StandardScaler,
    logger: logging.Logger = logging.getLogger(__name__),
) -> pd.DataFrame:
    unzipped = unzip_file(archive, logger)
    dataset = pd.read_csv(unzipped / "Train.csv", index_col=0).drop(columns=["Time"])

    x = dataset.drop(columns=["Power", "Location"])

    x_scaled = scaler.fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled, columns=x.columns)
    x_scaled["Power"] = np.log1p(dataset["Power"].values)
    # x_scaled["Location"] = (dataset["Location"] * 1).values

    # return pd.get_dummies(x_scaled, columns=["Location"], dtype=float)
    return x_scaled


def window_and_split_turbine_dataset(
    dataset: pd.DataFrame,
    test_train_ratio: float,
    train_valid_ratio: float,
    window_size: int,
) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    x_1, y_1 = __window_for_loc(dataset[dataset["Location_1"] == 1], window_size)
    x_2, y_2 = __window_for_loc(dataset[dataset["Location_2"] == 1], window_size)
    x_3, y_3 = __window_for_loc(dataset[dataset["Location_3"] == 1], window_size)
    x_4, y_4 = __window_for_loc(dataset[dataset["Location_4"] == 1], window_size)

    x = np.concat([x_1, x_2, x_3, x_4])
    y = np.concat([y_1, y_2, y_3, y_4])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=test_train_ratio
    )

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, train_size=train_valid_ratio
    )

    return (
        x_train.T,
        y_train.reshape(1, -1),
        x_valid.T,
        y_valid.reshape(1, -1),
        x_test.T,
        y_test.reshape(1, -1),
    )


def split_turbine_dataset(
    dataset: pd.DataFrame,
    test_train_ratio: float,
    train_valid_ratio: float,
) -> tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    y = dataset["Power"].to_numpy()
    x = dataset.drop(columns=["Power"]).to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=test_train_ratio
    )

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, train_size=train_valid_ratio
    )

    return (
        x_train.T,
        y_train.reshape(1, -1),
        x_valid.T,
        y_valid.reshape(1, -1),
        x_test.T,
        y_test.reshape(1, -1),
    )


def __window_for_loc(dataset: pd.DataFrame, window_size: int) -> pd.DataFrame:
    y = dataset["Power"].to_numpy()
    x = dataset.to_numpy()

    feat_count = x.shape[1]
    windowed_feat_count = feat_count * window_size

    x_windowed = rolling_window(x, window_size).reshape(-1, windowed_feat_count)
    y_windowed = y[window_size:]

    return x_windowed, y_windowed
