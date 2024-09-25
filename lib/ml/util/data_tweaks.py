import numpy as np
import pandas as pd

from lib.ml.util.types import ArrayLike


def rolling_window(a: ArrayLike, window: int) -> ArrayLike:
    result = []
    for i in range(a.shape[0] - window):
        result.append(a[i : i + window])
    return np.array(result)


def split_with_ration(
    source: pd.DataFrame, ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_row_count = int(len(source.index) * ratio)

    return source.iloc[:train_row_count, :], source.iloc[train_row_count:, :]
