import numpy as np

from lib.ml.util.types import ArrayLike


def rolling_window(a: ArrayLike, window: int) -> ArrayLike:
    result = []
    for i in range(a.shape[0] - window):
        result.append(a[i : i + window])
    return np.array(result)
