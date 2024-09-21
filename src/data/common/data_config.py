from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    raw_data_path: Path
    train_data_path: Path
    test_data_path: Path
    test_train_ratio: float
