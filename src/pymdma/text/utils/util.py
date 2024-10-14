from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    """Set random seed for numpy.

    https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f
    """
    rng = np.random.default_rng(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    return rng


def flatten_list(lst):
    flattened = []
    for sublist in lst:
        flattened.extend(sublist)
    return flattened


def create_hf_dataset(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame):
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(validation_df),
            "test": Dataset.from_pandas(test_df),
        },
    )


# EOF
