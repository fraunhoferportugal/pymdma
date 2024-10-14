import json
from pathlib import Path

from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, path: Path, data: list = None):
        assert path.exists(), f"{path} does not exist."
        self.data = [json.loads(line) for line in path.open("r").readlines()] if data is None else data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
