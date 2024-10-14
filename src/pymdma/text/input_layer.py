import json
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from pymdma.common.definitions import InputLayer
from pymdma.constants import ReferenceType, ValidationTypes

from .data.text_dataset import TextDataset

SUPPORTED_FILES = {".jsonl"}  # TODO might want to add others or change


def parse_data_files(data_src: Path) -> Tuple[Path]:
    files = [tf for tf in data_src.iterdir() if tf.suffix in SUPPORTED_FILES]

    assert len(files) > 0, f"No supported files found in {data_src}. Supported files are {SUPPORTED_FILES}"

    data = []
    for jsonl in files:
        with open(jsonl) as f:
            data.extend([json.loads(line) for line in f.readlines()])

    assert len(data) > 0, f"Empty data source: {data_src}"
    return data


def _custom_collate_fn(batch):
    # Custom collate function to handle varying sizes
    return batch


class TextInputLayer(InputLayer):
    """Abstraction layer for handling different types of input data with the
    given requirements from the data auditing module."""

    def __init__(
        self,
        validation_type: ValidationTypes,
        reference_type: ReferenceType,
        target_data: Path,
        reference_data: Optional[Path] = None,
        batch_size: int = 10,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """Initializes the input layer with the given parameters.

        Args:
            validation_type (ValidationTypes): validation type (input or synthetic)
            reference_type (ReferenceType): reference type (dataset or single tabular set)
            data_src1 (Path): data root or list of paths to the reference dataset in synthetic evaluation or the dataset in input validation.
            data_src2 (Path, optional): data root or list of paths to the synthetic dataset in synthetic ealuation
                                        or another dataset used for full reference input validation. Defaults to None.
            device (str): device to be used in the feature extraction module. Defaults to "cpu".
            scaler(str): type of data normalization applied. Defaults to None
            embed(str): type of embeddings for encoding features. Defaults to None
        """
        super().__init__()
        self.val_type = validation_type
        self.reference_type = reference_type
        self.device = device
        self.batch_size = batch_size
        self.instance_ids = []

        data = parse_data_files(target_data)
        # prepare target dataloader (original/reference records)
        target_data = TextDataset(
            path=target_data,
            data=data,
        )

        self.target_loader = DataLoader(
            dataset=target_data,
            batch_size=batch_size,
            collate_fn=_custom_collate_fn,
        )

    def __len__(self):
        return len(self.target_loader.dataset)

    def get_embeddings(
        self,
        model_name: str,
        model_instances: Optional[Dict[str, callable]] = None,
        offload_model: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Embeddings not supported for text data")

    @property
    def batched_samples(self) -> Generator[Tuple[np.ndarray], None, None]:
        if self.val_type == ValidationTypes.INPUT:
            for batch in self.target_loader:
                self.instance_ids.extend([sample["id"] for sample in batch])
                yield (batch,)

        return []
