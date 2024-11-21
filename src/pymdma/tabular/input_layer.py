import os
import sys
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from torch.utils.data import DataLoader

from pymdma.common.definitions import InputLayer
from pymdma.constants import ReferenceType, ValidationDomain

from .data.load import TabularDataset

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to the Python path if it's not already included
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

SUPPORTED_FILES = {".json", ".xml", ".csv", ".xlsx"}  # TODO might want to add others or change


def _custom_collate_fn(batch):
    """Custom collate function to split batches into several elements.

    Parameters
    ----------
    batch : list of tuples
        A list where each element is a tuple containing encoded data, scaled data,
        column names, column map, quasi-identifier names, and sensitive names.

    Returns
    -------
    np.ndarray
        An array containing the scaled data extracted from the batch.
    """

    # Custom collate function to handle varying sizes
    data_enc, data_s, column_names, col_map, qi_names, sens_names = zip(*batch)
    # return {
    #     "data": np.array(data),
    #     # "emb": np.array(emb),
    #     # "column_names": column_names[0],
    #     # "qi_names": qi_names[0],
    #     # "sens_names": sens_names[0],
    #     # "col_map": col_map[0],
    # }
    return np.array(data_s)


def _get_data_files_path(data_src: Union[List[str], Path]) -> List[Path]:
    """Get a list of data files from the given source directory or list of file
    paths.

    Parameters
    ----------
    data_src : Union[List[str], Path]
        A list of file paths or a directory path.

    Returns
    -------
    List[Path]
        A list of data file paths.

    Raises
    ------
    ValueError
        If an invalid item is encountered or if an unsupported file extension is found.
    """

    if isinstance(data_src, list):
        data_files_path = data_src
    elif not data_src.is_dir():
        data_files_path = []
    else:
        data_files_path = []
        for item in data_src.iterdir():
            assert item.is_file() or item.is_dir(), f"Invalid item encountered: {item}"
            if item.is_file():
                assert item.suffix in SUPPORTED_FILES, f"Unsupported file extension: {item.suffix} (file: {item})"
                data_files_path.append(item)
            elif item.is_dir():
                # Recursively search for data files in subdirectories
                for data_file in item.iterdir():
                    if data_file.is_file() and data_file.suffix in SUPPORTED_FILES:
                        data_files_path.append(data_file)
                    else:
                        logger.warning(f"Unsupported file extension: {data_file.suffix} (file: {data_file})")
                        continue

    # tabular data usually only has one path
    data_path = data_files_path[0] if data_files_path else data_files_path

    return data_path


class TabularInputLayer(InputLayer):
    """Abstraction layer for handling different types of input data with the
    requirements from the data auditing module.

    Parameters
    ----------
    validation_domain : ValidationDomain
        The type of validation (input or synthetic).
    reference_type : ReferenceType
        The type of reference data (dataset or single tabular set).
    target_data : Path
        The path to the target dataset.
    reference_data : Optional[Path], optional
        The path to the reference dataset (used in synthetic evaluation).
    device : str, optional
        The device to be used in the feature extraction module (default is "cpu").
    scaler : str, optional
        The type of data normalization applied (default is None).
    embed : str, optional
        The type of embeddings for encoding features (default is None).
    **kwargs : optional
        Additional arguments passed to the InputLayer class.
    """

    def __init__(
        self,
        validation_domain: ValidationDomain,
        reference_type: ReferenceType,
        target_data: Path,
        reference_data: Optional[Path] = None,
        device: str = "cpu",
        scaler: str = None,
        embed: str = None,
        **kwargs,
    ) -> None:
        """Initializes the input layer with the given parameters.

        Parameters
        ----------
        validation_domain : ValidationDomain
            The validation type (input or synthetic).
        reference_type : ReferenceType
            The reference type (dataset or single tabular set).
        target_data : Path
            The path to the target dataset.
        reference_data : Optional[Path], optional
            The path to the reference dataset (default is None).
        device : str
            The device to be used in the feature extraction module (default is "cpu").
        scaler : str, optional
            The type of data normalization applied (default is None).
        embed : str, optional
            The type of embeddings for encoding features (default is None).
        **kwargs : optional
            Additional arguments passed to the InputLayer class.
        """

        super().__init__()
        self.val_type = validation_domain
        self.reference_type = reference_type
        self.device = device

        # scaler type
        self.scaler = scaler

        # embed type
        self.embed = embed

        # prepare reference dataloader (original/reference records)
        # will also be used for input validation
        if validation_domain == ValidationDomain.SYNTH:
            reference_file = _get_data_files_path(reference_data)
            reference_dataset = TabularDataset(
                file_path=reference_file,
                scaler=self.scaler,
                embed=self.embed,
                **kwargs,
            )
            self.reference_loader = DataLoader(
                dataset=reference_dataset,
                batch_size=len(reference_dataset),
                collate_fn=_custom_collate_fn,
            )

            # assign fitted scaler and embedders
            self.embed = reference_dataset.embed
            self.scaler = reference_dataset.scaler

        # prepare target dataloader (synthetic/similar records)
        # Will be used for synthetic evaluation or full reference input validation
        target_file = _get_data_files_path(target_data)
        target_dataset = TabularDataset(
            file_path=target_file,
            scaler=self.scaler,  # use scaler fitted by reference
            embed=self.embed,  # use embedder fitted by reference
            **kwargs,
        )
        self.target_loader = DataLoader(
            dataset=target_dataset,
            batch_size=len(target_dataset),
            collate_fn=_custom_collate_fn,
        )

        # TODO review for data modality
        # Client input validation
        if validation_domain == ValidationDomain.INPUT and reference_type != ReferenceType.NONE:
            assert len(reference_dataset) == len(
                target_dataset,
            ), "Reference and target datasets must have the same size for input validation."

    def __len__(self):
        """Returns the number of samples in the target dataset.

        Returns
        -------
        int
            The number of samples in the target dataset.
        """
        return len(self.target_loader.dataset)

    def get_embeddings(
        self,
        extractor_name: Optional[str] = None,
        model_instances: Optional[any] = None,
        **kwargs,
    ):
        """Retrieves the embeddings from the reference and target datasets.

        Parameters
        ----------
        extractor_name : Optional[str], optional
            The name of the extractor to use (default is None).
        model_instances : Optional[any], optional
            Additional model instances to use (default is None).
        **kwargs : optional
            Additional arguments for the extraction process.

        Returns
        -------
        tuple
            A tuple containing the embeddings from the reference and target datasets.
        """
        return self.reference_loader.dataset.data_emb, self.target_loader.dataset.data_emb

    @property
    def data_properties(self) -> Dict[str, any]:
        """Returns the properties of the target dataset.

        Returns
        -------
        Dict[str, any]
            A dictionary containing the properties of the target dataset.
        """
        return self.target_loader.dataset.properties

    @property
    def batched_samples(self) -> Generator[Tuple[np.ndarray], None, None]:
        """Yields batches of samples from the reference and target loaders.

        Yields
        ------
        Tuple[np.ndarray]
            A tuple containing the samples from the reference and target loaders.
        """

        # yielding batch of samples for evaluation
        if self.reference_type == ReferenceType.NONE:
            for sample in self.target_loader:
                yield (sample,)
        else:
            yield from zip(self.reference_loader, self.target_loader)

    # def get_batched_samples(self) -> Generator[Tuple[np.ndarray], None, None]:
    #     # return all features for both reference and synthetic records as a single batch
    #     if self.val_type in [ValidationDomain.SYNTH]:
    #         # get reference and synthetic batches
    #         reference_features = next(iter(self.reference_loader))
    #         synth_features = next(iter(self.target_loader))

    #         # rename col
    #         reference_features["real_data"] = reference_features.pop("data")
    #         reference_features["real_emb"] = reference_features.pop("emb")
    #         synth_features["syn_data"] = synth_features.pop("data")
    #         synth_features["syn_emb"] = synth_features.pop("emb")

    #         # checkpoint
    #         assert (
    #             reference_features["column_names"] == synth_features["column_names"]
    #         ), "Reference and Synthetic datasets have mismatched columns"

    #         yield {**reference_features, **synth_features}

    #     # return all records
    #     if self.val_type in [ValidationDomain.INPUT]:
    #         # only reference records for no reference metrics
    #         yield from self.target_loader
