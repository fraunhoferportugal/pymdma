import os
import sys
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import DataLoader

from pymdma.common.definitions import InputLayer
from pymdma.constants import ReferenceType, ValidationDomain

from .data.simple_dataset import SimpleDataset
from .utils.extract_features import FeatureExtractor

# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to the Python path if it's not already included
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

SUPPORTED_FILES = {".dat", ".mat", ".csv"}  # TODO might want to add others or change


def _custom_collate_fn(batch):
    # Custom collate function to handle varying sizes
    signal, label, signal_id = zip(*batch)
    return list(signal), list(label), list(signal_id)


def _get_data_files_path(data_src: Union[List[str], Path]) -> List[Path]:
    """Get a list of data files from the given source directory or list of file
    paths.

    Parameters
    ----------
    data_src: (Union[List[str], Path])
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
    else:
        data_files_path = []
        for item in data_src.iterdir():
            if item.is_file():
                if item.suffix in SUPPORTED_FILES:
                    data_files_path.append(item)
                else:
                    raise AssertionError(f"Unsupported file extension: {item.suffix} (file: {item})")
            elif item.is_dir():
                item = Path(item)
                # Recursively search for data files in subdirectories
                for sig_file in item.iterdir():
                    if sig_file.is_file() and sig_file.suffix in SUPPORTED_FILES:
                        data_files_path.append(sig_file)
                    elif sig_file.is_file() and sig_file.suffix == ".hea":
                        continue
                    else:
                        raise AssertionError(f"Unsupported file extension: {sig_file.suffix} (file: {sig_file})")
            else:
                raise AssertionError(f"Invalid item encountered: {item}")

    return data_files_path


class TimeSeriesInputLayer(InputLayer):
    """Abstraction layer for handling different types of input data with the
    given requirements from the data auditing module."""

    def __init__(
        self,
        validation_domain: ValidationDomain,
        reference_type: ReferenceType,
        target_data: Union[Path, List[Path]],
        reference_data: Optional[Union[Path, List[Path]]] = None,
        batch_size: int = 64,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """Initializes the input layer with the given parameters.

        Parameters
        ----------
        validation_domain : ValidationDomain
            Valition type (input or synthetic).
        reference_type: ReferenceType
            Reference type (dataset or single signal).
        data_src1 : Path
            Data root or list of paths to the reference dataset in synthetic evaluation or the dataset in input validation.
        data_src2: (Path, optional)
            Data root or list of paths to the synthetic dataset in synthetic ealuation or another dataset used for full reference input validation. Defaults to None.
        batch_size: int
            Batch size to be used in the feature extraction module or the number of signals returned in the input validation. Defaults to 64.
        device : str
            Device to be used in the feature extraction module. Defaults to "cpu".
        feature_extractor_name : str
            Feature extractor name. Defaults to tsfel.
        """
        super().__init__()
        self.val_type = validation_domain
        self.reference_type = reference_type
        self.batch_size = batch_size
        self.device = device

        # ids for the signals in instance analysis
        # used later for instance level metrics ids
        self.instance_ids = []

        # prepare feature extractor for synthetic evaluation
        self.transform = np.asarray  # default transform to np.array

        collate_fn = _custom_collate_fn

        target_files = _get_data_files_path(target_data)

        # prepare reference dataloader (original/reference signals)
        # will also be used for input validation
        if reference_type != ReferenceType.NONE:
            reference_files = _get_data_files_path(reference_data)
            reference_dataset = SimpleDataset(file_paths=reference_files, transforms=self.transform)
            self.reference_loader = DataLoader(
                reference_dataset,
                batch_size=self.batch_size if self.batch_size > 0 else len(reference_files),
                shuffle=False,
                num_workers=4,
                collate_fn=collate_fn,
            )

        # prepare target dataloader (synthetic/similar signals)
        # Will be used for synthetic evaluation or full reference input validation
        target_dataset = SimpleDataset(file_paths=target_files, transforms=self.transform)
        self.target_loader = DataLoader(
            target_dataset,
            batch_size=self.batch_size if self.batch_size > 0 else len(target_dataset),
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )

        # TODO review this for modality
        # Client input validation
        if self.val_type == ValidationDomain.SYNTH:
            assert (
                len(target_files) > 4
            ), f"Synthetic datasets must have at least 5 signals for synthetic evaluation. Found {len(target_files)}."
        elif self.val_type == ValidationDomain.INPUT and reference_type != ReferenceType.NONE:
            assert len(reference_dataset) == len(
                target_dataset,
            ), "Reference and target datasets must have the same size for full reference input validation."

        self.instance_ids = [path.stem for path in target_dataset.sig_files]
        # if feature_extractor_name == "tsfel":
        #     self.target_loader.fs = target_dataset.fs
        #     self.target_loader.dims = target_dataset.dims

    def __len__(self):
        return len(self.target_loader.dataset)

    def get_embeddings(
        self,
        model_name: str,
        model_instances: Optional[Dict[str, callable]] = None,
        offload_model: bool = False,
    ):
        extractor = None
        if model_instances is not None:
            if model_name in model_instances:
                extractor = model_instances[model_name]
            elif model_name == "default" and FeatureExtractor.default in model_instances:
                extractor = model_instances[FeatureExtractor.default]
        extractor = FeatureExtractor(model_name, device=self.device) if extractor is None else extractor

        reference_features, _labels, _ = extractor.extract_features_dataloader(
            self.reference_loader,
            self.reference_loader.dataset.fs,
            self.reference_loader.dataset.dims,
        )
        synth_features, _labels, self.instance_ids = extractor.extract_features_dataloader(
            self.target_loader,
            self.target_loader.dataset.fs,
            self.target_loader.dataset.dims,
        )

        if offload_model and model_name != "tsfel":
            extractor._model = extractor._model.to("cpu")
            del extractor
        return reference_features, synth_features

    @property
    def batched_samples(self) -> Generator[Tuple[np.ndarray], None, None]:
        # only reference signals for no reference metrics
        if self.reference_type == ReferenceType.NONE:
            for no_ref_signals, _labels, _sig_ids in self.target_loader:
                # save signal ids for later
                # self.instance_ids.extend(sig_ids)
                yield (no_ref_signals,)
        else:  # full reference
            # iterate through both dataloaders and yield batches of signals
            ref_iter, sim_iter = iter(self.reference_loader), iter(self.target_loader)
            for _ in range(len(self.reference_loader)):
                ref_sigs, _, _ = next(ref_iter)
                sim_sigs, _, _sig_ids = next(sim_iter)

                # save signal ids for later
                # self.instance_ids.extend(list(sig_ids))

                yield ref_sigs, sim_sigs

    def get_full_samples(self):
        # only reference signals for no reference metrics
        if self.reference_type == ReferenceType.NONE:
            full_no_ref_signals = []
            for no_ref_signals, _labels, _sig_ids in self.target_loader:
                full_no_ref_signals.extend(no_ref_signals)
            return np.array(full_no_ref_signals)
        else:  # full reference
            # iterate through both dataloaders and return all signals
            full_ref_sigs = []
            full_sim_sigs = []
            ref_iter = iter(self.reference_loader)
            sim_iter = iter(self.target_loader)
            for _ in range(len(self.reference_loader)):
                ref_sigs, _, _ = next(ref_iter)
                sim_sigs, _, _sig_ids = next(sim_iter)
                full_ref_sigs.extend(ref_sigs)
                full_sim_sigs.extend(sim_sigs)

            return np.array(full_ref_sigs), np.array(full_sim_sigs)
