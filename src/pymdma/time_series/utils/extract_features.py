from pathlib import Path
from typing import List

import numpy as np
import tsfel
from torch.utils.data import DataLoader

from ..data.simple_dataset import _read_sig_file


class FeatureExtractor:
    default: str = "tsfel"

    def __init__(
        self,
        name: str,
        device: str = "cpu",
        **kwargs,
    ):
        """Initializes the feature extractor with the given parameters.

        Parameters
        ----------
        name: str
            identifier of the extractor to be used.
        device: str
            model device. Defaults to "cpu".
        **kwargs: Additional keyword arguments.

        Raises
        ------
        ValueError
            if invalid variable "name" is provided for the extractor.
        """
        self.name = name if name != "default" else "tsfel"
        self.device = device

        if self.name == "tsfel":
            self.extractor = TSFEL()
        else:
            raise ValueError(f"Invalid extractor name: {self.name}")

        if self.name != "tsfel":
            self.extractor._model.to(device)

    def extract_features_from_files(self, files: List[Path], fs: int, dims: List, batch_size: int = 4):
        """Extract features from a list of image files.

        Args:
        files: (List[Path])
            list of paths to signal files
        batch_size: int
            batch size for feature extraction. Defaults to 4.

        Returns
        -------
        np.ndarray
            array of features
        """
        if batch_size > len(files):
            batch_size = len(files)

        assert len(files) > 0, "No files to extract features from."

        n_batches = len(files) // batch_size
        batch_sizes = [batch_size for _ in range(n_batches)]
        if len(files) % batch_size != 0:
            batch_sizes.append(len(files) % batch_size)

        act_array = []
        start, end = 0, 0
        for bsize in batch_sizes:
            end = start + bsize
            signals = [_read_sig_file(f) for f in files[start:end]]
            batch = self.extractor.extract(signals, fs, dims)
            act_array.append(batch)
            start += bsize
        return np.concatenate(act_array, axis=0)

    def extract_features_dataloader(self, dataloader: DataLoader, fs: int, dims: List):
        """Use selected approach to extract features from all signals in the
        dataloader.

        Parameters
        ----------
        dataloader: DataLoader
            signals dataloader
        fs: int
            sampling frequency
        dims: List
             list containing the names of each signal dimension or channel.
             For example, in the context of ECG data, this would be a list of the names of each ECG lead.

        Returns
        -------
        Tuple[np.ndarray, List, List]
            Array of features, list of signal labels and list of signal IDs.
        """
        act_array = []
        labels_array = []
        ids_array = []

        for batch, labels, signal_ids in dataloader:
            batch_feat = self.extractor.extract(batch, fs, dims)
            act_array.append(batch_feat)
            labels_array.extend(labels)
            ids_array.extend(signal_ids)

        features = np.concatenate(act_array, axis=0)

        return features, labels_array, ids_array


class TSFEL:
    def __init__(self, domains=None):
        # Generate default domain value
        if domains is None:
            domains = ["temporal", "statistical", "spectral"]
        self.domains = domains

    def extract(self, batch_windows, fs, dims):
        """Extracts features from a batch of samples.

        Parameters
        ----------
        batch_windows: List
            Batch of signals with len(dims) chans.
        fs: int
            Sampling frequency
        dims: List(str)
           list with the names of each signal dimension/channel ex: name of each ECG Lead

        Returns
        -------
        features: DataFrame
            DataFrame with the features from each batch.
        """
        cfg_file = {}
        for domain in self.domains:
            cfg_file.update(tsfel.get_features_by_domain(domain))

        features = tsfel.time_series_features_extractor(
            cfg_file,
            batch_windows,
            fs=fs,
            window_size=None,
            header_names=dims,
        )

        return features
