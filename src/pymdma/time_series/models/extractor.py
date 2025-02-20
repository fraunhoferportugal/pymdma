from pathlib import Path
from typing import Callable, List, Union

import numpy as np
import torch
import tsfel
from torch import nn
from torch.utils.data import DataLoader

from pymdma.common.definitions import EmbedderInterface

from ..data.simple_dataset import _read_sig_file


class BaseTSExtractor(nn.Module, EmbedderInterface):
    default: str = "tsfel"
    extractor: Union[nn.Module, Callable] = None

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()
        pass

    @torch.no_grad()
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
            batch = self(signals, fs, dims)
            act_array.append(batch)
            start += bsize
        return np.concatenate(act_array, axis=0)

    @torch.no_grad()
    def _extract_features_dataloader(self, dataloader: DataLoader, fs: int, dims: List):
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
            batch_feat = self(batch, fs, dims)
            act_array.append(batch_feat)
            labels_array.extend(labels)
            ids_array.extend(signal_ids)

        features = np.concatenate(act_array, axis=0)
        return features, labels_array, ids_array
