import os
from pathlib import Path
from typing import Callable, List, Optional

import wfdb
import mne
from torch.utils.data import Dataset

# TODO support other formats?
SUPPORTED_FILES = {".dat", ".mat", ".csv", ".vhdr", ".vmrk", ".eeg"}  # TODO might want to add others or change



def _extract_diagnosis(file_path):
    """Extracts diagnosis information from a header file. Only works for this
    specific .hea file structure.

    Parameters
    ----------
    file_path: str
        The path to the header file.

    Returns
    --------
    diagnosis_list: list
        A list of diagnosis codes extracted from the header file.
    """
    with open(file_path) as f:
        lines = f.readlines()
    diagnosis = lines[15].strip().split(": ")[1]
    diagnosis_list = [diagnosis.strip() for diagnosis in diagnosis.split(",")]

    return diagnosis_list


def _extract_fs_dims(file_path):
    """Extracts the sampling frequency and the dimension names of the signal
    from a header file. Only works for this specific .hea file structure.

    Parameters
    ----------
    file_path: str
        The path to the header file.

    Returns
    -------
    fs : int
        Sampling frequency.
    dims: List(str)
        Names of the signal dimensions.
    """
    with open(file_path) as f:
        lines = f.readlines()
    dims = [lines[i].strip().split(" ")[-1] for i in range(1, 13)]
    fs = lines[0].strip().split(" ")[2]

    return int(fs), dims


def _read_sig_file(file_path):
    """Read a signal file from the supported file extensions.

    Parameters:
    -----------
    file_path: Union[str, Path])
        Path to the file.

    Returns
    --------
    dict
        Dictionary containing the data from the .mat file.

    Raises
    ------
    ValueError
        If a file extension different from .mat is found.
    """

    # Check if the file has a .mat extension
    if file_path.suffix in [".mat", ".dat"]:
        directory_path, file_name = os.path.split(file_path)
        file_path = os.path.join(directory_path, file_name.split(".")[0])
        data = wfdb.rdsamp(file_path)[0]
        return data, None, None
    elif file_path.suffix == ".vhdr": # brainvision header
        data = mne.io.read_raw_brainvision(file_path, preload=True)
        dims = data.ch_names
        return data.get_data().T, dims, data.info["sfreq"] # FIXME
    else:
        # Raise a ValueError for files with unsupported extensions
        raise AssertionError(f"Unsupported file extension: {Path(file_path).suffix} (file: {file_path})")
        # ? TBD add more supported file types


class SimpleDataset(Dataset):
    def __init__(
        self,
        file_paths: List[Path],
        transforms: Optional[Callable] = None,
        target_transforms: Optional[Callable] = None,
    ) -> None:
        self.sig_files = file_paths
        self.transform = transforms
        self.target_transform = target_transforms
        
        # parse brainvision format and filter  for only header files
        if any(file.suffix in {".vhdr", ".vmrk", ".eeg"} for file in self.sig_files):
            # filter to only include header files
            self.sig_files = [file for file in self.sig_files if file.suffix == ".vhdr"]
            data = mne.io.read_raw_brainvision(self.sig_files[0], preload=True)
            self.fs = data.info["sfreq"]
            self.dims = data.ch_names
        elif any(file.suffix == ".mat" for file in self.sig_files):
            # Define fs and dims of the dataset
            ex_hea_path = os.path.splitext(self.sig_files[0])[0] + ".hea"
            self.fs, self.dims = _extract_fs_dims(ex_hea_path)
        else:
            raise ValueError(f"Unsupported file extension: {Path(self.sig_files[0]).suffix} (file: {self.sig_files[0]})")


    def __len__(self):
        return len(self.sig_files)

    def __getitem__(self, idx):
        sig_path = Path(self.sig_files[idx])
        signal, dims, fs = _read_sig_file(sig_path) # FIXEM foe .hea
        
        # print("SHAPE ", signal.shape)

        # hea_path = os.path.splitext(sig_path)[0] + ".hea"
        # label = [None]  # TODO change latter if needed to: label = _extract_diagnosis(hea_path)

        signal_id = sig_path.stem

        if self.transform:
            signal = self.transform(signal)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return signal, (fs, dims), signal_id
