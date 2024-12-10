from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from pymdma.common.definitions import EmbedderInterface

# support for large datasets iwht small batch sizes
torch.multiprocessing.set_sharing_strategy("file_system")


class StandardTransform:
    def __init__(
        self,
        image_size: Tuple[int],
        interpolation: Image.Resampling = Image.Resampling.BILINEAR,
        preprocess_transform: Optional[Callable] = None,
    ) -> None:
        assert isinstance(image_size, tuple), "Image size must be a tuple."
        self.img_size = image_size
        self.interp = interpolation
        self.preprocess_transform = preprocess_transform

        self._to_tensor = transforms.PILToTensor()

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = self.preprocess_transform(image) if self.preprocess_transform is not None else image
        image = image.resize(self.img_size, self.interp)
        # bring image to the range [0, 1] and normalize to [-1, 1]
        image = self._to_tensor(image).float() / 255.0
        image = image * 2.0 - 1.0
        return image


class BaseExtractor(torch.nn.Module, EmbedderInterface):
    extractor: torch.nn.Module

    def __init__(
        self,
        input_size: Tuple[int],
        interpolation: Image.Resampling.BILINEAR,
    ):
        super().__init__()
        self.input_size = input_size
        self.interpolation = interpolation

    @torch.no_grad()
    def extract_features_from_files(
        self,
        files: List[Path],
        batch_size: int = 50,
        device: str = "cpu",
        preprocess_transform: Optional[Callable] = None,
    ) -> np.ndarray:
        """Extract features from a list of image files. Converts images to
        tensors and normalizes them to the [-1, 1] range.

        Parameters
        ----------
        files : List[Path]
            list of paths to image files
        batch_size : int, optional
            batch size for feature extraction, by default 50

        Returns
        -------
        np.ndarray
            array of features with shape (n_samples, n_features)
        """
        if batch_size > len(files):
            # print("Warning: batch size is bigger than the data size. " "Setting batch size to data size")
            batch_size = len(files)

        assert len(files) > 0, "No files to extract features from."

        n_batches = len(files) // batch_size
        batch_sizes = [batch_size for _ in range(n_batches)]
        if len(files) % batch_size != 0:
            batch_sizes.append(len(files) % batch_size)

        transform = StandardTransform(self.input_size, self.interpolation, preprocess_transform)
        self.extractor = self.extractor.to(device, dtype=torch.float32)

        act_array = []
        start, end = 0, 0
        for bsize in batch_sizes:
            end = start + bsize
            images = [transform(Image.open(f).convert("RGB")) for f in files[start:end]]
            batch = torch.stack(images).to(device)
            batch = self(batch).detach().cpu().numpy()
            act_array.append(batch)
            start += bsize
        return np.concatenate(act_array, axis=0)

    @torch.no_grad()
    def extract_features_from_dataloader(
        self,
        dataloader: DataLoader,
        normalize: bool = False,
        device: str = "cpu",
    ):
        """Extract features from a DataLoader.

        Parameters
        ----------
        dataloader : DataLoader
            PyTorch DataLoader
        normalize : bool, optional
            Wether to normalize the images to 0.5 mean and 0.5 std across channels.
            Assumes that the images are in the [0, 1] range, by default False
        device : str, optional
            device to use, by default "cpu"

        Notes
        -----
        Has the following assumptions:
         - Dataloader outputs a tuple with the first element being the image batch tensor.
         - Image tensors are of shape (batch_size, channels, height, width)
         - Tensors are in the [0, 1] range in which case you should set `normalize` to True.
         - Tensors have been normalized to the [-1, 1] range (0.5 mean and 0.5 std across channels) or to another range (e.g. ImageNet normalization).

        It is recommended to disable the `shuffle` option in the DataLoader for consistency.

        Depending on the model, you might need to resize the images to the model's input size.

        Returns
        -------
        np.ndarray
            array of features with shape (n_samples, n_features)
        """
        self.extractor = self.extractor.to(device, dtype=torch.float32)

        # validation dry run
        sample_batch = next(iter(dataloader))
        assert isinstance(
            sample_batch[0], torch.Tensor
        ), f"First element of the tuple must be a torch.Tensor. Got {type(sample_batch[0])}."

        if sample_batch[0].shape[2:] != self.input_size:
            logger.warning(
                f"Model default input size {self.input_size} does not match the size of the images in the dataloader {sample_batch[0][2:]}. Might lead to execution errors."
            )

        act_array = []
        for batch in tqdm(dataloader, total=len(dataloader)):
            images = batch[0]
            images = images.to(device, dtype=torch.float32)
            if normalize:
                images = images * 2.0 - 1.0
            act_array.append(self(images).detach().cpu().numpy())
        return np.concatenate(act_array, axis=0)

    @torch.no_grad()
    def _extract_features_dataloader(
        self,
        dataloader: DataLoader,
        device: str = "cpu",
        preprocess_transform: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Internal method to extract features from a DataLoader.

        Parameters
        ----------
        dataloader : DataLoader
            image dataloader
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: extracted features, labels, and image ids
        """
        logger.info("Extracting image features.")
        act_array = []
        labels_array = []
        ids_array = []

        self.extractor = self.extractor.to(device, dtype=torch.float32)
        dataloader.dataset.transform = StandardTransform(self.input_size, self.interpolation, preprocess_transform)
        for batch, labels, img_ids in tqdm(dataloader, total=len(dataloader)):
            batch = batch.to(device, dtype=torch.float32)
            batch = self(batch).detach().cpu().numpy()
            act_array.append(batch)
            labels_array.append(labels)
            ids_array.append(img_ids)

        return (
            np.concatenate(act_array, axis=0),
            np.concatenate(labels_array, axis=0),
            np.concatenate(ids_array, axis=0),
        )

    def forward(self, x):
        return self.extractor(x)
