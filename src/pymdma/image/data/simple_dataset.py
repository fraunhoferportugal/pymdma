from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# TODO support other formats?
SUPPORTED_FILES = {".png", ".jpg", ".jpeg"}


class SimpleDataset(Dataset):
    def __init__(
        self,
        file_paths: List[Path],
        transforms: Optional[Callable] = None,
        target_transforms: Optional[Callable] = None,
        train_transforms: Optional[Callable] = None,
    ) -> None:
        self.img_files = file_paths
        self.transform = transforms
        self.target_transform = target_transforms
        self.train_transform = train_transforms

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB")
        image_id = Path(img_path).stem

        label = 0
        if self.transform:
            image = self.train_transform(image) if self.train_transform else image
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, image_id


def build_img_dataloader(
    file_paths: List[Path],
    transforms: Optional[Callable] = None,
    batch_size: int = 10,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """Builds a PyTorch DataLoader for a list of image files.

    Parameters
    ----------
    file_paths : List[Path]
        List of image file paths.
    transforms : Optional[Callable], optional
        Transforms to be applied to the images, by default None.
    batch_size : int, optional
        Batch size, by default 10.
    num_workers : int, optional
        Number of workers, by default 0.
    **kwargs
        Additional keyword arguments to be passed to the DataLoader.

    Returns
    -------
    DataLoader
        PyTorch DataLoader for the image files. Returns a tuple of (image, label, image_id)
    """
    dataset = SimpleDataset(
        file_paths=file_paths,
        transforms=transforms if transforms is not None else np.array,
    )

    def _custom_collate_fn(batch):
        return zip(*batch)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_custom_collate_fn,
        **kwargs,
    )
