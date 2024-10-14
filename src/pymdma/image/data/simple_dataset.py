from pathlib import Path
from typing import Callable, List, Optional

from PIL import Image
from torch.utils.data import Dataset

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
