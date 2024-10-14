from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.multiprocessing
import torchvision.models as tvmodels
from loguru import logger
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from pymdma.common.definitions import EmbedderInterface

from .extractor import BaseExtractor


class ExtractorFactory(EmbedderInterface):
    default = "dino_vits8"

    def model_from_name(
        self,
        name: str,
        device: str = "cpu",
        **kwargs,
    ):
        """Initializes the feature extractor with the given parameters.

        Args:
            name (str): identifier of the extractor to be used.
            device (str): model device. Defaults to "cpu".
        """
        name = self.default if name == "default" else name
        super().__init__(name)
        self.device = device

        if name == "inception_v3":
            self.extractor = InceptionExtractor(**kwargs)
        elif "vgg" in name:
            self.extractor = VGGExtractor(model_name=name)
        elif "dino" in name:
            self.extractor = DinoExtractor(name)
        elif "vit" in name:
            self.extractor = ViTExtractor(name)
        else:
            raise ValueError(f"Model {name} not available.")

        self.extractor.eval()

    def get_transform(self):
        return self.extractor.transform

    # def extract_features_from_files(self, files: List[Path], batch_size: int = 50):
    #     """Extract features from a list of image files.

    #     Args:
    #         files (List[Path]): list of paths to image files
    #         batch_size (int): batch size for feature extraction. Defaults to 50.

    #     Returns:
    #         np.ndarray: array of features
    #     """
    #     if batch_size > len(files):
    #         # print("Warning: batch size is bigger than the data size. " "Setting batch size to data size")
    #         batch_size = len(files)

    #     assert len(files) > 0, "No files to extract features from."

    #     n_batches = len(files) // batch_size
    #     batch_sizes = [batch_size for _ in range(n_batches)]
    #     if len(files) % batch_size != 0:
    #         batch_sizes.append(len(files) % batch_size)

    #     act_array = []
    #     start, end = 0, 0
    #     for bsize in batch_sizes:
    #         end = start + bsize
    #         images = [Image.open(f).convert("RGB") for f in files[start:end]]
    #         transform = self.extractor.transform
    #         batch = np.array([transform(x).numpy() for x in images], dtype=np.float32)
    #         batch = torch.from_numpy(batch).to(self.device)
    #         batch = self.extractor.extract(batch).detach().cpu().numpy()
    #         act_array.append(batch)
    #         start += bsize
    #     return np.concatenate(act_array, axis=0)

    # @torch.no_grad()
    # def extract_features_dataloader(
    #     self,
    #     dataloader: DataLoader,
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """Use selected model to extract features from all images in
    #     dataloader.

    #     Args:
    #         dataloader (DataLoader): image dataloader
    #     Returns:
    #         Tuple[np.ndarray, np.ndarray]: array of features and array of image labels
    #     """
    #     logger.info("Extracting image features.")
    #     act_array = []
    #     labels_array = []
    #     ids_array = []

    #     self.extractor._model.to(self.device)
    #     dataloader.dataset.transform = self.extractor.transform
    #     for batch, labels, img_ids in tqdm(dataloader, total=len(dataloader)):
    #         batch = batch.to(self.device)
    #         batch = self.extractor.extract(batch).detach().cpu().numpy()
    #         act_array.append(batch)
    #         labels_array.append(labels)
    #         ids_array.append(img_ids)

    #     return (
    #         np.concatenate(act_array, axis=0),
    #         np.concatenate(labels_array, axis=0),
    #         np.concatenate(ids_array, axis=0),
    #     )


class InceptionExtractor(BaseExtractor):
    def __init__(self):
        super().__init__(
            input_size=(299, 299),
            interpolation=Image.Resampling.BILINEAR,
        )

        self.extractor = tvmodels.inception_v3(weights=tvmodels.Inception_V3_Weights.DEFAULT)
        print(self.extractor)

        self.activation = {}

        def get_activation(name):
            def hook(model, inp, output):
                self.activation[name] = output.detach()

            return hook

        # register hook to obtain activations at avgpool layer
        self.extractor.avgpool.register_forward_hook(get_activation("avgpool"))

    def forward(self, x):
        self.extractor(x)
        return self.activation["avgpool"][:, :, 0, 0]


class VGGExtractor(BaseExtractor):
    def __init__(self, model_name: str) -> None:
        super().__init__(
            input_size=(224, 224),
            interpolation=Image.Resampling.BILINEAR,
        )

        weights = tvmodels.vgg.__dict__[f"{model_name.upper()}_Weights"].DEFAULT
        self.extractor = tvmodels.vgg.__dict__[model_name](weights=weights)
        # remove classifier head
        self.extractor.classifier = self.extractor.classifier[:-1]

    def forward(self, x):
        return self.extractor(x)


class ViTExtractor(BaseExtractor):
    def __init__(self, model_name: str) -> None:
        super().__init__(
            input_size=(224, 224),
            interpolation=Image.Resampling.BILINEAR,
        )

        weights = tvmodels.vision_transformer.__dict__[f"{model_name.upper().replace('VIT', 'ViT')}_Weights"].DEFAULT
        self.extractor = tvmodels.vision_transformer.__dict__[model_name](weights=weights)

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self.extractor._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.extractor.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.extractor.encoder(x)
        # # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x


class DinoExtractor(BaseExtractor):
    def __init__(self, model_name) -> None:
        super().__init__(
            input_size=(224, 224),
            interpolation=Image.Resampling.BICUBIC,
        )

        # get model from the hub without classifier heads
        self.extractor = torch.hub.load(
            f"facebookresearch/{model_name.split('_')[0]}:main",
            model_name,
            pretrained=True,
        )

    def forward(self, batch):
        return self.extractor(batch)
