import torch
import torch.multiprocessing
import torchvision.models as tvmodels
from PIL import Image
from piq.feature_extractors import InceptionV3

from .extractor import BaseExtractor


class InceptionFID(BaseExtractor):
    def __init__(self):
        super().__init__(
            input_size=(299, 299),
            interpolation=Image.Resampling.BILINEAR,
        )

        self.extractor = InceptionV3(normalize_input=False)

    def forward(self, x):
        N = len(x)
        x = self.extractor(x)
        return x[0].view(N, -1)


class VGGExtractor(BaseExtractor):
    def __init__(self, model_name: str) -> None:
        super().__init__(
            input_size=(224, 224),
            interpolation=Image.Resampling.BILINEAR,
        )

        weights = tvmodels.vgg.__dict__[f"{model_name.upper()}_Weights"].DEFAULT
        self.extractor = tvmodels.vgg.__dict__[model_name](weights=weights)
        # features from the last fully connected layer
        self.extractor.classifier = self.extractor.classifier[:-2]

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
    def __init__(self, model_name, input_size: tuple[int, int] = (224, 224)) -> None:
        super().__init__(
            input_size=input_size,
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
