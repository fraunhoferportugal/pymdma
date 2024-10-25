import torch
import torch.multiprocessing
import torchvision.models as tvmodels
from PIL import Image

from .extractor import BaseExtractor


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
