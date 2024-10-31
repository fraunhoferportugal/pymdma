from .imagenet import DinoExtractor, InceptionFID, VGGExtractor, ViTExtractor


class ExtractorFactory:
    default = "vit_b_32"

    @staticmethod
    def model_from_name(
        name: str,
    ):
        """Initializes the feature extractor with the given parameters.

        Args:
            name (str): identifier of the extractor to be used.
            device (str): model device. Defaults to "cpu".
        """
        # name = self.default if name == "default" else name

        if name == "inception_fid":
            extractor = InceptionFID()
        elif "vgg" in name:
            extractor = VGGExtractor(name)
        elif "dino" in name:
            extractor = DinoExtractor(name)
        elif "vit" in name:
            extractor = ViTExtractor(name)
        else:
            raise ValueError(f"Model {name} not available.")

        return extractor.eval()
