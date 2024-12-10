import json
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from torch.utils.data import DataLoader

from pymdma.common.definitions import InputLayer
from pymdma.constants import ReferenceType, ValidationDomain

from .data.simple_dataset import SimpleDataset
from .models.features import ExtractorFactory
from .utils.processing import batch_downsample_to_largest

SUPPORTED_FILES = {".png", ".jpg", ".jpeg", ".bmp"}  # TODO might want to add others


def no_collate_fn(data):
    # A simple collate_fn for torch dataloaders that prevents the default
    # collate_fn from stacking the images into a single tensor. Used in input validation

    # images | labels | ids
    return [d[0] for d in data], [d[1] for d in data], [d[2] for d in data]


def source_to_list(data_src: Union[Path, List[Path]]) -> List[Path]:
    """Helper function that transforms a data source to a list of paths to each
    file.

    Args:
        data_src (Union[Path, List[Path]]): source directory or list of paths to the images.

    Returns:
        List[Path]: list of paths to the images.
    """
    if isinstance(data_src, list):
        return data_src
    return [im for im in data_src.iterdir() if im.suffix in SUPPORTED_FILES] if data_src else []


def _annotation_from_folder(data_src: Union[Path, None]) -> Path:
    """Helper function that returns the annotation file from a folder."""
    if not data_src:
        return None

    if data_src.is_dir():
        return next(data_src.glob("*.json"), None)
    elif data_src.suffix == ".json":
        return data_src


class ImageInputLayer(InputLayer):
    """Abstraction layer for handling different types of input data with the
    given requirements from the data auditing module.

    Starts by creating the dataloaders for the reference and target
    datasets (if needed), and then send batches of images in the
    required format by the metrics functions.

    Parameters
    ----------
    validation_domain : ValidationDomain
        valition type (input or synthetic)
    reference_type : ReferenceType
        reference type (dataset or single image)
    target_data : Union[Path, List[Path]]
        data root or list of paths to the target dataset
    reference_data : Optional[Union[Path, List[Path]]], optional
        data root or list of paths to the reference dataset, by default None
    annot_file : Optional[Path], optional
        path to the COCO annotation file of the reference dataset, by default None
    repeat_reference : bool, optional
        If True, the reference dataset will be repeated to match the target dataset size, by default False
    batch_size : int, optional
        Batch size to be in the image batch, by default 20
    device : str, optional
        Device to be used in the feature extraction module, by default "cpu"
    features_cache : Optional[Path], optional
        Path to the cache file for the features, by default
    """

    def __init__(
        self,
        validation_domain: ValidationDomain,
        reference_type: ReferenceType,
        target_data: Union[Path, List[Path]],
        reference_data: Optional[Union[Path, List[Path]]] = None,
        annot_file: Optional[Path] = None,
        repeat_reference: bool = False,
        batch_size: int = 20,
        device: str = "cpu",
        features_cache: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.val_type = validation_domain
        self.reference_type = reference_type
        self.batch_size = batch_size
        self.device = device
        self.features_cache = features_cache

        annot_file = _annotation_from_folder(annot_file)

        self._annotations = None
        if annot_file is not None:
            annot_file = Path(annot_file)
            assert annot_file.exists() and annot_file.suffix == ".json", f"Invalid annotation file {annot_file}"
            self._annotations = json.load(annot_file.open("r"))
        # ids for the images in instance analysis
        # used later for instance level metrics ids
        self.instance_ids = []  # if not self.annotations else [img["id"] for img in self.annotations.dataset["images"]]

        self.transform = np.asarray

        # do not use default collate_fn for input validation (autoconverts images to tensors in the range of 0-1)
        collate_fn = no_collate_fn if self.val_type == ValidationDomain.INPUT else None

        reference_files = source_to_list(reference_data)
        target_files = source_to_list(target_data)
        if self.reference_type != ReferenceType.NONE:
            # repeat reference dataset to match the target dataset size
            # (only in specific cases like input validation with instance reference type)
            if repeat_reference and len(reference_files) < len(target_files):
                logger.warning(
                    "Reference dataset is smaller than the target dataset. Repeating the reference dataset to match the target dataset size.",
                )
                reference_files = (
                    reference_files * (len(target_files) // len(reference_files))
                    + reference_files[: len(target_files) % len(reference_files)]
                )

            # prepare reference dataloader (real/reference images)
            reference_dataset = SimpleDataset(
                file_paths=reference_files,
                transforms=self.transform,
            )
            self.reference_loader = DataLoader(
                reference_dataset,
                batch_size=self.batch_size if self.batch_size > 0 else len(reference_files),
                shuffle=False,
                num_workers=4,
                collate_fn=collate_fn,  # no need to collate images
            )

            assert self.val_type == ValidationDomain.SYNTH or (
                self.val_type == ValidationDomain.INPUT and len(target_files) == len(reference_files)
            ), "Reference and target datasets must have the same size for full reference input validation."

        # prepare target dataloader (evaluation images)
        target_dataset = SimpleDataset(file_paths=target_files, transforms=self.transform)
        self.target_loader = DataLoader(
            target_dataset,
            batch_size=self.batch_size if self.batch_size > 0 else len(target_files),
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,  # no need to collate images
        )

        self.instance_ids = [path.stem for path in target_dataset.img_files]

    def __len__(self):
        return len(self.target_loader.dataset)

    @property
    def annotations(self) -> Dict["str", any]:
        assert self._annotations is not None, "Annotations were not provided for this dataset."
        return self._annotations

    @property
    def dataset_files(self) -> List[Path]:
        return self.target_loader.dataset.img_files

    # TODO separate embeddings to support multiple models like in the case of FID
    def get_embeddings(
        self,
        model_name: str,
        model_instances: Optional[Dict[str, callable]] = None,
        offload_model: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts the features from the target and reference datasets using
        the given model.

        Parameters
        ----------
        model_name : str
            name of the model to be used for feature extraction
        model_instances : Optional[Dict[str, callable]], optional
            dictionary of model instances, by default None
        offload_model : bool, optional
            If True, the model will be offloaded to the CPU after the feature extraction, by default False

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            features of the reference and target datasets
        """

        extractor = None
        if model_instances is not None:
            if model_name in model_instances:
                extractor = model_instances[model_name]
            elif model_name == "default" and ExtractorFactory.default in model_instances:
                extractor = model_instances[ExtractorFactory.default]

        if extractor is None:
            offload_model = True  # unload model when finished
            model_name = ExtractorFactory.default if model_name == "default" else model_name
            extractor = ExtractorFactory.model_from_name(model_name) if extractor is None else extractor

        # extractor = model_instance if model_instance is not None else FeatureExtractor(model_name, device=self.device)
        reference_feats, _labels, _reference_ids = extractor._extract_features_dataloader(
            self.reference_loader,
            device=self.device,
        )
        synthetic_feats, _labels, synthetic_ids = extractor._extract_features_dataloader(
            self.target_loader,
            device=self.device,
        )

        if not self.instance_ids:
            self.instance_ids = synthetic_ids.tolist()

        if offload_model:
            extractor = extractor.to("cpu")
            del extractor

        return reference_feats, synthetic_feats

    @property
    def batched_samples(self) -> Generator[Tuple[np.ndarray], None, None]:
        """Returns batches of images from the target dataset in the required
        format."""
        if self.reference_type == ReferenceType.NONE:
            for no_ref_images, _labels, _img_ids in self.target_loader:
                yield (no_ref_images,)
        else:
            ref_iter, sim_iter = iter(self.reference_loader), iter(self.target_loader)
            for _ in range(len(self.reference_loader)):
                ref_imgs, _labels, _img_ids = next(ref_iter)
                sim_imgs, _labels, _img_ids = next(sim_iter)

                ref_imgs, sim_imgs = batch_downsample_to_largest(ref_imgs, sim_imgs)
                yield ref_imgs, sim_imgs
