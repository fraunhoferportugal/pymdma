from typing import Dict, List, Optional, Tuple, Union

from loguru import logger
from pycocotools.coco import COCO

from pymdma.common.definitions import Metric
from pymdma.common.output import MetricResult
from pymdma.constants import AnnotationType, EvaluationLevel, MetricGroup, OutputsTypes, ReferenceType

_SUPPORTED_ANNOT_TYPES = {
    "segmentation": AnnotationType.MASK,
    "bbox": AnnotationType.BBOX,
    "keypoints": AnnotationType.KEYPOINTS,
    "category_id": AnnotationType.LABEL,
}


def infer_annotation_types(dataset: Dict[str, any]) -> List[AnnotationType]:
    """Infer annotation types from the dataset."""
    # infer annotation types from dataset
    annotation_types = set()
    for annotation in dataset["annotations"]:
        annotation_types.update(annotation.keys() & _SUPPORTED_ANNOT_TYPES.keys())
    annotation_types = [_SUPPORTED_ANNOT_TYPES[annot_type] for annot_type in annotation_types]
    # logger.info(f"Annotation types found: {annotation_types}")
    return annotation_types


def get_anns_from_img_id(dataset: Dict[str, any], img_id: int) -> List[int]:
    """Get all annotations from a single image."""
    return [ann for ann in dataset["annotations"] if "image_id" in ann and ann["image_id"] == img_id]


def get_anns_from_cat_id(dataset: Dict[str, any], cat_id: int) -> List[int]:
    """Get all annotations with a specific category id."""
    return [ann for ann in dataset["annotations"] if "category_id" in ann and ann["category_id"] == cat_id]


class DatasetCompletness(Metric):
    """Evalute the completeness of the COCO dataset.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments for compatibility (unused).

    References
    ----------
    Lin et al., Microsoft COCO: Common Objects in Context (2014).
    https://arxiv.org/abs/1405.0312

    Examples
    --------
    >>> import json
    >>> dataset_completness = DatasetCompletness()
    >>> with open("annotations.json", "r") as f:
    ...     dataset = json.load(f)
    >>> result: MetricResult = dataset_completness.compute(dataset)
    """

    reference_type = ReferenceType.NONE
    evaluation_level = [EvaluationLevel.DATASET, EvaluationLevel.INSTANCE]
    metric_group = MetricGroup.VALIDITY
    annotation_type = [AnnotationType.LABEL, AnnotationType.BBOX, AnnotationType.MASK, AnnotationType.KEYPOINTS]

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def _completness_labels(self, dataset: Dict[str, any]) -> List[int]:
        """Checks whether all images have at least one category assigned and if
        it is different from 0.

        In COCO format format, a categoy_id = 0 means no label.
        """
        img_id_no_label = []

        category_ids = {cat["id"] for cat in dataset["categories"]}

        for image in dataset["images"]:
            annotations_for_image = get_anns_from_img_id(dataset, image["id"])
            # annotations_for_image = dataset.loadAnns(dataset.getAnnIds(imgIds=image_id))
            if len(annotations_for_image) == 0:
                # logger.warning(f"No annotations found for image {image_id}")
                img_id_no_label.append(image["id"])

            for annot in annotations_for_image:
                if "category_id" not in annot:
                    # logger.warning(f"Category_id not found for annotation {annot['id']}")
                    img_id_no_label.append(annot["id"])
                    continue
                if annot["category_id"] == 0 or annot["category_id"] not in category_ids:
                    img_id_no_label.append(annot["id"])
                    # logger.warning(f"Invalid label for image {image_id}")
        return img_id_no_label

    def _completness_annots(self, dataset: Dict[str, any]) -> List[int]:
        """Checks whether all images have at least one annotation for each
        annotation type type (bbox, mask, or keypoints).

        Returns
        -------
        value - list where the first element is a bool indicating success (True) or not (False) of the metric and the second element is a list representing the images ids where the metric failed (if verified).
        """

        annotation_types = {"bbox", "segmentation", "keypoints"}
        invalid_annot_ids = []
        for annot in dataset["annotations"]:
            if not annot.keys() & annotation_types:
                # logger.warning(f"Annotation {annot['id']} does not have any annotation.")
                invalid_annot_ids.append(annot["id"])
                continue
        return invalid_annot_ids

    def _coco_parsing(self, dataset: Dict[str, any]) -> Union[str, None]:
        try:
            coco = COCO(None)
            coco.dataset = dataset
            coco.createIndex()
        except Exception as e:
            logger.error(f"Error parsing COCO file: {e}")
            return f"Non standard COCO annotation, review incorrect fields: {str(e)}"
        return None

    def compute(
        self,
        dataset: Dict[str, any],
        **kwargs,
    ) -> MetricResult:
        """Compute the dataset completeness metric.

        Parameters
        ----------
        dataset : dict
            Dictionary with COCO dataset format.

        Returns
        -------
        result : MetricResult
            - Dataset-level metric result with the following keys:
                - "img_no_annot" - list of image ids with no annotations.
                - "annot_missing_fields" - list of annotation ids with missing fields.
            - Errors - list of errors found during the evaluation.
        """

        img_missing_labels = self._completness_labels(dataset)
        annot_missing_fields = self._completness_annots(dataset)

        coco_parsing_error = self._coco_parsing(dataset)

        return MetricResult(
            dataset_level={
                "dtype": OutputsTypes.KEY_ARRAY,
                "subtype": "int",
                "value": {
                    "img_no_annot": img_missing_labels,
                    "annot_missing_fields": annot_missing_fields,
                },
            },
            errors=[coco_parsing_error] if coco_parsing_error is not None else None,
        )


class AnnotationCorrectness(Metric):
    """Evalute the annotation correctness of the COCO dataset.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments for compatibility (unused).

    References
    ----------
    Lin et al., Microsoft COCO: Common Objects in Context (2014).
    https://arxiv.org/abs/1405.0312

    Examples
    --------
    >>> import json
    >>> ann_correct = AnnotationCorrectness()
    >>> with open("annotations.json", "r") as f:
    ...     dataset = json.load(f)
    >>> result: MetricResult = ann_correct.compute(dataset)
    """

    reference_type = ReferenceType.NONE
    evaluation_level = [EvaluationLevel.DATASET, EvaluationLevel.INSTANCE]
    metric_group = MetricGroup.VALIDITY
    annotation_type = [AnnotationType.LABEL, AnnotationType.BBOX, AnnotationType.MASK, AnnotationType.KEYPOINTS]

    def __init__(
        self,
        annotation_types: List[AnnotationType] = None,
        num_keypoints: Optional[int] = None,
        mask_area_range: Optional[Tuple[int, int]] = None,
        valid_label_names: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.annotation_types = annotation_types
        self.num_keypoints = num_keypoints

        assert (
            mask_area_range is None or type(mask_area_range) in {tuple, list} and len(mask_area_range) == 2
        ), "mask_area_range must be a tuple or list with two elements."
        self.min_mask_area = mask_area_range[0] if mask_area_range else None
        self.max_mask_area = mask_area_range[1] if mask_area_range else None

        self.valid_label_names = valid_label_names

    def _correctness_bbox(self, dataset: Dict[str, any]) -> List[int]:
        """Checks whether the bounding box coordinates are valid (i.e., have 4
        coords)."""
        invalid_bbox_ann_ids = []
        for annotation in dataset["annotations"]:
            if "bbox" in annotation and len(annotation["bbox"]) != 4:
                invalid_bbox_ann_ids.append(annotation["id"])
                # logger.warning(f"Invalid bbox for annotation {annotation['id']}: {annotation['bbox']}")
        return invalid_bbox_ann_ids

    def _correctness_keypoints(self, dataset: Dict[str, any]) -> List[int]:
        """Checks whether the number of keypoints is valid for each image."""
        annot_ids_incomplete_keypoints = []

        if not self.num_keypoints:
            logger.warning("Number of keypoints not provided. Skipping keypoint evaluation.")
            return []

        for annotation in dataset["annotations"]:
            if "keypoints" not in annotation:
                continue
            kp = annotation["keypoints"]
            if len(kp) != self.num_keypoints:
                annot_ids_incomplete_keypoints.append(annotation["id"])
                # logger.warning(f"Invalid number of keypoints for annotation {annotation['id']}: {kp}")

    def _correctness_mask(self, dataset: Dict[str, any]) -> Dict[str, List[int]]:
        """Checks whether the mask area, counts, size, and bbox are valid for
        each mask annotation."""
        area_mask_bounds = []
        missing_area = []
        missing_counts = []
        missing_size = []
        missing_bbox = []

        for annotation in dataset["annotations"]:
            if "segmentation" not in annotation:
                continue
            if "area" not in annotation:
                missing_area.append(annotation["id"])
            else:
                area = annotation["area"]
                if (self.min_mask_area is not None and area < self.min_mask_area) or (
                    self.max_mask_area is not None and area > self.max_mask_area
                ):
                    area_mask_bounds.append(annotation["id"])
                    # logger.warning(f"Invalid area for annotation {annotation['id']}: {area}")
            if "counts" not in annotation["segmentation"]:
                # logger.warning(f"Missing counts for annotation {annotation['id']}")
                missing_counts.append(annotation["id"])
            if "size" not in annotation["segmentation"]:
                # logger.warning(f"Missing size for annotation {annotation['id']}")
                missing_size.append(annotation["id"])
            if "bbox" not in annotation:
                # logger.warning(f"Missing bbox for annotation {annotation['id']}")
                missing_bbox.append(annotation["id"])
        return {
            "annots_mask_oob": area_mask_bounds,
            "annots_missing_area": missing_area,
            "annots_missing_counts": missing_counts,
            "annots_missing_size": missing_size,
            "annots_missing_bbox": missing_bbox,
        }

    def _correctness_labels(self, dataset: Dict[str, any]) -> List[int]:
        """Checks whether the categories contained in the annotations file are
        valid and if annotation corresponds to an invalid category id."""
        invalid_labels = []
        annots_ids_invalid_labels = []

        for cat_values in dataset["categories"]:
            cat_id = cat_values["id"]
            if self.valid_label_names and cat_values["name"] not in self.valid_label_names:
                invalid_labels.append(cat_id)
                # logger.warning(f"Invalid category found: {cat_values['name']}")
                invalid_cat_anns = get_anns_from_cat_id(dataset, cat_id)
                annots_ids_invalid_labels.extend([ann["id"] for ann in invalid_cat_anns])
        return invalid_labels, annots_ids_invalid_labels

    def compute(
        self,
        dataset: Dict[str, any],
        **kwargs,
    ) -> MetricResult:
        """Compute the annotation completeness metric.

        Parameters
        ----------
        dataset : dict
            Dictionary with COCO dataset format.

        Returns
        -------
        result : MetricResult
            - Dataset-level metric result with the following keys:
                - "invalid_labels" - list of categories with invalid label names.
                - "annots_invalid_label" - list of annotation ids with invalid label names.
                - "annots_invalid_bbox" - list of annotation ids with invalid bounding boxes.
                - "annots_invalid_kp" - list of annotation ids with invalid keypoints.
                - "annots_mask_oob" - list of annotation ids with masks out of bounds.
                - "annots_missing_area" - list of annotation ids with missing area.
                - "annots_missing_counts" - list of annotation ids with missing counts.
                - "annots_missing_size" - list of annotation ids with missing size.
                - "annots_missing_bbox" - list of annotation ids with missing bounding boxes.
        """

        annotation_errors = {}
        self.annotation_types = (
            infer_annotation_types(dataset) if self.annotation_types is None else self.annotation_types
        )
        for annot_type in self.annotation_types:
            if annot_type == AnnotationType.LABEL:
                invalid_labels, annots_ids_invalid_labels = self._correctness_labels(dataset)
                annotation_errors["invalid_labels"] = invalid_labels
                annotation_errors["annots_invalid_label"] = annots_ids_invalid_labels
            if annot_type == AnnotationType.BBOX:
                invalid_bbox_ann_ids = self._correctness_bbox(dataset)
                annotation_errors["annots_invalid_bbox"] = invalid_bbox_ann_ids
            if annot_type == AnnotationType.KEYPOINTS:
                annot_ids_incomplete_keypoints = self._correctness_keypoints(dataset)
                annotation_errors["annots_invalid_kp"] = annot_ids_incomplete_keypoints
            if annot_type == AnnotationType.MASK:
                mask_errors = self._correctness_mask(dataset)
                annotation_errors.update(mask_errors)

        return MetricResult(
            dataset_level={"dtype": OutputsTypes.KEY_ARRAY, "subtype": "int", "value": annotation_errors},
        )


class AnnotationUniqueness(Metric):
    """Evalute the annotation uniqueness of the COCO dataset.

    Parameters
    ----------
    annotation_types : list
        List of annotation types to evaluate. Default is ["bbox", "segmentation", "keypoints"].
    **kwargs : dict, optional
        Additional keyword arguments for compatibility (unused).

    Notes
    -----
    This metric checks if there are duplicated annotations for each annotation type (bbox, mask, or keypoints) for the same image and if they have different categories assigned.
    This metric should only be used if the goal of your dataset is to have unique annotations for each image.

    References
    ----------
    Lin et al., Microsoft COCO: Common Objects in Context (2014).
    https://arxiv.org/abs/1405.0312

    Examples
    --------
    >>> import json
    >>> ann_unique = AnnotationUniqueness()
    >>> with open("annotations.json", "r") as f:
    ...     dataset = json.load(f)
    >>> result: MetricResult = ann_unique.compute(dataset)
    """

    reference_type = ReferenceType.NONE
    evaluation_level = [EvaluationLevel.DATASET, EvaluationLevel.INSTANCE]
    metric_group = MetricGroup.VALIDITY
    annotation_type = [AnnotationType.BBOX, AnnotationType.MASK, AnnotationType.KEYPOINTS]

    def __init__(
        self,
        annotation_types: List[AnnotationType] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.annotation_types = annotation_types
        self.annotation_types = (
            {"bbox", "segmentation", "keypoints"} if self.annotation_types is None else self.annotation_types
        )

    def compute(
        self,
        dataset: Dict[str, any],
        **kwargs,
    ) -> MetricResult:
        """Compute the annotation completeness metric.

        Parameters
        ----------
        dataset : dict
            Dictionary with COCO dataset format.

        Returns
        -------
        result : MetricResult
            Dataset-level metric results with a dictionary that maps from image id to a list of repeated annotations
        """
        imgs_ids_repeated_annot = {}
        # image_id -> category_id -> annotation_types
        image_annot_type_map = {}
        for ann in dataset["annotations"]:
            if "image_id" not in ann or "category_id" not in ann:
                continue
            img_id = str(ann["image_id"])
            annot_cat = ann["category_id"]
            annot_types = list(ann.keys() & self.annotation_types)
            image_annot_type_map.setdefault(img_id, dict())
            image_annot_type_map[img_id].setdefault(annot_cat, set())

            # check if annotation type already exists for the image
            if annot_cat in image_annot_type_map[img_id]:
                if any(annot_type in image_annot_type_map[img_id][annot_cat] for annot_type in annot_types):
                    imgs_ids_repeated_annot.setdefault(img_id, list()).append(ann["id"])

            image_annot_type_map[img_id][annot_cat].update(annot_types)

        return MetricResult(
            dataset_level={"dtype": OutputsTypes.KEY_ARRAY, "subtype": "int", "value": imgs_ids_repeated_annot},
        )


__all__ = ["DatasetCompletness", "AnnotationCorrectness", "AnnotationUniqueness"]
