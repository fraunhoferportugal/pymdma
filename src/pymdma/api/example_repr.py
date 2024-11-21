import os
from pathlib import Path
from typing import Optional

from ..common.selection import select_modality_input_layer
from ..constants import DataModalities, ReferenceType, ValidationDomain

# Access point to data in the host
# this will not be used in the final implementation
# as dataloaders are supposed to be serialized
_CONTAINER_DATA_ROOT = os.getenv("API_DATA_ROOT", "./data/test")
_COMPUTE_DEVICE = os.getenv("COMPUTE_DEVICE", "cpu")


def prepare_input_layer(
    data_modality: DataModalities,
    validation_domain: ValidationDomain,
    reference_type: ReferenceType,
    annotation_file: Optional[str] = None,
):
    assert data_modality in {"image", "time_series", "tabular", "text"}, f"Unknown data modality {data_modality}"

    batch_size = os.getenv(f"{data_modality.value.upper}_BATCH_SIZE", 10)

    if reference_type != ReferenceType.NONE:
        reference_data = Path(f"{_CONTAINER_DATA_ROOT}/{data_modality.value}/{validation_domain.value}/reference")
    else:
        reference_data = None

    target_data = Path(f"{_CONTAINER_DATA_ROOT}/{data_modality.value}/{validation_domain.value}/dataset")

    if annotation_file is not None:
        assert validation_domain == ValidationDomain.INPUT, "Annotations are only supported for input validation"
        annotation_file = Path(
            f"{_CONTAINER_DATA_ROOT}/{data_modality.value}/{validation_domain.value}/annotations/{annotation_file}",
        )

    return select_modality_input_layer(
        data_modality,
        validation_domain,
        reference_type,
        target_data,
        reference_data,
        batch_size,
        device=_COMPUTE_DEVICE,
        annotation_file=annotation_file,
    )
