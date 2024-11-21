import traceback
from contextlib import asynccontextmanager
from os import getenv
from pathlib import Path
from typing import Dict, Union

from fastapi import APIRouter, Depends, FastAPI, status
from fastapi.responses import JSONResponse
from loguru import logger

if not __package__:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    # sys.path.insert(0, Path(__file__).parent.parent)

from pymdma.api.api_types import (
    DatasetEvalResponse,
    DatasetParams,
    InstanceEvalResponse,
    Message,
    MetricInfo,
    MetricInfoParams,
    SpecificFunctionParams,
)
from pymdma.api.example_repr import prepare_input_layer
from pymdma.api.hooks import load_models_hook
from pymdma.common.compute import ComputationManager
from pymdma.common.output import create_output_structure
from pymdma.common.selection import get_metrics_metadata, select_metric_functions, select_specific_metric_functions
from pymdma.constants import DataModalities

ml_models = {}
_compute_nworkers = int(getenv("COMPUTE_N_WORKERS", 1))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load inference models when the app starts and clear them when the app
    stops.

    Args:
        app (FastAPI): fast api application instance.
    """
    if getenv("PERSISTENT_MODELS", "false").lower() == "true":
        logger.info("Loading models in persistent mode.")
        load_models_hook(ml_models, device=getenv("COMPUTE_DEVICE", "cpu"))
    yield
    ml_models.clear()


app = FastAPI(
    title="Data Audting Metrics API",
    version="0.2",
    description="API for calculating metrics to evaluate data.",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    return {"message": "Welcome from the API."}


# Public endpoint.
@app.get("/api/healthcheck")
async def health_check():
    """Health check."""
    return {"message": "API is alive!"}


metrics_router = APIRouter(
    prefix="/metrics",
    tags=["metrics"],
)


@metrics_router.get(
    "/info",
    response_model=Dict[str, MetricInfo],
    responses={status.HTTP_404_NOT_FOUND: {"model": Message}},
)
def metric_info(
    params: MetricInfoParams = Depends(),
) -> Dict[str, MetricInfo]:
    """Information of available modalities and their respective metrics."""
    logger.info(params.model_dump())
    info = get_metrics_metadata(params.data_modalities, params.validation_domains, params.metric_categorys)

    if len(info) < 1:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "msg": "No metrics found for the provided parameters.",
                "inputs": params.model_dump(),
                "type": "error",
            },
        )
    return info


@metrics_router.get(
    "/{data_modality}",
    response_model=Union[DatasetEvalResponse, InstanceEvalResponse],
    responses={status.HTTP_501_NOT_IMPLEMENTED: {"model": Message}, status.HTTP_400_BAD_REQUEST: {"model": Message}},
)
def dataset_eval(
    data_modality: DataModalities,
    params: DatasetParams = Depends(),
) -> Union[DatasetEvalResponse, InstanceEvalResponse]:
    """Compute metrics for a given data modality."""

    try:
        group_functions = select_metric_functions(
            data_modality,
            params.validation_domain,
            params.reference_type,
            params.evaluation_level,
            params.metric_category,
            params.metric_group,
        )

        if len(group_functions) < 1 or any(len(funcs) < 1 for funcs in group_functions.values()):
            missing_groups = {key: len(val) for key, val in group_functions.items() if len(val) < 1}
            return JSONResponse(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                content={
                    "msg": "No implemented metrics for the provided parameters.",
                    "inputs": params.model_dump(),
                    "info": {"missing_groups": list(missing_groups.keys()) if len(missing_groups) > 0 else None},
                    "type": "error",
                },
            )

        data_input_layer = prepare_input_layer(
            data_modality,
            params.validation_domain,
            params.reference_type,
            params.annotation_file,
        )

        for group, funcs in group_functions.items():
            logger.info(f"{group.split('.')[-1].title()} Metrics: {', '.join([func.__name__ for func in funcs])}")

        manager = ComputationManager(
            group_functions,
            data_input_layer,
            data_modality,
            params.validation_domain,
            n_workers=_compute_nworkers,
        )

        results = manager.compute_metrics(ml_models)

        return create_output_structure(
            results,
            schema="v1",
            instance_ids=data_input_layer.instance_ids if hasattr(data_input_layer, "instance_ids") else None,
            n_samples=len(data_input_layer),
        )

    except AssertionError as e:
        logger.error(e)
        traceback.print_exc()
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"msg": str(e), "inputs": params.model_dump(), "type": "error"},
        )


@metrics_router.get(
    "/{data_modality}/specific-metrics",
    response_model=Union[DatasetEvalResponse, InstanceEvalResponse],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": Message},
        status.HTTP_400_BAD_REQUEST: {"model": Message},
        status.HTTP_501_NOT_IMPLEMENTED: {"model": Message},
    },
)
def specific_metrics(
    data_modality: DataModalities,
    params: SpecificFunctionParams = Depends(),
) -> Union[DatasetEvalResponse, InstanceEvalResponse]:
    """Compute the metrics from list of metric names.

    Must provide the metric names to be computed as well as the
    validation and reference types.
    """

    try:
        group_functions = select_specific_metric_functions(
            params.metric_names,
            data_modality,
            params.validation_domain,
            params.reference_type,
        )

        selected = {func.__name__ for funcs in group_functions.values() for func in funcs}
        if len(selected) != len(params.metric_names):
            missing = set(params.metric_names) - selected
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "msg": "Some metrics were not found for the provided parameters.",
                    "inputs": params.model_dump(),
                    "type": "error",
                    "info": {"missing_metrics": list(missing)},
                },
            )

        data_input_layer = prepare_input_layer(
            data_modality,
            params.validation_domain,
            params.reference_type,
            annotation_file=params.annotation_file,
        )

        for group, funcs in group_functions.items():
            logger.info(f"{group.split('.')[-1].title()} Metrics: {', '.join([func.__name__ for func in funcs])}")

        manager = ComputationManager(
            group_functions,
            data_input_layer,
            data_modality,
            params.validation_domain,
            n_workers=_compute_nworkers,
        )

        results = manager.compute_metrics(ml_models)

        return create_output_structure(
            results,
            schema="v1",
            instance_ids=data_input_layer.instance_ids if hasattr(data_input_layer, "instance_ids") else None,
            n_samples=len(data_input_layer),
        )
    except AssertionError as e:
        logger.error(e)
        traceback.print_exc()
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"msg": str(e), "inputs": params.model_dump(), "type": "error"},
        )


app.include_router(metrics_router)

if __name__ == "__main__":
    import uvicorn

    port = int(getenv("API_PORT", 8000))
    domain = "0.0.0.0"
    workers = int(getenv("GUNICORN_WORKERS", 1))

    uvicorn.run(
        "run_api:app",
        host=domain,
        port=port,
        reload=True,
        workers=workers,
    )
