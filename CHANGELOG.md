# Changelog

All notable changes to this project will be documented in this file.
This format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.2] - 2024-10-28
### Added
 - CLIP device specification
 - CLI compute arguments in reports dir
 - Preprocessing transform in image datasets when using the API

### Fixed
 - Moved `sklearn` and `gudhi` dependencies to the main dependency tree

### Changed
 - Default image feature extractor is now `vit_b_32`
 - Confusing synthetic and input metric goals where aggregated to `quality`, `privacy`, `annotation` and `utility` categories



## [0.1.1] - 2024-10-24
### Fixed
 - Fixed project configuration conflict between setup.py and pyproject.toml by reverting to poetry as main build engine
 - Updated dependencies in `pyproject.toml` and ignoring versions in requirements folder
 - Fixed error in validation type for the cli
 - Fixed example `tabular` and `time-series` scripts to use cli application

### Removed
 - Removed `setup.py` file as it is deprecated in favor of `pyproject.toml`
 - Removed `cookiecutter` template files as they are not needed for the project

### Changed
 - Changed default image extractor model from `dino_vits8` to `vit_b_32`


## [0.1.0] - 2024-10-16
 - Bumped version to 0.1.0 to indicate the first stable release of the package


## [0.0.41] - 2024-08-31
### Added
 - Documentation for image, tabular and time-series class metrics
 - Tabular and time-series unit tests for direct import from the package
 - Dummy notebook with examples of quality evaluation on the imagenette dataset
 - Mkdocs documentation for the package (only available locally and for metric classes)
   - Each modality has a seciton in the documentation site
   - Metrics are organized as input_val and synthesis_val groups
   - Groups may be divided into other subgroups depending on metrics' specificities
- Added distance metric parameter for the PRDC metrics


### Fixed
 - Entire code structure moved to package level under src/
   - Allows for easire import of metrics and models and correct package builds
 - Package submodules defined in upper __init__.py files to allow for direct import of metrics
 - Updated `mkdocs` from `1.5.3` to `1.6.1` to fix a bug in the previous version
 - Updated `mkdocs-material` from `9.4.14` to `9.5.34` to fix a bug in the previous version
 - Updated `mkdocstrings` from `0.24.0` to `0.26.0` to fix a bug in the previous version


## [0.0.40] - 2024-07-31

### Added
 - Standard version of `main.py` and documentation on how to run locally
 - Support for specific feature extractors in the `main.py` script
 - Added key -> value/array dtypes for metric results
 - API data access is now controlled by the `API_DATA_ROOT` environment variable in .env
 - Modality batch sizes are now controlled by the `<MODALITY>_BATCH_SIZE` environment variables in .env
 - Reorganized metrics into metric group categories to account for shared requirements. A diagram explaining the new metric structure can be found [here](docs/hierarchy_diagram.png)
    - input metrics are grouped under `quality`, `privacy` or `annotation` categories
    - Synthesized metrics are grouped under `feature` or `data` categories
 - Older and confusing `metric_subgoals` were moved to the `metric_goal` field
    - `metric_goal` field is an optional subcategory of the `metric_group` field
 - Integrated image COCO annotation metrics into the API
    - Annotation files must be present under the input_val/annotations/ directory
    - Annotation files must be in COCO format and the filename given as a parameter in the API
 - API unit testing for every modality with FastAPI testing utilities
 - Documentation of shared metrics and image metrics
 - Added annotation_file field to the API for image datasets


### Fixed
 - Using resize of 224 in DINO-ViT for image datasets
 - Updated `pycaret` from `3.1.0` to `3.3.1` due to a bug in the previous version
 - Removed redundant hooks in the Makefile
 - Text datasets are now kept under text/input_val/dataset/ for consistency with the other modalities
 - Input layers now extend the InputLayer abastract class for consistency
 - Renamed input variables in input layers from `data_src1`, `data_src2` to `reference` and `target`
 - Refactored code struture from properties-based structure to a more modular class-based structure
    - Metrics are now classes that inherit from a base `Metric` class
    - Each metric has a `compute` method that returns the metric value
    - Older property attributes are now class attributes of the specific metric class
    - Introduced the `MetricResult` class to enforce and validate return types of the `compute` method
    - Input layers provide batched samples to the metrics via the `batched_samples` property
    - Embedder models are now passed to the `get_embeddings` method of the input layers (allows for more flexibility)
    - Moved common API and CLI code to the `common`package to avoid code repetition
    - Added the `ComputationManager` class to manage the computation of metrics. Allows for the batch computation and merger of metric results and the computation of metrics in parallel. Also keeps track of a shared state of the metrics and their results.
 - Removed FhP git dependencies from the pyproject.toml file (not needed)
 - Dockerfile now uses only the `requirements-prod.txt` file for production builds
 - Dockerfile pip install with torch-cpu source for lighter builds
 - Removed unused environment variables from the .env file
 - Standard error messages being returned in the API for invalid requests


### Changed
 - `evaluation_level` field in the API is now optional
 - `metric_goal` field in the API is now optional
 - `metric_group` field in the API is mandatory and indicates the group of the metric
 - `annotation_file` optional field for the API to specify the annotation file for image datasets
 - `metric_subgoals` field in the API was removed and replaced by the `metric_goal` field
 - `metric_plots` field in the returned JSON of the API was removed
    - Each metric now has a `plot_params` field that contains the plot parameters
 - Each metric now has an optional `stats` field that contains additional statistical information of the metric results
    - Previous iterations returned these statistics in separate metric results
 - The response body now contains an `errors` field to indicate any errors caught during the metric computation
    - This is only used in known errors/warnings that are known to occur in a specific metric (e.g. a metric that is supposed to be between 0 and 1 but is not due to numerical instability). Other API errors are still returned as HTTP status codes.
 - Renamed metric result `type` field to `dtype` to avoid conflicts with Python's `type` function
 - Renamed the `dtype` in the metric result to `subtype`
