# pyMDMA - Multimodal Data Metrics for Auditing real and synthetic datasets

[![Python](https://img.shields.io/badge/python-3.9+-informational.svg)](<>)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=black)](https://pycqa.github.io/isort)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://mkdocstrings.github.io)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![pytest](https://img.shields.io/badge/pytest-enabled-brightgreen)](https://github.com/pytest-dev/pytest)
[![conventional-commits](https://img.shields.io/badge/conventional%20commits-1.0.0-yellow)](https://github.com/commitizen-tools/commitizen)
[![Read The Docs](https://readthedocs.org/projects/pymdma/badge/?version=latest)](https://pymdma.readthedocs.io/en/latest/installation/)

<!-- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fraunhoferportugal/pymdma.git/main?labpath=notebooks%2Fimage_examples.ipynb) -->

Data auditing is essential for ensuring the reliability of machine learning models by maintaining the integrity of the datasets upon which these models rely. As synthetic data use increases to address data scarcity and privacy concerns, there is a growing demand for a robust auditing framework.

Existing repositories often lack comprehensive coverage across various modalities or validation types. This work introduces a dedicated library for data auditing, presenting a comprehensive suite of metrics designed for evaluating synthetic data. Additionally, it extends its focus to the quality assessment of input data, whether synthetic or real, across time series, tabular, and image modalities.

This library aims to serve as a unified and accessible resource for researchers, practitioners, and developers, enabling them to assess the quality and utility of their datasets. This initiative encourages collaborative contributions by open-sourcing the associated code, fostering a community-driven approach to advancing data auditing practices. This work is intended for publication in an open-source journal to facilitate widespread dissemination, adoption, and impact tracking within the scientific and technical community.

For more information check out the official documentation [here](https://pymdma.readthedocs.io/en/latest/).

## Prerequisites

You will need:

- `pip`
- `python` (see `pyproject.toml` for full version)
- `anaconda` or similar (recommended)
- `Git` (developers)
- `Make` and `poetry` (developers)
- load environment variables from `.env` (developers)

## 1. Installing

You can either install this package via pip (if you want access to individual modules) or clone the repository (if you wish to contribute to the project or change the code in any way). Currently, the package supports the following modalities: `image`, `tabular`, and `time_series`.

You should install the package in a virtual environment to avoid conflicts with system packages. Please consult the official [documentation](https://docs.python.org/3/library/venv.html) for developing with virtual environments.

### 1.1 Installing via pip (recommended)

Before running any commands, make sure you have the latest versions of `pip` and `setuptools` installed.
The package can be installed with the following command:

```bash
pip install pymdma
```

Depending on the data modality you want to use, you may need to install additional dependencies. The following commands will install the dependencies for each modality:

```bash
pip install pymdma[image] # image dependencies
pip install pymdma[tabular] # tabular dependencies
pip install pymdma[time_series] # time series dependencies
pip install pymdma[all] # dependencies for all modalities
```

Choose the one(s) that best suits your needs.

> **Note:** for a minimal installation without CUDA support, you can install the package without the `cuda` dependencies. This can be done by forcing pip to install torch from the CPU index with the `--find-url=https://download.pytorch.org/whl/cpu/torch_stable.html` command. **You will not have access to the GPU-accelerated features.**

### 1.1 Installing from source

You can install the package from source with the following command:

```bash
pip install "pymdma @ git+https://github.com/fraunhoferportugal/pymdma.git"
```

Depending on the data modality you want to use, you may need to install additional dependencies. The following commands will install the dependencies for each modality:

```bash
pip install "pymdma[image] @ git+https://github.com/fraunhoferportugal/pymdma.git" # image dependencies
pip install "pymdma[tabular] @ git+https://github.com/fraunhoferportugal/pymdma.git" # tabular dependencies
pip install "pymdma[tabular] @ git+https://github.com/fraunhoferportugal/pymdma.git" # time series dependencies
```

For a minimal installation, you can install the package without CUDA support by forcing pip to install torch from the CPU index with the `--find-url` command.

## 2. Execution Examples

The package provides a CLI interface for automatically evaluating folder datasets. You can also import the metrics for a specific modality and use them in your code.
Before running any commands make sure the package was correctly installed.

### 2.1. Importing Modality Metrics

You can import the metrics for a specific modality and use them in your code. The following example shows how to import an image metric and use it to evaluate input images in terms of sharpness. Note that this metric only returns the sharpness value for each image (i.e. the instance- level value). The dataset-level value is none.

```python
from pymdma.image.measures.input_val import Tenengrad
import numpy as np

images = np.random.rand(10, 224, 224, 3)  # 10 random RGB images of size 224x224

tenengrad = Tenengrad()  # sharpness metric
sharpness = tenengrad.compute(images)  # compute on RGB images

# get the instance level value (dataset level is None)
_dataset_level, instance_level = sharpness.value
```

For evaluating synthetic datasets, you also have access to the synthetic metrics. The following example shows the steps necessary to process and evaluate a synthetic dataset in terms of the feature metrics. We load one of the available feature extractors, extract the features from the images and then compute the precision and recall metrics for the synthetic dataset in relation to the reference dataset.

```python
from pymdma.image.models.features import ExtractorFactory

test_images_ref = Path("./data/test/image/synthesis_val/reference")  # real images
test_images_synth = Path("./data/test/image/sythesis_val/dataset")  # synthetic images

# Get image filenames
images_ref = list(test_images_ref.glob("*.jpg"))
images_synth = list(test_images_synth.glob("*.jpg"))

# Extract features from images
extractor = ExtractorFactory.model_from_name(name="vit_b_32")
ref_features = extractor.extract_features_from_files(images_ref)
synth_features = extractor.extract_features_from_files(images_synth)
```

Now you can calculate the Improved Precision and Recall of the synthetic dataset in relation to the reference dataset.

```python
from pymdma.image.measures.synthesis_val import ImprovedPrecision, ImprovedRecall

ip = ImprovedPrecision()  # Improved Precision metric
ir = ImprovedRecall()  # Improved Recall metric

# Compute the metrics
ip_result = ip.compute(ref_features, synth_features)
ir_result = ir.compute(ref_features, synth_features)

# Get the dataset and instance level values
precision_dataset, precision_instance = ip_result.value
recall_dataset, recall_instance = ir_result.value

# Print the results
print(f"Precision: {precision_dataset:.2f} | Recall: {recall_dataset:.2f}")
print(f"Precision: {precision_instance} | Recall: {recall_instance}")
```

You can find more examples of execution in the [notebooks](notebooks) folder.

### 2.2. CLI Execution

To evaluate a dataset, you can use the CLI interface. The following command will list the available commands:

```bash
pymdma --help # list available commands
```

Following is an example of executing the evaluation of a synthetic dataset with regard to a reference dataset:

```bash
pymdma --modality image \
    --validation_domain synth \
    --reference_type dataset \
    --evaluation_level dataset \
    --reference_data data/test/image/synthesis_val/reference \
    --target_data data/test/image/synthesis_val/dataset \
    --batch_size 3 \
    --metric_category feature \
    --output_dir reports/image_metrics/
```

This will evaluate the synthetic dataset in the `data/test/image/synthesis_val/dataset` with regard to the reference dataset in `data/test/image/synthesis_val/reference`. The evaluation will be done at the dataset level and the report will be saved in the `reports/image_metrics/` folder in JSON format. Only feature metrics will be computed for this evaluation.

## Documentation

Full documentation is available here: [`docs/`](docs).

<!-- ## Dev

See the [Developer](docs/DEVELOPER.md) guidelines for more information. -->

## Contributing

Contributions of any kind are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details and
the process for submitting pull requests to us.

If you change the code in any way, please follow the developer guidelines in [DEVELOPER.md](DEVELOPER.md).

## Changelog

See the [Changelog](CHANGELOG.md) for more information.

## Security

Thank you for improving the security of the project, please see the [Security Policy](docs/SECURITY.md)
for more information.

## License

This project is licensed under the terms of the `LGPL-3.0` license.
See [LICENSE](LICENSE) for more details.

## Citation

If you publish work that uses pyMDMA, please cite pyMDMA as follows:

```bibtex
@misc{pymdma,
  title = {{pyMDMA}: Multimodal Data Metrics for Auditing real and synthetic datasets},
  url = {https://github.com/fraunhoferportugal/pymdma},
  author = {Fraunhofer AICOS},
  license = {LGPL-3.0},
  year = {2024},
}
```

## Acknowledgments

This work was funded by AISym4Med project number 101095387, supported by the European Health and Digital Executive Agency (HADEA), granting authority under the powers delegated by the European Commission. More information on this project can be found [here](https://aisym4med.eu/).

This work was supported by European funds through the Recovery and Resilience Plan, project ”Center for Responsible AI”, project number C645008882-00000055. Learn more about this project [here](https://centerforresponsible.ai/).
