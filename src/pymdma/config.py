import os
import platform
from os import environ, getenv
from os.path import dirname, join
from pathlib import Path

import dotenv

# Project root
project_dir = os.getcwd()

# Load the environment variables from the `.env` file, overriding any system environment variables
env_path = join(project_dir, ".env")
dotenv.load_dotenv(env_path, override=True)

# Load secrets from the `.secrets` file, overriding any system environment variables
secrets_path = join(project_dir, ".secrets")
dotenv.load_dotenv(secrets_path, override=True)

# Some common paths
_reports_dir = Path(str(getenv("DIR_REPORTS")))
report_dir = join(project_dir, _reports_dir)

_figures_dir = Path(str(getenv("DIR_FIGURES")))
figures_dir = join(project_dir, _figures_dir)

_models_dir = Path(str(getenv("DIR_MODELS")))
models_dir = join(project_dir, _models_dir)

_notebook_dir = Path(str(getenv("DIR_NOTEBOOKS")))
notebook_dir = join(project_dir, _notebook_dir)

_data_dir = Path(str(getenv("DIR_DATA")))
data_dir = join(project_dir, _data_dir)

_data_raw_dir = Path(str(getenv("DIR_DATA_RAW")))
data_raw_dir = join(project_dir, _data_raw_dir)

_data_interim_dir = Path(str(getenv("DIR_DATA_INTERIM")))
data_interim_dir = join(project_dir, _data_interim_dir)

_data_processed_dir = Path(str(getenv("DIR_DATA_PROCESSED")))
data_processed_dir = join(project_dir, _data_processed_dir)

_data_test_dir = Path(str(getenv("DIR_DATA_TEST")))
data_test_dir = join(project_dir, _data_test_dir)

_package_dir = Path(str(getenv("PACKAGE_NAME")))
package_base_dir = join(project_dir, _package_dir)

# CUDA Enable
ENABLE_CUDA = True
if not ENABLE_CUDA:
    environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environ["CUDA_VISIBLE_DEVICES"] = "-1"
    environ["USE_CPU"] = "1"

# Hydra
environ["HYDRA_FULL_ERROR"] = "1"

# log to mlflow
LOG_TO_MLFLOW = False

_IS_WINDOWS = platform.system() == "Windows"

# EOF
