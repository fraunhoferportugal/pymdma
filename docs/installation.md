# Installation

PyPI is currently unavailable. To install the package, you can install it directly from the git repository. To do so, run the following command:

```bash
$ pip install "pymdma @ git+https://github.com/fraunhoferportugal/pymdma.git"
```

<!-- It is recommended to install the package in a virtual environment. To install the package, run the following command:

```bash
$ pip install pymdma
``` -->

Depending on the data modality you are working with, you may need to install additional dependencies. We have three groups of denpendencies: `image`, `tabular` and `time_series`. As an example, to work with image data, you will need to run the following command:

```bash
$ pip install "pymdma[image] @ git+https://github.com/fraunhoferportugal/pymdma.git"
```

You can also install multiple modalities by passing the desired modalities as a comma-separated list. For example, to install both image and tabular modalities, you can run the following command:

```bash
$ pip install "pymdma[image,tabular] @ git+https://github.com/fraunhoferportugal/pymdma.git"
```

## Minimal Version (CPU)

For a minimal installation (without GPU support), you can install the package with CPU version of torch, which will skip the installation of CUDA dependencies. To do so, run the following command:

```bash
$ pip install pymdma[...] @ git+https://github.com/fraunhoferportugal/pymdma.git --find-url=https://download.pytorch.org/whl/cpu/torch_stable.html
```
