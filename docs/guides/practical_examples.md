# Practical Examples

If you are looking for practical examples of how to use the package in your own code, you can find applications for both evaluation domains across all supported modalities.

We provide usage examples in the form of Jupyter notebooks, which demonstrate how to load data, configure evaluation settings, and interpret the output of various metrics. These notebooks are designed to be self-contained and easy to follow, with explanations that guide you through the steps of using the package effectively.

You can find all example notebooks in the [notebooks](https://github.com/fraunhoferportugal/pymdma/tree/main/notebooks) folder of the repository.

Whether you're working with images, structured tables, or sequential data, these notebooks offer practical insights and best practices for integrating the package into your data science workflow.

## Image Modality

[image_examples.ipynb](https://github.com/fraunhoferportugal/pymdma/blob/main/notebooks/image_examples.ipynb) demonstrates how to perform input validation on a subsample of ImageNet images and their distorted versions using pyMDMA. This notebook shows how different distortions affect image quality and how these changes are reflected in the results of various metrics.

In the synthetic validation section, we provide examples of how to process and evaluate a synthetic dataset. We leverage [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images), a synthetic version of the popular [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. This section illustrates how to evaluate both real and synthetic image sets using the available metrics. Additionally, we demonstrate how to identify the best and worst samples according to each metric, based on their scores.

______________________________________________________________________

## Tabular Modality

[tabular_examples.ipynb](https://github.com/fraunhoferportugal/pymdma/blob/main/notebooks/tabular_examples.ipynb) showcases the evaluation of tabular data using available metrics on a variety of synthetic tabular datasets. It includes examples with different configurations (e.g., varying numbers of features, redundant features, etc.) and shows how these properties influence the metric results.

In the synthetic validation section, we reuse the same datasets from the input validation section to compare real and synthetic tabular data. This section demonstrates how to process both sets and apply the available metrics to evaluate data quality and similarity.

______________________________________________________________________

## Time-Series Modality

[time_series_examples.ipynb](https://github.com/fraunhoferportugal/pymdma/blob/main/notebooks/time_series_examples.ipynb) illustrates the application of time-series metrics on both real and synthetic ECG traces. It includes utility functions for reading, preprocessing, and evaluating time-series data.

In the synthetic evaluation section, we demonstrate how to extract features from signals using the [TSFEL](https://tsfel.readthedocs.io/en/latest/index.html) library and how to use these features to compare real and synthetic datasets. We also show how to identify realistic and unrealistic samples based on metric scores, and provide insights into metric behavior across different time-series characteristics.
