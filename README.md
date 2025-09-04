# pyMDMA - Multimodal Data Metrics for Auditing real and synthetic datasets
## ECML PKDD 2025 Tutorial 

pyMDMA is an open-source Python library that provides metrics for evaluating both real and synthetic datasets across **image**, **tabular**, and **time-series** data modalities. It was developed to address gaps in existing evaluation frameworks which often lack metrics for specific data modalities, do not include certain state-of-the-art metrics, and do not provide a comprehensive categorization.

pyMDMA provides a standard code base throughout all modalities, to make the integration and usage of metrics easier. The library is organized according to a new proposed taxonomy, that categorizes more clearly and intuitively the existing methods according to specific auditing goals for input (e.g., perceptual quality, uniqueness, correlation, etc.) and synthetic data (e.g. fidelity, diversity, authenticity, etc.). In particular, each metric class is organized based on the data modality (image, tabular, and time-series), validation domain (input and synthesis), metric category (data-based, annotation-based, and feature-based), and group (quality, privacy, validity, utility). We provide additional statistics for each metric result to help the user reach more concrete conclusions about the audited data.

In this tutorial, we will begin with a brief overview of the pyMDMA library, followed by three hands-on sessions, each focusing on a different data modality: **image**, **time-series**, and **tabular**. In each session, we will explore practical use cases in depth, demonstrating how to leverage this framework effectively. Specifically, we will cover data preparation and data quality evaluation for both real and synthetic datasets.

**Keywords:** Data Auditing; Data Quality Assessment; Real Data; Synthetic Data; Image; Time-Series; Tabular

You can find the source code on [GitHub](https://github.com/fraunhoferportugal/pymdma) and the documentation on [ReadTheDocs](https://pymdma.readthedocs.io/en/latest/).

## Target Audience

This tutorial is designed for a diverse audience across multiple domains, including professionals and researchers working with **image**, **time-series**, and **tabular** data. It is relevant to those handling **real-world datasets** as well as individuals focused on **auditing and analyzing synthetic data**. By addressing data quality assessment across different modalities, this tutorial appeals to data scientists, auditors, AI researchers, and practitioners seeking robust multimodal evaluation methodologies to ensure the reliability of real data or assess the quality of generated datasets.

## Tutorial Outline

The tutorial will take about 4 hours, including 30 min break. The first part of the tutorial will be a presentation of the library and its metrics. The remaining time will be dedicated to hands-on sessions focused on each data modality. A detailed schedule is provided bellow.

### pyMDMA presentation (30 mins) – Luís Rosado

- Introduction
- Target modalities
- Metric taxonomy description
- Available metrics, installation, and contribution

### Image Tutorial (60 mins) – Ivo Façoco

- Dataset presentation
  - Public RGB dataset
- Input Validation
  - Extraction of image quality metrics
  - Distribution analysis of extracted metrics
- Synthetic Validation
  - Synthetic dataset explanation (model used, number of instances and type of conditioning)
  - Feature extraction with pre-trained models
  - Evaluation of fidelity and diversity concepts
  - Sample selection through quality-based ranking. Comparison of best/worst generated examples via metric outputs.

### Time-Series Tutorial (60 mins) – Maria Russo

- Dataset Presentation
  - Overview of the ECG dataset and its characteristics
- Input Validation
  - Extraction of signal quality metrics
  - Distribution analysis of extracted metrics
- Synthetic Validation
  - Explanation of the synthetic dataset
  - Feature extraction using the Time Series Feature Extraction Library (TSFEL)
  - Evaluation of fidelity and diversity concepts
  - Selection of synthetic samples using metric outputs.

### Short Break (30 mins)


### Tabular Tutorial (60 mins) – Pedro Matias

- Dataset Presentation
  - Dataset Loading
    - High-quality public tabular datasets
    - Low-quality public tabular datasets
  - Data Preparation
    - Attribute type detection, encoding, and scaling;
    - Visualization through 2D-embeddings.
- Input Validation
  - Extraction of tabular quality metrics;
  - Dataset selection through quality-based global ranking.
- Synthetic Validation
  - Synthetic Datasets
    - Description of generative models (traditional vs. deep learning);
    - Visualization of real vs. synthetic using 2D-embeddings.
  - Evaluation of fidelity and diversity concepts;

## Presenters

### Luís Rosado

Senior Scientist (Fraunhofer Portugal AICOS)\
**email:** luis.rosado@aicos.fraunhofer.pt

### Ivo Façoco

Scientist (Fraunhofer Portugal AICOS)\
**email:** ivo.facoco@aicos.fraunhofer.pt

### Maria Russo
Scientist (Fraunhofer Portugal AICOS)\
**email:** maria.russo@aicos.fraunhofer.pt

### Pedro Matias

Scientist (Fraunhofer Portugal AICOS)\
**email:** pedro.matias@aicos.fraunhofer.pt

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
