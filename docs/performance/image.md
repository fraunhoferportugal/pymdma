# Image Metric Performance

In this section, we measure the performance of the image metrics. All computations are performed on the same instance with a constant hardware configuration. No CUDA acceleration is used.

## Input Validation

To validate the performance of the image metrics, we measured the mean batch time for a large image dataset. A total of 5000 images with a resolution of 512x512 were randomly generated for the purposes of this experiment. We split the computation into batches of 50 images at a time. We measured the mean batch time, total time, and peak memory usage for each metric.

| Metric | Mean Batch Time (s) | Total Time (s) | Mean Peak Memory (GiB) | Max Peak Memory (GiB) |
|:-------------------|----------------------:|-----------------:|-------------------------:|------------------------:|
| BRISQUE | 2.02 ± 1.17 | 201.91 | 1.14 ± 0.00 | 1.15 |
| Brightness | 0.27 ± 0.01 | 27.25 | 1.11 ± 0.00 | 1.11 |
| CLIPIQA | 13.65 ± 0.53 | 1364.88 | 4.03 ± 0.03 | 4.05 |
| Colorfulness | 0.20 ± 0.01 | 20.48 | 1.12 ± 0.00 | 1.12 |
| DOM | 8.51 ± 0.57 | 850.74 | 1.09 ± 0.00 | 1.09 |
| EME | 0.19 ± 0.00 | 19.39 | 1.12 ± 0.00 | 1.12 |
| ExposureBrightness | 0.19 ± 0.00 | 18.51 | 1.12 ± 0.00 | 1.12 |
| MSSSIM | 3.74 ± 0.88 | 374.33 | 3.09 ± 0.09 | 3.27 |
| PSNR | 0.38 ± 0.01 | 37.79 | 0.64 ± 0.00 | 0.64 |
| SSIM | 2.21 ± 0.46 | 220.83 | 3.10 ± 0.09 | 3.26 |
| Tenengrad | 0.21 ± 0.00 | 20.82 | 1.12 ± 0.00 | 1.12 |
| TenengradRelative | 0.92 ± 0.01 | 91.55 | 1.12 ± 0.00 | 1.12 |

## Synthesis Validation

In these experiments we measure the performance of the image metrics on synthetic data. We generated random reference and target embeddings of size 5000x2048 to mimic the embedding size of the Inception V3 model on two sets of 50000 images. We measure the total execution time of the metrics executed individually on the random embeddings, as well as the peak memory usage of the metrics.

| Metric | Total Time (s) | Peak Memory (GiB) |
|:----------------------------|-----------------:|--------------------:|
| Authenticity | 141.82 | 24.25 |
| Coverage | 153.4 | 24.27 |
| Density | 153.17 | 24.19 |
| FrechetDistance | 17.15 | 3.38 |
| GIQA | 42.84 | 2.7 |
| ImprovedPrecision | 140.46 | 24.63 |
| ImprovedRecall | 145.24 | 24.56 |
| MultiScaleIntrinsicDistance | 504.51 | 2.55 |
| PrecisionRecallDistribution | 83.23 | 4.45 |
