import warnings

import numpy as np
import pytest

from pymdma.constants import OutputsTypes
from pymdma.time_series.measures.input_val import quality as input_metrics
from pymdma.time_series.measures.synthesis_val.feature import _shared as synth_shared_metrics
from pymdma.time_series.measures.synthesis_val.feature import distance as synth_distance_metrics

# ###################################################################################################
# ################################ INPUT VALIDATION METRICS TESTS ###################################
# ###################################################################################################


@pytest.mark.parametrize(
    "input_metric_cls",
    [
        input_metrics.Uniqueness,
        input_metrics.SNR,
    ],
)
def test_batch_input_metrics(ts_dataset, input_metric_cls):
    metric = input_metric_cls()

    result = metric.compute(ts_dataset)
    _, instance_level = result.value

    assert isinstance(instance_level, list), "Instance level is not a list"
    assert len(instance_level) == len(ts_dataset), "Instance level length does not match input length"


def test_snr_order(sample_distribution):
    """Test the signal to noise ratio order."""
    signals_low_snr = sample_distribution((64, 1000, 12), sigma=3, mu=1)
    signals_middle = sample_distribution((64, 1000, 12), sigma=2, mu=1)
    signals_high_snr = sample_distribution((64, 1000, 12), sigma=1, mu=1)

    result_low = np.mean(input_metrics.SNR().compute(signals_low_snr).instance_level.value)
    result_middle = np.mean(input_metrics.SNR().compute(signals_middle).instance_level.value)
    result_high = np.mean(input_metrics.SNR().compute(signals_high_snr).instance_level.value)

    assert result_low < result_middle < result_high, "SNR: unexpected order"


def test_snr_value(sample_distribution):
    """Test the signal to noise ratio values."""
    mu = 1
    sigma = 3
    signals_low_snr = sample_distribution((200, 1000, 12), sigma=sigma, mu=mu)
    result_low = np.mean(input_metrics.SNR().compute(signals_low_snr).instance_level.value)

    assert (
        mu / sigma - APROXIMATION_TOLERANCE < result_low < mu / sigma + APROXIMATION_TOLERANCE
    ), f"SNR: unexpected value of {result_low}. Extepected:{mu/sigma}"


def test_uniqueness_order(sample_distribution):
    """Test the uniqueness order."""
    batch_size = 12
    num_batches = 200
    sig_length = 1000

    # Function to generate a sine wave with flat tops
    def generate_flat_top_sine_wave(frequency, amplitude, sampling_rate, signal_length, flat_top_threshold=0.5):
        t = np.arange(0, signal_length / sampling_rate, 1 / sampling_rate)
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
        sine_wave[np.abs(sine_wave) > flat_top_threshold * amplitude] = flat_top_threshold * amplitude
        return sine_wave.T

    # Generate sine waves with flat tops
    flat_top_sin = np.array(
        [[generate_flat_top_sine_wave(30, 4, 100, 1000) for _ in range(batch_size)] for _ in range(num_batches)],
    ).reshape(num_batches, sig_length, batch_size)

    constant_signals = np.zeros((num_batches, sig_length, batch_size))

    normal_dist_signals = sample_distribution((num_batches, sig_length, batch_size))
    result_high = np.mean(input_metrics.Uniqueness().compute(constant_signals).instance_level.value)
    result_middle = np.mean(input_metrics.Uniqueness().compute(flat_top_sin).instance_level.value)
    result_low = np.mean(input_metrics.Uniqueness().compute(normal_dist_signals).instance_level.value)

    assert result_low < result_middle < result_high, "Uniqueness: unexpected order"


def test_uniqueness_values(sample_distribution):
    """Test the uniqueness limit values."""
    batch_size = 12
    num_batches = 200
    sig_length = 1000

    constant_signals = np.zeros((num_batches, sig_length, batch_size))

    normal_dist_signals = sample_distribution((num_batches, sig_length, batch_size))

    result_high = np.mean(input_metrics.Uniqueness().compute(constant_signals).instance_level.value)
    result_low = np.mean(input_metrics.Uniqueness().compute(normal_dist_signals).instance_level.value)

    assert abs(result_high - 1) <= TOLERANCE and abs(result_low - 0) <= TOLERANCE, "Uniqueness: unexpected values"


# ###################################################################################################
# ################################ SYNTHETIC METRICS TESTS #########################################
# ###################################################################################################

TOLERANCE = 1e-4  # numeric tolerance for float comparison
APROXIMATION_TOLERANCE = 1e-2  # numeric tolerance for approximations


# AUXILIARY FUNCTION
def validate_stats(stats_values, stats, expected_stats):
    if stats in stats_values:
        value = stats_values[stats]
        assert (
            expected_stats - TOLERANCE < value < expected_stats + TOLERANCE
        ), f"{stats}: unexpected value of {value} for the same distribution"


@pytest.mark.parametrize(
    "extractor_name",
    [
        "tsfel",
    ],
)
def test_extractor_models(ts_feature_extractor, synth_ts_filenames, extractor_name):
    extractor = ts_feature_extractor(extractor_name)

    features = extractor.extract_features_from_files(
        synth_ts_filenames,
        fs=50,
        dims=["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8", "ch9", "ch10", "ch11", "ch12"],
        batch_size=6,
    )

    assert features.shape[0] == len(synth_ts_filenames), "Feature length does not match input length"
    assert features.shape[1] > 0, "Empty second dimension"

    prec = synth_shared_metrics.ImprovedPrecision()
    result = prec.compute(features, features)
    assert result.dataset_level is not None and result.instance_level is not None, "Eval level is None"
    dataset_level, instance_level = result.value
    assert dataset_level > 0.90, "Dataset level is below threshold"
    assert all(inst == 1 for inst in instance_level), "Same image instance should be precise"


@pytest.mark.parametrize(
    "synth_metric_cls, expected",
    [
        (synth_distance_metrics.WassersteinDistance, 0.0),
        (synth_distance_metrics.MMD, 0.0),
        (synth_distance_metrics.CosineSimilarity, 1.0),
        (synth_shared_metrics.FrechetDistance, 0.0),
        (synth_shared_metrics.Authenticity, 0.0),
        (synth_shared_metrics.ImprovedPrecision, 1.0),
        (synth_shared_metrics.ImprovedRecall, 1.0),
        (synth_shared_metrics.Coverage, 1.0),
        (synth_shared_metrics.PrecisionRecallDistribution, None),
    ],
)
@pytest.mark.parametrize(
    "stats, expected_stats",
    [
        ("dispersion_ratio", 1.0),
        ("distance_ratio", 1.0),
        ("prd_to_max_f_beta", 1.0),
        ("prd_to_max_f_beta_inv", 1.0),
    ],
)
def test_same_distribution(synth_metric_cls, sample_distribution, expected, stats, expected_stats):
    """Tests the limit values of bounded measures when they receive equal
    distributions as input. That is:

    - statistical divergence measures --> 0
    - similarity measures--> 1
    - dispersion ratios --> 1
    - distance ratios --> 1
    - precision-related measures --> 1
    - recall-relatd measures --> 1
    - authenticity-relatd measures --> 0

    Also tests the length of the instance-level metrics
    """
    x_ref = sample_distribution((500, 128))

    metric = synth_metric_cls()
    result = metric.compute(x_ref, x_ref)

    if result.dataset_level.dtype == OutputsTypes.KEY_ARRAY:
        if result.dataset_level.stats:
            required_max_f_beta = {"max_f_beta", "max_f_beta_inv"}

            if required_max_f_beta <= result.dataset_level.stats.keys():
                max_f_beta_value = result.dataset_level.stats["max_f_beta"]
                max_f_beta_inv_vale = result.dataset_level.stats["max_f_beta_inv"]

                stats_values = {
                    "max_f_beta": max_f_beta_value,
                    "max_f_beta_inv": max_f_beta_inv_vale,
                }

                validate_stats(stats_values, stats, expected_stats)

    elif result.dataset_level.dtype == OutputsTypes.NUMERIC:
        value = result.dataset_level.value

        assert (
            expected - TOLERANCE < value < expected + TOLERANCE
        ), f"{synth_metric_cls.__name__}: unexpected value of {value} for the same distribution"

        if result.dataset_level.stats:
            required_ratios = {"dispersion_ratio", "distance_ratio"}

            # Check if both required ratios exist in the stats dictionary
            if required_ratios <= result.dataset_level.stats.keys():
                # Extract the values of interest
                dispersion_ratio_value = result.dataset_level.stats["dispersion_ratio"]
                distance_ratio_value = result.dataset_level.stats["distance_ratio"]

                # Create a mapping from stats names to their values
                stats_values = {
                    "dispersion_ratio": dispersion_ratio_value,
                    "distance_ratio": distance_ratio_value,
                }

                validate_stats(stats_values, stats, expected_stats)

    else:
        warnings.warn(f"Unknown output type: {result['dataset_level']['type']}. Skipping comparison.", stacklevel=2)

    if "instance_level" in result:
        value_inst = result.instance_level.value
        assert len(value_inst) == len(x_ref), "Instance-level values do not match the number of synthetic samples"


@pytest.mark.parametrize(
    "metric_name, order",
    [
        (synth_distance_metrics.WassersteinDistance, "normal"),
        (synth_distance_metrics.MMD, "normal"),
        (synth_distance_metrics.CosineSimilarity, "inv"),
        (synth_shared_metrics.PrecisionRecallDistribution, "equal"),
        (synth_shared_metrics.FrechetDistance, "normal"),
        (synth_shared_metrics.Authenticity, "equal"),
        (synth_shared_metrics.ImprovedPrecision, "equal"),
        (synth_shared_metrics.ImprovedRecall, "equal"),
        (synth_shared_metrics.Density, "equal"),
        (synth_shared_metrics.Coverage, "equal"),
    ],
)
def test_non_intersect_equal_distributions_order(sample_distribution, metric_name, order, show_dist=False):
    """Tests if the measures respect the expected order when they receive
    increasingly separate non-intersecting distributions with the same shape.
    That is:

    - statistical divergence measures --> increase
    - similarity measures --> decrease
    - dispersion ratios --> constant
    - distance ratios --> increase
    - precision-related measures --> constant
    - recall-relatd measures --> constant
    - authenticity-relatd measures --> constant
    """

    x_ref_min = sample_distribution((300, 2), sigma=0.1, mu=-0.5)
    y_synth_min = sample_distribution((300, 2), sigma=0.1, mu=0.5)

    x_ref_middle = sample_distribution((300, 2), sigma=0.1, mu=-1)
    y_synth_middle = sample_distribution((300, 2), sigma=0.1, mu=1)

    x_ref_max = sample_distribution((300, 2), sigma=0.1, mu=-2)
    y_synth_max = sample_distribution((300, 2), sigma=0.1, mu=2)

    metric = metric_name()
    result_min = metric.compute(x_ref_min, y_synth_min)
    result_middle = metric.compute(x_ref_middle, y_synth_middle)
    result_max = metric.compute(x_ref_max, y_synth_max)

    if show_dist and metric_name == metric_name[0]:
        from src.general.visualization.features_visualization import plot_data

        plot_data(x_ref_min, y_synth_min, title="Scatter Plot of min distance data")
        plot_data(x_ref_middle, y_synth_middle, title="Scatter Plot of middle distance data")
        plot_data(x_ref_max, y_synth_max, title="Scatter Plot of max distance data")

    if result_min.dataset_level.dtype == OutputsTypes.NUMERIC:
        value_min = result_min.dataset_level.value
        value_middle = result_middle.dataset_level.value
        value_max = result_max.dataset_level.value

        if order == "normal":
            assert value_min < value_middle < value_max, f"{metric_name}: unexpected order"
        elif order == "inv":
            assert value_max < value_middle < value_min, f"{metric_name}: unexpected order"
        elif order == "equal":
            assert abs(value_max - value_middle) <= TOLERANCE
            assert abs(value_middle - value_min) <= TOLERANCE

    elif result_min.dataset_level.dtype == OutputsTypes.KEY_ARRAY:

        x_values_min = np.unique(result_min.dataset_level.value["precision_values"])
        y_values_min = np.unique(result_min.dataset_level.value["recall_values"])

        x_values_middle = np.unique(result_middle.dataset_level.value["precision_values"])
        y_values_middle = np.unique(result_middle.dataset_level.value["recall_values"])

        x_values_max = np.unique(result_max.dataset_level.value["precision_values"])
        y_values_max = np.unique(result_max.dataset_level.value["recall_values"])

        if order == "equal":
            assert (
                x_values_min == x_values_middle == x_values_max and y_values_min == y_values_middle == y_values_max
            ), f"{metric_name}: All results should be the same and they are not."

    else:
        warnings.warn(
            f"{metric_name} | Unknown output type: {result_min['dataset_level']['type']}. Skipping comparison.",
            stacklevel=2,
        )


@pytest.mark.parametrize(
    "metric_name, expected",
    [
        (synth_shared_metrics.PrecisionRecallDistribution, 0.0),
        (synth_shared_metrics.Authenticity, 1.0),
        (synth_shared_metrics.ImprovedPrecision, 0.0),
        (synth_shared_metrics.ImprovedRecall, 0.0),
        (synth_shared_metrics.Density, 0.0),
        (synth_shared_metrics.Coverage, 0.0),
        (synth_distance_metrics.WassersteinDistance, None),
        (synth_distance_metrics.MMD, None),
        (synth_shared_metrics.FrechetDistance, None),
    ],
)
@pytest.mark.parametrize(
    "stats, expected_stats",
    [
        ("dispersion_ratio", 1.0),
        ("prd_to_max_f_beta", 0.0),
        ("prd_to_max_f_beta_inv", 0.0),
    ],
)
def test_non_intersect_equal_distributions_values(
    metric_name,
    sample_distribution,
    expected,
    stats,
    expected_stats,
    show_dist=False,
):
    """Tests the limit values of bounded measures when they receive non-
    intersecting distributions with the same shape. That is:

    - dispersion ratios --> 1
    - precision-related measures --> 0
    - recall-relatd measures --> 0
    - authenticity-relatd measures --> 1
    """
    x_ref = sample_distribution((300, 2), sigma=0.1, mu=-0.5)
    y_synth = sample_distribution((300, 2), sigma=0.1, mu=0.5)

    if show_dist and metric_name == synth_distance_metrics.WassersteinDistance:
        from src.general.visualization.features_visualization import plot_data

        plot_data(x_ref, y_synth, title="Scatter Plot of min distance data")

    metric = metric_name()
    result = metric.compute(x_ref, y_synth)

    if result.dataset_level.dtype == OutputsTypes.NUMERIC:
        if expected is not None:
            value = result.dataset_level.value

            assert (
                abs(value - expected) <= TOLERANCE
            ), f"{metric_name}: unexpected value of {value} for the non intersect distribution."

        if result.dataset_level.stats:
            required_ratios = {"dispersion_ratio", "distance_ratio"}

            if required_ratios <= result.dataset_level.stats.keys():

                dispersion_ratio_value = result.dataset_level.stats["dispersion_ratio"]
                distance_ratio_value = result.dataset_level.stats["distance_ratio"]

                stats_values = {
                    "dispersion_ratio": dispersion_ratio_value,
                    "distance_ratio": distance_ratio_value,
                }

                validate_stats(stats_values, stats, expected_stats)

    elif result.dataset_level.dtype == OutputsTypes.KEY_ARRAY:
        x_values = result.dataset_level.value["precision_values"]
        y_values = result.dataset_level.value["recall_values"]

        assert (
            np.unique(x_values) == np.unique(y_values) == expected
        ), f"{metric_name}: All results should be the same and they are not."
    else:
        warnings.warn(f"Unknown output type: {result['dataset_level']['type']}. Skipping comparison.", stacklevel=2)

    if "instance_level" in result:
        value_inst = result["instance_level"]["value"]
        assert len(value_inst) == len(y_synth), "Instance-level values do not match the number of synthetic samples"


@pytest.mark.parametrize(
    "metric_name, order",
    [
        (synth_distance_metrics.WassersteinDistance, "normal"),
        (synth_distance_metrics.MMD, "normal"),
        (synth_distance_metrics.CosineSimilarity, "inv"),
        (synth_shared_metrics.PrecisionRecallDistribution, "inv"),
        (synth_shared_metrics.Authenticity, "normal"),
        (synth_shared_metrics.ImprovedPrecision, "inv"),
        (synth_shared_metrics.ImprovedRecall, "inv"),
        (synth_shared_metrics.Density, "inv"),
        (synth_shared_metrics.Coverage, "inv"),
    ],
)
def test_intersect_equal_distributions_order(metric_name, sample_distribution, order, show_dist=False):
    """Tests if the measures respect the expected order when they receive
    increasingly separate intersecting distributions with the same shape. That
    is:

    - statistical divergence measures --> increase
    - similarity measures --> decrease
    - dispersion ratios --> constant
    - distance ratios --> increase
    - precision-related measures --> decrease
    - recall-relatd measures --> decrease
    - authenticity-relatd measures --> increase
    """
    x_ref_min = sample_distribution((300, 2), sigma=0.2, mu=0)
    y_synth_min = sample_distribution((300, 2), sigma=0.2, mu=0.1)

    x_ref_middle = sample_distribution((300, 2), sigma=0.2, mu=0)
    y_synth_middle = sample_distribution((300, 2), sigma=0.2, mu=0.2)

    x_ref_max = sample_distribution((300, 2), sigma=0.2, mu=0)
    y_synth_max = sample_distribution((300, 2), sigma=0.2, mu=0.3)

    metric = metric_name()
    result_min = metric.compute(x_ref_min, y_synth_min)
    result_middle = metric.compute(x_ref_middle, y_synth_middle)
    result_max = metric.compute(x_ref_max, y_synth_max)

    if show_dist and metric_name == metric_name[0]:
        from src.general.visualization.features_visualization import plot_data

        plot_data(x_ref_min, y_synth_min, title="Scatter Plot of min distance data")
        plot_data(x_ref_middle, y_synth_middle, title="Scatter Plot of middle distance data")
        plot_data(x_ref_max, y_synth_max, title="Scatter Plot of max distance data")

    if result_min.dataset_level.dtype == OutputsTypes.NUMERIC:
        value_min = result_min.dataset_level.value
        value_middle = result_middle.dataset_level.value
        value_max = result_max.dataset_level.value

        if order == "normal":
            assert value_min < value_middle < value_max, f"{metric_name}: unexpected order"
        elif order == "inv":
            assert value_max < value_middle < value_min, f"{metric_name}: unexpected order"
        elif order == "equal":
            assert abs(value_max - value_middle) <= TOLERANCE
            assert abs(value_middle - value_min) <= TOLERANCE

    elif result_min.dataset_level.dtype == OutputsTypes.KEY_ARRAY:

        x_values_min = np.unique(result_min.dataset_level.value["precision_values"])
        y_values_min = np.unique(result_min.dataset_level.value["recall_values"])

        x_values_middle = np.unique(result_middle.dataset_level.value["precision_values"])
        y_values_middle = np.unique(result_middle.dataset_level.value["recall_values"])

        x_values_max = np.unique(result_max.dataset_level.value["precision_values"])
        y_values_max = np.unique(result_max.dataset_level.value["recall_values"])

        if order == "equal":
            assert (
                x_values_min == x_values_middle == x_values_max and y_values_min == y_values_middle == y_values_max
            ), f"{metric_name}: All results should be the same and they are not."
        elif order == "inv":
            assert np.mean(x_values_min) > np.mean(x_values_middle) > np.mean(x_values_max) and np.mean(
                y_values_min,
            ) > np.mean(y_values_middle) > np.mean(
                y_values_max,
            ), f"{metric_name}: All results should be the same and they are not."

    else:
        warnings.warn(
            f"{metric_name} | Unknown output type: {result_min['dataset_level']['type']}. Skipping comparison.",
            stacklevel=2,
        )

    if "instance_level" in result_min:
        value_inst = result_min["instance_level"]["value"]
        assert len(value_inst) == 300, "Instance-level values do not match the number of synthetic samples"


@pytest.mark.parametrize(
    "metric_name, expected_upper, sigma_y",
    [
        (synth_shared_metrics.ImprovedPrecision, 0.2, 0.5),  # portion of Y that cover X
        (synth_shared_metrics.Density, 0.3, 0.5),  # portion of X that cover Y (with density factor)
        (synth_shared_metrics.Coverage, 0.2, 0.1),  # portion of X that cover Y
        (synth_shared_metrics.Authenticity, 0.8, 0.5),
    ],
)
def test_distribution_shift(metric_name, sample_distribution, expected_upper, sigma_y, show_dist=False):
    x_ref = sample_distribution((100, 2), sigma=0.5)
    y_synth = sample_distribution((100, 2), sigma=sigma_y)
    y_synth[:80] += 5  # shift 20% of the data

    if show_dist:
        from src.general.visualization.features_visualization import plot_data

        plot_data(x_ref, y_synth)

    metric = metric_name()
    result = metric.compute(x_ref, y_synth)

    value = result.dataset_level.value

    assert value < expected_upper + TOLERANCE, f"{metric_name}: unexpected value of {value} for shifted distribution"

    if "instance_level" in result:
        assert len(result["instance_level"]["value"]) == len(
            y_synth,
        ), "Instance-level values do not match the number of synthetic samples"


@pytest.mark.parametrize(
    "metric_name, expected",
    [
        (synth_distance_metrics.WassersteinDistance, 0.5000000002399999),
        (synth_distance_metrics.MMD, 0.50000000024),
        (synth_distance_metrics.CosineSimilarity, 0.8370494332671239),
        (synth_shared_metrics.PrecisionRecallDistribution, (0.5350352553311978, 0.5246773400038105)),
        (synth_shared_metrics.FrechetDistance, 0.5000000060902672),
        (synth_shared_metrics.MultiScaleIntrinsicDistance, 51.405824554830545),
        (synth_shared_metrics.Authenticity, 0.5),
        (synth_shared_metrics.ImprovedPrecision, 1.0),
        (synth_shared_metrics.ImprovedRecall, 0.8),
        (synth_shared_metrics.Density, 0.5800000000000001),
        (synth_shared_metrics.Coverage, 0.8),
    ],
)
def test_reproducibility(metric_name, expected, show_dist=False):
    """Test if the measures are reproducible."""

    # From sample_distribution((10, 2), sigma=0.5, mu = 0):
    x_ref = np.array(
        [
            [0.88202617, 0.2000786],
            [0.48936899, 1.1204466],
            [0.933779, -0.48863894],
            [0.47504421, -0.0756786],
            [-0.05160943, 0.20529925],
            [0.07202179, 0.72713675],
            [0.38051886, 0.06083751],
            [0.22193162, 0.16683716],
            [0.74703954, -0.10257913],
            [0.15653385, -0.42704787],
        ],
    )

    # From sample_distribution((10, 2), sigma=0.5, mu = 0.5):
    y_synth = np.array(
        [
            [1.38202617e00, 7.00078604e-01],
            [9.89368992e-01, 1.62044660e00],
            [1.43377900e00, 1.13610601e-02],
            [9.75044209e-01, 4.24321396e-01],
            [4.48390574e-01, 7.05299251e-01],
            [5.72021786e-01, 1.22713675e00],
            [8.80518863e-01, 5.60837508e-01],
            [7.21931616e-01, 6.66837164e-01],
            [1.24703954e00, 3.97420868e-01],
            [6.56533851e-01, 7.29521303e-02],
        ],
    )

    if show_dist:
        from src.general.visualization.features_visualization import plot_data

        plot_data(x_ref, y_synth)

    metric = metric_name()
    result = metric.compute(x_ref, y_synth)

    if result.dataset_level.dtype == OutputsTypes.KEY_ARRAY:
        x_values = result.dataset_level.value["precision_values"]
        y_values = result.dataset_level.value["recall_values"]
        assert np.mean(x_values) == pytest.approx(expected[0]) and np.mean(y_values) == pytest.approx(
            expected[1],
        ), f"{metric_name}: unexpected value of {np.mean(x_values)} or {np.mean(y_values)}."

    elif result.dataset_level.dtype == OutputsTypes.NUMERIC:
        value = result.dataset_level.value

        assert value == pytest.approx(expected), f"{metric_name}: unexpected value of {value}."
    else:
        warnings.warn(f"Unknown output type: {result['dataset_level']['type']}. Skipping comparison.", stacklevel=2)


@pytest.mark.parametrize(
    "metric_name, kernel",
    [
        (synth_distance_metrics.MMD, "linear"),
        (synth_distance_metrics.CosineSimilarity, "multi_gaussian"),
        (synth_distance_metrics.CosineSimilarity, "sigmoid"),
    ],
)
def test_mmd_kerneis(metric_name, kernel, sample_distribution):

    x_ref = sample_distribution((300, 2), sigma=0.1, mu=0.3)
    y_synth = sample_distribution((300, 2), sigma=0.1, mu=0.5)

    metric = metric_name(kernel=kernel)
    result = metric.compute(x_ref, y_synth)

    value = result.dataset_level.value

    assert isinstance(value, float)


# ###### Test General Metrics Import #######


def test_all_attribute_exists():
    # Ensure __all__ exists and is a list
    assert hasattr(synth_shared_metrics, "__all__"), "__all__ attribute is missing"
    assert isinstance(synth_shared_metrics.__all__, list), "__all__ should be a list"


def test_import_classes():
    # Test if the classes are imported correctly
    assert (
        synth_shared_metrics.PrecisionRecallDistribution is not None
    ), "PrecisionRecallDistribution is not imported correctly"
    assert synth_shared_metrics.Authenticity is not None, "Authenticity is not imported correctly"
    assert synth_shared_metrics.Coverage is not None, "Coverage is not imported correctly"
    assert synth_shared_metrics.Density is not None, "Density is not imported correctly"
    assert synth_shared_metrics.ImprovedPrecision is not None, "ImprovedPrecision is not imported correctly"
    assert synth_shared_metrics.ImprovedRecall is not None, "ImprovedRecall is not imported correctly"


def test_extractor_model_name():
    # Set the default model name for the feature extractors
    synth_shared_metrics.FrechetDistance.extractor_model_name = "default"
    synth_shared_metrics.MultiScaleIntrinsicDistance.extractor_model_name = "default"

    # Test if the model names are set correctly
    assert (
        synth_shared_metrics.FrechetDistance.extractor_model_name == "default"
    ), "FrechetDistance extractor name not set correctly"
    assert (
        synth_shared_metrics.MultiScaleIntrinsicDistance.extractor_model_name == "default"
    ), "MultiScaleIntrinsicDistance extractor name not set correctly"
