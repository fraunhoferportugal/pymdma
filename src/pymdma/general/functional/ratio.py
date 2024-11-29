def dispersion_ratio(func, x_feat_1, x_feat_2, y_feat_1, y_feat_2):
    """Calculates the ratio of the distance between fake samples and the
    distance between real samples, using the metric passed in the argument
    "func".

    The dispersion ratio is a measure of how well the generator model
    is able to replicate the dispersion or spread of features present
    in the real samples.

    ratio = 1 --> distance between fake samples is equal to the
                  distance between real samples.(ideal scenario)
    ratio > 1 --> distance between fake samples is greater then
                  the distance between real samples.
    ratio < 1 --> distance between fake samples is smaller then
                  the distance between real samples.

    Parameters
    ----------
    func : callable
        A distance metric or function that takes two arrays of features
        as input and returns a scalar distance. This function is used to
        compute distances between samples.
    x_feat_1 : array-like of shape [n_samples/2, n_features]
        2D array with features of the original samples.
    x_feat_2 : array-like of shape [n_samples/2, n_features]
        2D array with features of the fake samples.
    y_feat_1 : array-like of shape [n_samples/2, n_features]
        2D array with features of the original samples.
    y_feat_2 : array-like of shape [n_samples/2, n_features]
        2D array with features of the fake samples.

    Returns
    -------
    dispersion_ratio : float
        The dispersion ratio calculated with the specified distance metric.
    """
    rr = func(x_feat_1, x_feat_2)
    ff = func(y_feat_1, y_feat_2)

    return ff / rr


def distance_ratio(func, x_feat_1, x_feat_2, y_feat_1, y_feat_2):
    """Calculates the ratio of the distance between real and fake samples and
    the distance of between real samples, using the metric passed in the
    argument "func".

    The distance ratio is a measure of how well the generator model
    is able to replicate the features from the real samples.

    ratio = 1 --> distance between real and fake samples is equal
                  to the distance between real samples.(ideal scenario)
    ratio > 1 --> distance between real and fake samples is greater
                  then the distance between real samples.
    ratio < 1 --> distance between real and fake samples is smaller
                  then the distance between real samples.

    Parameters
    ----------
    func : callable
        A distance metric or function that takes two arrays of features
        as input and returns a scalar distance. This function is used to
        compute distances between samples.
    x_feat_1 : array-like of shape [n_samples/2, n_features]
        2D array with features of the original samples.
    x_feat_2 : array-like of shape [n_samples/2, n_features]
        2D array with features of the fake samples.
    y_feat_1 : array-like of shape [n_samples/2, n_features]
        2D array with features of the original samples.
    y_feat_2 : array-like of shape [n_samples/2, n_features]
        2D array with features of the fake samples.

    Returns
    -------
    distance_ratio : float
        The distance ratio computed with the specified distance metric.
    """
    rr = func(x_feat_1, x_feat_2)
    fr = (func(x_feat_1, y_feat_2) + func(x_feat_2, y_feat_1)) / 2

    return fr / rr
