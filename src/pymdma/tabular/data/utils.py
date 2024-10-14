from typing import List

import numpy as np
import pandas as pd


def is_float(
    string: str,
):
    """Check if the provided string can be converted to a float.

    Parameters
    ----------
    string : str
        The string to check for float conversion.

    Returns
    -------
    bool
        True if the string can be converted to float, False otherwise.
    """

    try:
        float(string)
        return True
    except ValueError:
        return False


def is_categorical(
    data: np.ndarray,
    thresh: float = 0.4,
):
    """Determine whether the provided data is categorical based on the ratio of
    unique values.

    Parameters
    ----------
    data : np.ndarray
        The data to be evaluated for categorical nature.
    thresh : float, optional
        The threshold ratio of non-unique values to unique values for categorization.
        Default is 0.4.

    Returns
    -------
    bool
        True if the data is categorical based on the threshold, False otherwise.
    """

    # Number of samples
    len_data = len(data)

    # Number of unique samples
    len_unq = len(np.unique(data))

    # Ratio
    ratio = (len_data - len_unq) / (len_data + len_unq)

    return ratio > thresh


def is_numeric(
    data: np.ndarray,
):
    """Check if all values in the provided data can be converted to numeric
    values.

    Parameters
    ----------
    data : np.ndarray
        The data to check for numeric conversion.

    Returns
    -------
    bool
        True if the data is numeric, False otherwise.
    """

    try:
        # check if it is conversible to float
        _ = np.array(data).astype(float)
        is_num = True
    except Exception as e:
        is_num = False
    return is_num


def replace_missings_df(
    data: pd.DataFrame,
    missing_values: List = [999, -1, "NaN", "nan"],
    **kwargs,
):
    """Replace specified missing values in a pandas DataFrame with NaNs.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame where missing values need to be replaced.
    missing_values : list, optional
        A list of values to treat as missing (default is [999, -1, "NaN", "nan"]).

    Returns
    -------
    pd.DataFrame
        DataFrame with specified missing values replaced by NaN.
    """

    return data.replace(missing_values, np.nan)


def replace_missings_np(
    data: np.ndarray,
    missing_values: List = [999, -1, "NaN", "nan"],
    **kwargs,
):
    """Replace specified missing values in a NumPy array with NaNs.

    Parameters
    ----------
    data : np.ndarray
        The array where missing values need to be replaced.
    missing_values : list, optional
        A list of values to treat as missing (default is [999, -1, "NaN", "nan"]).

    Returns
    -------
    np.ndarray
        A new array with specified missing values replaced by NaN.
    """

    # copy of the original vector
    aux_data = data.copy()

    # replace all the instances of missing values with NaN tag
    aux_data[np.isin(aux_data, missing_values)] = np.nan

    return aux_data


def auto_metadata_generator(
    data: np.ndarray,
    cols: list,
    **kwargs,
):
    """Automatically generate metadata about the columns of a dataset,
    identifying whether each column is categorical or numerical.

    Parameters
    ----------
    data : np.ndarray
        The data for which metadata will be generated.
    cols : list of str
        A list of column names corresponding to the data columns.
    **kwargs : optional
        Additional keyword arguments for handling missing values or other preprocessing.

    Returns
    -------
    dict
        A dictionary containing metadata for each column, including type (categorical
        or continuous) and data type (numerical or categorical).
    """

    vtype_d = {}

    # encode missing values
    data_ = replace_missings_np(data, **kwargs)

    # Loop over columns
    for idx, col_ in enumerate(cols):
        # auxiliary vector
        scores_d = {}

        if is_categorical(data_[:, idx]):
            # unique values
            opt_unq = np.unique(data_[:, idx])

            # remove NaN values
            opt_unq = opt_unq[~pd.isnull(opt_unq)]

            # encode
            opt_map = {v: k for k, v in enumerate(opt_unq, 1)}

            # assign
            scores_d["type"] = {"tag": "discrete", "opt": opt_map}
        else:
            # assign
            scores_d["type"] = {"tag": "continuous", "opt": dict()}

        if is_numeric(data_[:, idx]):
            scores_d["dtype"] = "numerical"
        else:
            scores_d["dtype"] = "categorical"

        # assign
        vtype_d[col_] = scores_d

    return vtype_d


def get_dtypes_from_type_map(vtype_map: dict):
    """Automatically generate metadata about the columns of a dataset,
    identifying whether each column is categorical or numerical.

    Parameters
    ----------
    data : np.ndarray
        The data for which metadata will be generated.
    cols : list of str
        A list of column names corresponding to the data columns.
    **kwargs : optional
        Additional keyword arguments for handling missing values or other preprocessing.

    Returns
    -------
    dict
        A dictionary containing metadata for each column, including type (categorical
        or continuous) and data type (numerical or categorical).
    """

    aux_map = {"numerical": float, "categorical": str}

    # define variable mappers
    opt_map, dtype_map = {}, {}

    # get data type mapper per column
    for col, vtype in vtype_map.items():
        # column option range
        if vtype["type"]["tag"] != "numerical":
            opt_map[col] = vtype["type"].get("opt")
        else:
            opt_map[col] = dict()

        # dtype options
        dtype_map[col] = aux_map.get(vtype["dtype"], str)

    return dtype_map, opt_map
