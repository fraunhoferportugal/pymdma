from typing import List

import numpy as np
import pandas as pd


def is_float(
    string: str,
):
    """
    Check if the provided string can be converted to a float.

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
    thresh: int = 10,
):
    """
    Determine whether the provided data is categorical based on the ratio of unique values.

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

    return len_unq < thresh


def is_numeric(
    data: np.ndarray,
):
    """
    Check if all values in the provided data can be converted to numeric values.

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

def is_integer(
    data: np.ndarray,
):
    """
    Check if all values in the provided data can be converted to integer values.

    Parameters
    ----------
    data : np.ndarray
        The data to check for integer conversion.

    Returns
    -------
    bool
        True if the data is integer, False otherwise.
    """
    try:
        # for numerical arrays
        is_int = (np.array(data, int) == data).all()
    except Exception as e:
        # for non-numerical arrays
        is_int = False

    return is_int

def replace_missings_df(
    data: pd.DataFrame,
    missing_values: List = [999, -1, "NaN", "nan"],
    **kwargs,
):
    """
    Replace specified missing values in a pandas DataFrame with NaNs.

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
    """
    Replace specified missing values in a NumPy array with NaNs.

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
    """
    Automatically generate metadata about the columns of a dataset, identifying
    whether each column is categorical or numerical.

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

        # dtypes
        is_num = is_numeric(data_[:, idx])
        is_categ = is_categorical(data_[:, idx])
        is_int = is_integer(data_[:, idx])

        if is_categ or not is_num:
            # unique values
            opt_unq = np.unique(data_[:, idx]).astype(float if is_num else str)

            # remove NaN values
            opt_unq = opt_unq[~pd.isnull(opt_unq)]

            # encode
            opt_map = {v: k for k, v in enumerate(opt_unq, 1)}

            # categorical check
            is_categ = int((len(opt_unq)/len(data)) < 0.03)  # needs to be less than 3%

            # assign
            scores_d["type"] = {"tag": "discrete", "is_categ": is_categ, "opt": opt_map}
        else:
            # assign
            scores_d["type"] = {"tag": "continuous", "is_categ": 0, "opt": {}}

        # data type 
        if is_num:
            scores_d["dtype"] = "numerical" if not is_int else "integer"
        else:
            scores_d["dtype"] = "string"

        # assign
        vtype_d[col_] = scores_d

    return vtype_d


def get_dtypes_from_type_map(vtype_map: dict):
    """
    Automatically generate metadata about the columns of a dataset, identifying
    whether each column is categorical or numerical.

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
    
    aux_map = {"numerical": "float", "string": "str", "integer": "int"}

    # define variable mappers
    opt_map, dtype_map = {}, {}

    # get data type mapper per column
    for col, vtype in vtype_map.items():
        # column option range
        if vtype["type"]["tag"] != "numerical":
            opt_map[col] = vtype["type"].get("opt")
        else:
            opt_map[col] = {}

        # dtype options
        dtype_map[col] = aux_map.get(vtype["dtype"], "str")

    return dtype_map, opt_map
