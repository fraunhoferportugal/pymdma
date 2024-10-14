import numpy as np


def row_to_text(cols: np.ndarray, vals: np.ndarray, shuffle: bool = True):
    """Converts a row of data into a textual description where each column-
    value pair is expressed as a sentence.

    Parameters
    ----------
    cols : np.ndarray
        The array of column names corresponding to the values in the row.
    vals : np.ndarray
        The array of values corresponding to the columns for a single row.
    shuffle : bool, optional (default=True)
        If True, the column-value pairs in the text are shuffled randomly. If False, they appear in the original order.

    Returns
    -------
    str
        A string that describes the row as a series of "column is value" sentences, joined by commas.

    Example
    -------
    >>> out = row_to_text(np.array(['age', 'height']), np.array([25, 180]))
    >>> print(out)
    'height is 180, age is 25'
    """

    # record encoded to text
    text_ = [f"{col} is {val}" for col, val in zip(cols, vals)]

    # if shuffle is enabled
    if shuffle:
        np.random.shuffle(text_)

    return ", ".join(text_)


# Table to Text
def tabular_to_text(data: np.ndarray, column_names: np.ndarray, shuffle: bool = False, **kwargs):
    """Converts a tabular dataset into text by converting each row into a
    textual description.

    Parameters
    ----------
    data : np.ndarray
        Input tabular data. Should be a 2D array.
    column_names : np.ndarray
        List of feature names.
    shuffle : bool, optional (default=False)
        If True, the column-value pairs for each row are shuffled randomly in the textual output.
    **kwargs
        Additional arguments to be passed to the function.

    Returns
    -------
    list
        A list of strings where each string is the textual description of a row.

    Example
    -------
    >>> data = np.array([[25, 180], [30, 175]])
    >>> column_names = np.array(['age', 'height'])
    >>> out = tabular_to_text(data, column_names)
    >>> print(out)
    ['age is 25, height is 180', 'age is 30, height is 175']
    """

    return [row_to_text(column_names, row, shuffle=shuffle) for row in data]
