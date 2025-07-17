from pathlib import Path
from typing import List, Optional, Union, Tuple, Callable

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..data.harmonize import MetaEncoder
from ..embeddings.embed import GlobalEmbedder
from ..embeddings.scale import GlobalNorm
from ..embeddings.impute import GlobalImputer


def _read_xml(path: Path, **kwargs):
    """Read an XML file and converts it into a pandas DataFrame.

    Parameters
    ----------
    path : Path
        The path to the XML file.
    **kwargs : optional
        Additional arguments passed to pandas' read_xml function.

    Returns
    -------
    pd.DataFrame or None
        A DataFrame containing the data read from the XML file, or None if an error occurs.
    """

    try:
        table = pd.read_xml(path, **kwargs)
    except Exception as e:
        table = None
    return table


def _read_json(path, **kwargs):
    """Read a JSON file converts it into a pandas DataFrame.

    Parameters
    ----------
    path : str
        The path to the JSON file.
    **kwargs : optional
        Additional arguments passed to pandas' read_json function.

    Returns
    -------
    pd.DataFrame or None
        A DataFrame containing the data read from the JSON file, or None if an error occurs.
    """

    try:
        table = pd.read_json(path, orient="table", **kwargs)
    except Exception as e:
        table = None
    return table


def _read_xlsx(path: Path, **kwargs):
    """Read an Excel file and converts it into a pandas DataFrame.

    Parameters
    ----------
    path : Path
        The path to the Excel file.
    **kwargs : optional
        Additional arguments passed to pandas' read_excel function.

    Returns
    -------
    pd.DataFrame or None
        A DataFrame containing the data read from the Excel file, or None if an error occurs.
    """

    try:
        table = pd.read_excel(path, **kwargs)
    except Exception as e:
        table = None
    return table


def _read_csv(path: Path, **kwargs):
    """Read a CSV file and converts it into a pandas DataFrame.

    Parameters
    ----------
    path : Path
        The path to the CSV file.
    **kwargs : optional
        Additional arguments passed to pandas' read_csv function.

    Returns
    -------
    pd.DataFrame or None
        A DataFrame containing the data read from the CSV file, or None if an error occurs.
    """

    try:
        table = pd.read_csv(path, **kwargs)
    except Exception as e:
        table = None
    return table


class TabularDataset(Dataset):
    """
    A class for handling tabular datasets, including loading, encoding, and scaling.

    Parameters
    ----------
    file_path : Path or Callable
        The path to the dataset file or a callable for loading data.
    data : pd.DataFrame, optional
        A DataFrame containing the dataset. If provided, it bypasses loading from file.
    tag_name : str, optional
        A string to tag or label the dataset (e.g., 'real', 'synthetic').
    qi_names : list of str, optional
        List of quasi-identifier column names.
    sens_names : list of str, optional
        List of sensitive attribute column names.
    scaler : str or object, optional
        The scaling method or a scaler object to be used for data normalization.
    embed : str or object, optional
        The embedding method or an embedder object to be used for dimensionality reduction.
    meta : object, optional
        An instance of a meta encoder for variable type inference.
    scaler_kwargs : dict, optional
        Additional arguments for the scaler.
    embed_kwargs : dict, optional
        Additional arguments for the embedder.
    meta_kwargs : dict, optional
        Additional arguments for the meta encoder.
    with_onehot : bool, optional
        Whether to apply one-hot encoding to categorical variables.
    init_proc : bool, optional
        Whether to perform initial data processing.
    **kwargs : optional
        Additional arguments passed to the Dataset class.
    """
    
    LOAD_MAP = {
        "xml": _read_xml,
        "json": _read_json,
        "xlsx": _read_xlsx,
        "csv": _read_csv,
    }

    EMBED_AVAIL = [
        "umap",
        "pca",
        "lle",
        "isomap",
    ]

    SCALE_AVAIL = [
        "identity",
        "standard",
        "maxabs",
        "minmax",
        "robust",
    ]

    def __init__(
        self,
        file_path: Optional[Union[Path, str]],
        data: Optional[pd.DataFrame] = None,
        tag_name: Optional[str] = 'real',
        qi_names: Optional[str] = None,
        sens_names: Optional[str] = None,
        scaler: Optional[str] = None,
        embed: Optional[str] = None,
        imputer: Optional[str] = None,
        meta: Optional[object] = None,
        scaler_kwargs: Optional[dict] = {},
        embed_kwargs: Optional[dict] = {},
        imputer_kwargs: Optional[dict] = {},
        meta_kwargs: Optional[dict] = {},
        with_onehot: Optional[bool] = False,
        init_proc: Optional[bool] = True,
        **kwargs,
    ):
        """
        Initializes the TabularDataset with data loading, encoding, and scaling options.

        Parameters
        ----------
        file_path : Path
            The path to the dataset file.
        data : pd.DataFrame, optional
            A DataFrame containing the dataset. If provided, it bypasses loading from file.
        qi_names : list of str, optional
            List of quasi-identifier column names.
        sens_names : list of str, optional
            List of sensitive attribute column names.
        scaler : str or object, optional
            The scaling method or a scaler object to be used for data normalization.
        embed : str or object, optional
            The embedding method or an embedder object to be used for dimensionality reduction.
        meta : object, optional
            An instance of a meta encoder for variable type inference.
        scaler_kwargs : dict, optional
            Additional arguments for the scaler.
        embed_kwargs : dict, optional
            Additional arguments for the embedder.
        meta_kwargs : dict, optional
            Additional arguments for the meta encoder.
        with_onehot : bool, optional
            Whether to apply one-hot encoding to categorical variables.
        **kwargs : optional
            Additional arguments passed to the Dataset class.
        """
        
        # file path
        self.fpath = file_path

        # one hot encoding option
        self.oh_flag = with_onehot
        
        # dataset tag name
        self.tag = tag_name

        # metadata encoder
        self.col_map = None
        if meta is None:
            self.meta = MetaEncoder(**meta_kwargs)
            self.fit_meta = True
        else:
            self.meta = meta
            self.fit_meta = False

        # scaler
        if isinstance(scaler, (type(None), str)):
            self.scaler = GlobalNorm(n_type=scaler, **scaler_kwargs)
            self.fit_sc = True
        elif isinstance(scaler, object):
            self.scaler = scaler
            self.fit_sc = False
        else:
            self.scaler, self.fit_sc = None, None

        # embedder
        if isinstance(embed, (type(None), str)):
            self.embed = GlobalEmbedder(n_type=embed, **embed_kwargs)
            self.fit_emb = True
        elif isinstance(embed, object):
            self.embed = embed
            self.fit_emb = False
        else:
            self.embed, self.fit_emb = None, None

        # imputer
        if isinstance(imputer, (type(None), str)):
            self.imputer = GlobalImputer(n_type=imputer, **imputer_kwargs)
            self.fit_imp = True
        elif isinstance(imputer, object):
            self.imputer = imputer
            self.fit_imp = False
        else:
            self.imputer, self.fit_imp = None, None

        # data
        self.data = data
        self.data_enc = None
        self.data_s = None
        self.data_emb = None
        self.cols = None
        self.col_inds = None

        # special names
        self.qi_names, self.qi_cols = qi_names, None
        self.sens_names, self.sens_cols = sens_names, None

        # include data processing in init
        if init_proc:
            self.setup(self.data)

    def setup(self, data: Optional[pd.DataFrame] = None):
        # data transformations
        self.data, self.data_enc, self.data_s, self.cols = self.transform(
            path=self.fpath,
            data=data,
            scale_fit=self.fit_sc,
            meta_fit=self.fit_meta,
        )

        # data embeddings
        self.data_emb = self.embed_encode(
            data=self.data_s,
            with_fit=self.fit_emb,
            column_names=self.cols,
        )

        # special attributes
        self.qi_cols = self.get_special_columns(self.qi_names)  # quasi-identifiers
        self.sens_cols = self.get_special_columns(self.sens_names)  # sensitive attributes

    @property
    def properties(self):
        """
        Returns the properties of the dataset including column names,
        quasi-identifiers, sensitive attributes, and column mappings.

        Returns
        -------
        dict
            A dictionary containing the dataset properties.
        """
        
        return {
            "column_names": self.cols,
            "qi_names": self.qi_cols,
            "sens_names": self.sens_cols,
            "col_map": self.col_map,
        }

    def __len__(self):
        """
        Returns the number of rows in the dataset.

        Returns
        -------
        int
            The number of rows in the dataset.
        """
        
        return len(self.data)

    def __getcols__(self):
        """
        Returns the column names of the dataset provided.

        Returns
        -------
        list
            A list of column names in the dataset.
        """
        
        return self.cols

    def __getitem__(self, idx):
        """
        Returns the target data to use at a specific index, whether it is encoded, scaled, or embedded.

        Parameters
        ----------
        idx : int
            The index of the data to retrieve.

        Returns
        -------
        array
            An array containing target data at the specified index.
        """
        return self.data_s[idx]

    def __getdata__(self):
        """
        Returns the target data to use whether it is encoded, scaled, or embedded.

        Parameters
        ----------

        Returns
        -------
        array
            An array containing the whole target dataset.
        """
        return self.data_s
    
    def get_data(self):
        return self.__getdata__()
    
    def get_cols_inds(self):
        return self.col_inds

    def get_special_columns(self, names: List[str] = None):
        """
        Retrieves specified special columns based on provided names.

        Parameters
        ----------
        names : list of str, optional
            A list of names of the special columns to retrieve.

        Returns
        -------
        list
            A list of special columns found in the dataset.
        """
        
        # get special columns
        if names is not None:
            filt_names = []
            # check if names provided are acceptable real column names
            for name in names:
                find_ = np.char.find(self.cols, name) + 1
                if sum(find_ > 0) > 0:
                    filt_names.append(name)
            if not filt_names:
                filt_names = self.cols
        else:
            filt_names = self.cols

        return filt_names

    def meta_encode(
        self,
        data: np.ndarray,
        column_names: list = None,
        with_fit: bool = True,
        **kwargs,
    ):
        """
        Encodes the dataset using the meta encoder.

        Parameters
        ----------
        data : np.ndarray
            The input data to be encoded.
        column_names : list, optional
            The names of the columns to encode.
        with_fit : bool, optional
            Whether to fit the meta encoder before encoding.

        Returns
        -------
        np.ndarray
            The encoded dataset.
        """
        
        # infer variable dtypes from tabular data provided
        if with_fit:
            _ = self.meta.metadata(
                data=data,
                cols=column_names,
                **kwargs,
            )

        # encode data
        data_enc = self.meta.encode(
            data=data,
            column_names=column_names,
            **kwargs,
        )

        return data_enc

    def meta_decode(
        self,
        data: np.ndarray,
        columns_names: list = None,
        **kwargs,
    ):
        """
        Decodes the dataset using the meta encoder.

        Parameters
        ----------
        data : np.ndarray
            The encoded data to be decoded.
        columns_names : list, optional
            The names of the columns to decode.

        Returns
        -------
        np.ndarray
            The decoded dataset.
        """
        
        return self.meta.decode(data=data, column_names=columns_names, **kwargs)

    def scale_encode(
        self,
        data: np.ndarray,
        cols: List[int] = None,
        with_fit: bool = True,
        **kwargs,
    ):
        """
        Normalizes the dataset using the specified scaler.

        Parameters
        ----------
        data : np.ndarray
            The input data to be normalized.
        with_fit : bool, optional
            Whether to fit the scaler to the data before transforming.

        Returns
        -------
        np.ndarray
            The normalized dataset.
        """
        # copy of the data
        data_s = np.copy(data)

        # normalize data provided
        if with_fit:
            if cols is not None:
                data_s[:, cols] = self.scaler.fit_transform(data[:, cols], **kwargs)
            else:
                data_s = self.scaler.fit_transform(data, **kwargs)
        else:
            if cols is not None:
                data_s[:, cols] = self.scaler.transform(data[:, cols], **kwargs)
            else:
                data_s = self.scaler.transform(data, **kwargs)

        return data_s.astype(float)

    def scale_decode(
        self,
        data: np.ndarray,
        cols: List[int] = None,
        **kwargs,
    ):
        """
        Inverses the normalization on the dataset.

        Parameters
        ----------
        data : np.ndarray
            The normalized data to be inversely transformed.

        Returns
        -------
        np.ndarray
            The original dataset after inverse normalization.
        """
        # copy of the data
        data_cp = np.copy(data)

        # inverse transform
        if cols is not None:
            data_cp[:, cols] = self.scaler.inverse_transform(data[:, cols])
        else:
            data_cp = self.scaler.inverse_transform(data)

        return data_cp

    def embed_encode(
        self,
        data: np.ndarray,
        with_fit: bool = True,
        **kwargs,
    ):
        """
        Transforms the dataset using the specified embedding method.

        Parameters
        ----------
        data : np.ndarray
            The input data to be embedded.
        with_fit : bool, optional
            Whether to fit the embedder to the data before transforming.

        Returns
        -------
        np.ndarray
            The embedded dataset.
        """
        
        # if fit is needed before transforming
        if with_fit:
            data_emb = self.embed.fit_transform(data, **kwargs)
        
        # otherwise
        else:
            data_emb = self.embed.transform(data, **kwargs)

        return data_emb

    def embed_decode(
        self,
        data: np.ndarray,
        **kwargs,
    ):
        """
        Reverses the transformation on the dataset.

        Parameters
        ----------
        data : np.ndarray
            The embedded data to be inversely transformed.

        Returns
        -------
        np.ndarray
            The original dataset after inverse embedding.
        """
        
        return self.embed.inverse_transform(data)

    def read_data(
        self,
        path: Optional[Union[Path, str]] = None,
        data: pd.DataFrame = None,
        **kwargs,
    ):
        """
        Reads the dataset from a file or returns the provided DataFrame.

        Parameters
        ----------
        path : Path, optional
            The path to the dataset file.
        data : pd.DataFrame, optional
            A DataFrame to return if provided.

        Returns
        -------
        pd.DataFrame or np.ndarray
            The loaded dataset as a DataFrame or numpy array.
        """
        
        # data reading
        if isinstance(data, (pd.DataFrame, np.ndarray)):
            data_r, tgt_r = data, None

        elif isinstance(path, (str, Path)):
            fmt = Path(path).suffix[1:].lower()
            loader = self.LOAD_MAP.get(fmt)

            # get data
            data_r = loader(path, **kwargs) if loader is not None else None
            tgt_r = None

        elif isinstance(path, Callable):
            data_r, tgt_r = path()

        else:
            data_r, tgt_r = None, None

        return data_r, tgt_r
    
    def impute_missing(self, data: np.ndarray, cols: List[int] = None, miss_inds: List[Tuple[int, int]] = None, meta_fit: bool = True, **kwargs):
        # meta encoding
        _ = self.meta_encode(
            data=data,
            column_names=cols,
            with_fit=meta_fit,
            with_onehot=False,
        )

        # for imputation
        data_ = self.meta_encode(
            data=data,
            column_names=cols,
            with_fit=False,
            with_onehot=False,
        )

        # replace back missing values
        if miss_inds:
            data_[tuple(zip(*miss_inds))] = np.nan
        
        # impute missing values
        imp = GlobalImputer(n_type='knn', norm_type='stardard', **kwargs)
        data_imp = imp.fit_transform(data_)

        # decode back
        data_dec = self.meta_decode(
            data=data_imp, 
            columns_names=cols, 
            with_onehot=False
        )

        # reset metadata params
        if meta_fit:
            self.meta.reset_params()

        return data_dec

    def transform(
        self,
        path: Path = None,
        data: pd.DataFrame = None,
        scale_fit: bool = True,
        meta_fit: bool = True,
        **kwargs,
    ):
        """
        Loads, encodes, and scales the dataset.

        Parameters
        ----------
        path : Path, optional
            The path to the dataset file.
        data : pd.DataFrame, optional
            A DataFrame to be processed if provided.
        scale_fit : bool, optional
            Whether to fit the scaler to the data.
        meta_fit : bool, optional
            Whether to fit the meta encoder to the data.

        Returns
        -------
        tuple
            A tuple containing the original data, encoded data, scaled data, and column names.
        """
        
        # data loader
        data_r, _ = self.read_data(path, data, **kwargs)

        # get missing indices
        miss_inds = list(zip(*np.where(data_r.isna().to_numpy())))

        # replace missing values
        data_r = data_r.fillna(data_r.mode().iloc[0])
        
        # check if data was correctly loaded
        if data_r is not None:
            # get columns/attributes
            cols = list(data_r.columns)

            # numpy array
            data_np = data_r.to_numpy()

            # impute missing values
            data_imp = self.impute_missing(
                data=data_np,
                cols=cols,
                miss_inds=miss_inds,
                meta_fit=meta_fit
            )

            # meta encoding
            data_enc = self.meta_encode(
                data=data_imp,
                column_names=cols,
                with_fit=meta_fit,
                with_onehot=self.oh_flag,
            )

            # column map
            self.col_map = self.meta.vtype_map

            # column indices
            or_inds = self.meta.or_inds
            self.col_inds = self.meta.col_inds  # assign col indices

            # norm scaling
            data_s = self.scale_encode(
                data=data_enc, 
                cols=or_inds, 
                with_fit=scale_fit
            )

        else:
            # default values
            data_r, data_enc, data_s, cols = [], [], [], []

        return data_r, data_enc, data_s, cols

    def inverse_transform(self, data: np.ndarray, column_names: list = None):
        """
        Applies the inverse transformations on the dataset.

        Parameters
        ----------
        data : np.ndarray
            The data to apply inverse transformations to.
        column_names : list, optional
            Full list of column names.

        Returns
        -------
        np.ndarray
            The dataset after applying inverse transformations.
        """
        
        # inverse scaling
        col_inds = self.meta.or_inds
        data = self.scale_decode(data, cols=col_inds)

        # meta decoding
        data = self.meta_decode(
            data=data,
            columns_names=column_names,
            with_onehot=self.oh_flag,
        )

        return data
    
if __name__ == '__main__':
    import pandas as pd
    
    # data
    data = pd.DataFrame({'a': [1,2,3,4,5], 'b': [1,2,3,4,5], 'c': [0.2,0.3,0.5,0.6,5.4], 'd': ['a','b','c','d','e'], 'e': [0,1,0,1,0]})

    # loader
    loader = TabularDataset(data=data, init_proc=False)

    # setup
    data = loader.setup(data=data)

