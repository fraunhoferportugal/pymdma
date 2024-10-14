from typing import Optional

import numpy as np
import spacy
import torch
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from transformers import BertModel, BertTokenizer
from umap.umap_ import UMAP

from ..embeddings import GenericEmbedder
from ..embeddings.utils import tabular_to_text


class UMAPEmbedder(GenericEmbedder):
    """UMAP (Uniform Manifold Approximation and Projection) embedder using
    default and customizable parameters for dimensionality reduction.

    Example
    -------
    >>> import numpy as np
    >>> embedder = UMAPEmbedder(n_neighbors=10, n_components=2)
    >>> X = np.random.rand(100, 10)
    >>> X_embed = embedder.fit_transform(X)
    >>> print(X_embed)
    """

    D_PARAMS = {
        "n_neighbors": 15,
        "n_components": 3,
        "metric": "euclidean",
        "output_metric": "euclidean",
        "n_epochs": None,
        "learning_rate": 1.0,
        "init": "spectral",
        "min_dist": 0.1,
        "spread": 1.0,
        "low_memory": True,
        "n_jobs": 1,
        "random_state": 42,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # embedder params
        emb_params = {**self.D_PARAMS, **kwargs}

        # initialize
        try:
            # try instance with provided params
            self.emb_obj = UMAP(**emb_params)
        except Exception as e:
            logger.debug(f"Error:\n{e}")

            # default instance (in case of any error)
            self.emb_obj = UMAP(**self.D_PARAMS)

        # has inverse transform
        self.has_inv = True

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        force_all_finite: bool = "allow-nan",
        **kwargs,
    ) -> None:
        self.emb_obj.fit(X, y, force_all_finite)
        return self

    def transform(
        self,
        X: np.ndarray,
        force_all_finite: bool = "allow-nan",
        **kwargs,
    ) -> np.ndarray:
        return self.emb_obj.transform(X, force_all_finite)

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        force_all_finite: bool = "allow-nan",
        **kwargs,
    ) -> np.ndarray:
        return self.emb_obj.fit_transform(X, y, force_all_finite)

    def inverse_transform(self, X, **kwargs):
        return self.emb_obj.inverse_transform(X)


class PCAEmbedder(GenericEmbedder):
    """PCA (Principal Component Analysis) embedder for dimensionality
    reduction.

    Example
    -------
    >>> import numpy as np
    >>> embedder = PCAEmbedder(n_components=5)
    >>> X = np.random.rand(100, 10)
    >>> X_embed = embedder.fit_transform(X)
    >>> print(X_embed)
    """

    D_PARAMS = {
        "n_components": None,
        "copy": True,
        "whiten": False,
        "svd_solver": "auto",
        "tol": 0.0,
        "iterated_power": "auto",
        "n_oversamples": 10,
        "power_iteration_normalizer": "auto",
        "random_state": 42,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # embedder params
        emb_params = {**self.D_PARAMS, **kwargs}

        # initialize
        try:
            self.emb_obj = PCA(**emb_params)
        except Exception as e:
            print(f"Error: {e}")
            self.emb_obj = PCA(**self.D_PARAMS)

        # has inverse transform
        self.has_inv = True

    def fit(self, X, y=None, **kwargs):
        self.emb_obj.fit(X, y)
        return self

    def transform(self, X, **kwargs):
        return self.emb_obj.transform(X)

    def fit_transform(self, X, **kwargs):
        return self.emb_obj.fit_transform(X)

    def inverse_transform(self, X, **kwargs):
        return self.emb_obj.inverse_transform(X)


class LLEEmbedder(GenericEmbedder):
    """Locally Linear Embedding (LLE) embedder for non-linear dimensionality
    reduction.

    Example
    -------
    >>> import numpy as np
    >>> embedder = LLEEmbedder(n_neighbors=10, n_components=2)
    >>> X = np.random.rand(100, 10)
    >>> X_embed = embedder.fit_transform(X)
    >>> print(X_embed)
    """

    D_PARAMS = {
        "n_neighbors": 5,
        "n_components": 2,
        "reg": 0.001,
        "eigen_solver": "dense",
        "random_state": 42,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # embedder params
        emb_params = {**self.D_PARAMS, **kwargs}

        # initialize
        try:
            self.emb_obj = LocallyLinearEmbedding(**emb_params)
        except Exception as e:
            logger.debug(f"Error:\n{e}")
            self.emb_obj = LocallyLinearEmbedding(**self.D_PARAMS)

        # has inverse transform
        self.has_inv = False

    def fit(self, X, y=None, **kwargs):
        self.emb_obj.fit(X, y)
        return self

    def transform(self, X, **kwargs):
        return self.emb_obj.transform(X)

    def fit_transform(self, X, y=None, **kwargs):
        return self.emb_obj.fit_transform(X, y)


class IsomapEmbedder(GenericEmbedder):
    """Isomap embedder for manifold learning and dimensionality reduction.

    Example
    -------
    >>> import numpy as np
    >>> embedder = IsomapEmbedder(n_neighbors=5, n_components=3)
    >>> X = np.random.rand(100, 10)
    >>> X_embed = embedder.fit_transform(X)
    >>> print(X_embed)
    """

    D_PARAMS = {
        "n_neighbors": 6,
        "radius": None,
        "n_components": 3,
        "eigen_solver": "auto",
        "tol": 0,
        "max_iter": None,
        "path_method": "auto",
        "neighbors_algorithm": "auto",
        "n_jobs": -1,
        "metric": "minkowski",
        "p": 2,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # embedder params
        emb_params = {**self.D_PARAMS, **kwargs}

        # initialize
        try:
            self.emb_obj = Isomap(**emb_params)
        except Exception as e:
            print(f"Error: {e}")
            self.emb_obj = Isomap(**self.D_PARAMS)

        # has inverse transform
        self.has_inv = False

    def fit(self, X, **kwargs):
        self.emb_obj.fit(X)
        return self

    def transform(self, X, **kwargs):
        return self.emb_obj.transform(X)

    def fit_transform(self, X, **kwargs):
        return self.emb_obj.fit_transform(X)


class SpacyWordEmbedder(GenericEmbedder):
    """Word embedding model using spaCy for text data transformation into
    vectors.

    Parameters
    ----------
    model_name : str
        The name of the spaCy model to use for word embeddings.

    Example
    -------
    >>> import numpy as np
    >>> embedder = SpacyWordEmbedder(model_name='en_core_web_md')
    >>> X = np.random.rand(100, 10)
    >>> X_embed = embedder.transform(X)
    >>> print(X_embed)
    """

    def __init__(self, model_name: Optional[str] = "en_core_web_md"):
        super().__init__()
        try:
            # try to load model (check if exists)
            self.emb_obj = spacy.load(model_name)
        except (SystemExit, SystemError, OSError) as e:
            logger.debug(f"Error:\n{e}")
            # download it
            spacy.cli.download(model_name)

            # now load it again
            self.emb_obj = spacy.load(model_name)

        # has inverse transform
        self.has_inv = False

    def table_to_text(self, X, **kwargs):
        # records to prompt
        prompts = tabular_to_text(X, **kwargs)
        return prompts

    def fit(self, X, y=None, **kwargs):
        # word embeddings are pre-trained,
        # so no fitting needed
        return self

    def transform(self, X, **kwargs):
        # prompt-encoded data
        logger.debug(X)
        X_ = self.table_to_text(X, **kwargs)

        # get embeddings
        embeddings = [self.emb_obj(sentence).vector for sentence in X_]

        return np.array(embeddings)

    def fit_transform(self, X, **kwargs):
        return self.transform(X, **kwargs)


class BertSentenceEmbedder(GenericEmbedder):
    """BERT-based sentence embedder for transforming text data into dense
    vectors.

    Parameters
    ----------
    model_name : str
        The name of the BERT model to use for sentence embeddings.

    Example
    -------
    >>> import numpy as np
    >>> embedder = BertSentenceEmbedder()
    >>> X = np.random.rand(100, 10)
    >>> X_embed = embedder.transform(X)
    >>> print(X_embed)
    """

    def __init__(self, model_name: Optional[str] = "bert-base-uncased"):
        super().__init__()
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.emb_obj = BertModel.from_pretrained(model_name)
        except Exception as e:
            logger.debug(f"Error:\n{e}")
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.emb_obj = BertModel.from_pretrained("bert-base-uncased")

        # has inverse transform
        self.has_inv = False

    def table_to_text(self, X, **kwargs):
        # encode tabular rows into text prompts
        table_prompts = tabular_to_text(data=X, **kwargs)

        return table_prompts

    def fit(self, X, y=None, **kwargs):
        # sentence embeddings are pre-trained, so no fitting needed
        return self

    def transform(self, X, **kwargs):
        # prompt-encoded data
        X_ = self.table_to_text(X, **kwargs)

        # tokenize input
        tokenized = self.tokenizer(
            X_,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # get embeddings
        with torch.no_grad():
            outputs = self.emb_obj(**tokenized)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def fit_transform(self, X, **kwargs):
        return self.transform(X, **kwargs)


class GlobalEmbedder:
    """Global embedder that selects and applies the appropriate embedding
    method.

    Parameters
    ----------
    n_type : Optional[str]
        The type of embedding method to use.
        Must be one of: "umap", "pca", "lle", "isomap", "spacy", "bert", "identity".

    Example
    -------
    >>> import numpy as np
    >>> embedder = GlobalEmbedder(n_type="pca")
    >>> X = np.random.rand(100, 10)
    >>> X_embed = embedder.fit_transform(X)
    >>> print(X_embed)
    """

    E_MAP = {
        "umap": UMAPEmbedder,
        "pca": PCAEmbedder,
        "lle": LLEEmbedder,
        "isomap": IsomapEmbedder,
        "spacy": SpacyWordEmbedder,
        "bert": BertSentenceEmbedder,
        "identity": GenericEmbedder,
    }

    def __init__(self, n_type: Optional[str] = "umap", **kwargs):
        # get embedding object
        if isinstance(n_type, str) and n_type in self.E_MAP:
            # tag
            norm_tag = n_type.lower()

            # embed class
            cls_ = self.E_MAP.get(norm_tag, GenericEmbedder)

            # embed instance
            self.emb_obj = cls_(**kwargs)
        else:
            # default embed instance
            self.emb_obj = GenericEmbedder()

        # has inverse transform
        self.has_inv = self.emb_obj.has_inv

    def fit(self, X, **kwargs):
        self.emb_obj.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        return self.emb_obj.transform(X, **kwargs)

    def fit_transform(self, X, **kwargs):
        return self.emb_obj.fit_transform(X, **kwargs)

    def inverse_transform(self, X, **kwargs):
        return self.emb_obj.inverse_transform(X, **kwargs)
