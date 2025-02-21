from typing import List, Optional

import tsfel

from .extractor import BaseTSExtractor


class TSFEL(BaseTSExtractor):
    def __init__(
        self,
        domains: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        # Generate default domain value
        """Initializes the TSFEL feature extractor with the specified domains
        and verbosity.

        Parameters
        ----------
        domains : Optional[List[str]]
            A list of domains to extract features from. If None, the default domains
            ["temporal", "statistical", "spectral"] will be used.
        verbose : bool
            If True, enables verbose output during feature extraction.
        """

        if domains is None:
            domains = ["temporal", "statistical", "spectral"]
        self.domains = domains
        self.verbose = verbose

        # update domain configurations
        self.cfg_file = {}
        for domain in self.domains:
            self.cfg_file.update(tsfel.get_features_by_domain(domain))

    def __call__(self, batch_windows, fs, dims):
        """Extracts features from a batch of samples.

        Parameters
        ----------
        batch_windows: List
            Batch of signals with len(dims) chans.
        fs: int
            Sampling frequency
        dims: List(str)
           list with the names of each signal dimension/channel ex: name of each ECG Lead

        Returns
        -------
        features: DataFrame
            DataFrame with the features from each batch.
        """
        features = tsfel.time_series_features_extractor(
            self.cfg_file,
            batch_windows,
            fs=fs,
            window_size=None,
            header_names=dims,
            verbose=int(self.verbose),
        )

        return features
