from typing import List, Optional

from .tsfel import TSFEL


class ExtractorFactory:
    default = "tsfel"

    @staticmethod
    def model_from_name(
        name: str,
        domains: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initializes the feature extractor with the given parameters.

        Args:
            name (str): identifier of the extractor to be used.
            device (str): model device. Defaults to "cpu".
        """
        if name == "tsfel":
            return TSFEL(domains, **kwargs)
        else:
            raise ValueError(f"Model {name} not available.")
