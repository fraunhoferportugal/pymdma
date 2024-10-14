import os
from typing import Literal, Optional

import nltk
import numpy as np

from pymdma.common.definitions import Metric
from pymdma.common.output import MetricResult
from pymdma.config import package_base_dir
from pymdma.constants import EvaluationLevel, InputPrivacyMetrics, OutputsTypes, ReferenceType

from ...api_anonimization import TextAnonymizer


class Identifiability(Metric):
    """Computes the identifiability of a text dataset. Detects the presence of
    personal information in a text dataset. Uses a pre-trained model to
    anonymize the text and returns the number of personal information detected.

    Parameters
    ----------
    language: str
        Language of the text for indentifiability evaluation. Can be "en" for English or "es" for Spanish.
        If not provided, each text sample must have a language specifier.
    **kwargs : dict
        Additional arguments for compatiblity.

    Examples
    --------
    >>> ident = Identifiability(language="en")
    >>> data = [{"id": 1, "text": "John Doe is a patient.", "language": "en"}]
    >>> result: MetricResult = ident.compute(data)
    """

    reference_type = ReferenceType.NONE
    evaluation_level = EvaluationLevel.INSTANCE
    metric_goal = InputPrivacyMetrics.CONFIDENTIALITY

    higher_is_better: bool = False
    min_value: float = 0
    max_value: float = np.inf

    def __init__(
        self,
        language: Optional[Literal["en", "es"]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        nltk.download("punkt")

        self.eng_anonymizer = None
        self.es_anonymizer = None

        if language is None or language == "en":
            model_path = os.path.join(package_base_dir, "text", "models", "ClinicalBERT")
            self.eng_anonymizer = TextAnonymizer(
                repo_name=model_path,
            )
        if language is None or language == "es":
            model_path = os.path.join(package_base_dir, "text", "models", "bert-spanish-cased-finetuned-ner")
            self.es_anonymizer = TextAnonymizer(
                repo_name=model_path,
            )
        self.language = language

    def compute(
        self,
        input_data: np.ndarray,
        **kwargs,
    ) -> MetricResult:
        """Computes the identifiability of a text dataset.

        Parameters
        ----------
        input_data : np.ndarray
            Array of dictionaries with the text samples to evaluate. Each dictionary must have an "id", "text" and "language" key.

        Returns
        -------
        result : MetricResult
            A dictionary with the dataset-level and instance-level identifiability scores.
        """
        scores = []
        errors = []
        for entry in input_data:
            assert "text" in entry, f"Entry {entry['id']} does not have text."
            assert "language" in entry, f"Entry {entry['id']} does not have language specifier."

            language = entry["language"] if self.language is None else self.language
            if language == "es" and self.es_anonymizer is not None:
                result = self.es_anonymizer.metric(
                    content_to_anonymize=entry["text"],
                )
            elif language == "en" and self.eng_anonymizer is not None:
                result = self.eng_anonymizer.metric(
                    content_to_anonymize=entry["text"],
                )
            else:
                errors.append(
                    f"Entry with id {entry['id']} does not have valid language specifier or language model was not initiated. Skipping.",
                )
            scores.append(result)

        return MetricResult(
            dataset_level={"dtype": OutputsTypes.NUMERIC, "subtype": "int", "value": sum(scores)},
            instance_level={"dtype": OutputsTypes.ARRAY, "subtype": "int", "value": scores},
            errors=errors if len(errors) > 0 else None,
        )


__all__ = ["Identifiability"]
