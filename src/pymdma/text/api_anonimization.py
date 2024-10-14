import os
from collections import Counter
from typing import Union

import torch
from accelerate import Accelerator
from config import data_raw_dir, package_base_dir
from loguru import logger
from tqdm import tqdm

from .inference import TransformerInferenceAgentForNER
from .model_inference import ModelLoader
from .utils.anonymization_utils import custom_anonymizer
from .utils.text_processing import tokenize_text_to_sentences


class TextAnonymizer:
    def __init__(
        self,
        repo_name: str = "bert-base-cased",
        tokens_per_sample: int = 128,
        replacement_strategy: str = "category",
    ):
        """Implements a text anonymizer based on a pretrained transformer-based
        model. Model is loaded from disk.

        Args:
            repo_name (str, optional): folder where model is stored
            tokens_per_sample (int, optional): number of tokens per sample
            replacement_strategy (str, optional): if 'category' replace with category, else replace entities with [REDACTED]
        """
        super().__init__()
        self.metadata_dir = data_raw_dir
        self.repo_name = repo_name
        self.model_name = repo_name.split("/")[-1]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = os.path.join(
            package_base_dir,
            "text/models",
            self.model_name,
        )  # os.path.join(models_dir, self.model_name)
        self.tokens_per_sample = tokens_per_sample
        self.replacement_strategy = replacement_strategy
        self.lowercase = True
        self.dict_name = "categories.json"
        self.dict_dir = self.repo_name
        self.accelerator = Accelerator(mixed_precision="fp16")
        if self.device == "cuda":

            self.device = self.accelerator.device

        self.tokenizer, self.model = self._initialize_transformer_tokenizer_and_model()

    def _initialize_transformer_tokenizer_and_model(self):
        """Initializes the tokenizer and the model to be used for inference."""
        return ModelLoader(
            repo_name=self.repo_name,
            device=self.device,
            dict_name=self.dict_name,
            dict_dir=self.dict_dir,
        ).load_model_for_inference(f"{self.model_name}_model_500_epochs.pt")

    def _assert_list_input(self, input_content: Union[list, str]):
        """Asserts that the input content is a list of strings."""

        if isinstance(input_content, str):
            return [input_content]
        else:
            return input_content

    def _get_most_prevalent_term(self, input_list: list):

        term_counts = Counter(input_list)
        most_common_term, _ = term_counts.most_common(1)[0]

        return most_common_term

    def _collapse_tags(self, nes: list):
        """Merges tags corresponding to the same entity.

        Args:
            nes (list): list of metadata refering to the named entities identified.

        Returns:
            list: collapsed list of named entities
        """
        collapsed_nes_list = []

        for nes_list in nes:
            current_entity = None
            merged_tags = []
            categories = []
            for tag in sorted(nes_list, key=lambda x: x["START"]):
                categories.append(tag["CATEGORY"][2::])
                if current_entity is None or tag["START"] > current_entity["END"] + 1:
                    if current_entity is not None:
                        categories = [tag["CATEGORY"][2::]]

                    # Start a new entity
                    current_entity = tag.copy()
                    merged_tags.append(current_entity)

                else:
                    # Merge with the current entity
                    current_entity["TEXT"] += " " + tag["TEXT"]
                    current_entity["END"] = tag["END"]
                    current_entity["CATEGORY"] = self._get_most_prevalent_term(
                        categories,
                    )

            collapsed_nes_list.append(merged_tags)

        return collapsed_nes_list

    def _join_predictions_for_the_same_sentence(self, found_entities: list):
        joined_found_entities: list = []
        sentence = -1
        for prediction in found_entities:
            if sentence == prediction[-1]["SENTENCE"]:
                joined_found_entities[-1].extend(prediction[0:-1])
            else:
                joined_found_entities.append(prediction[0:-1])
            sentence = prediction[-1]["SENTENCE"]

        return joined_found_entities

    def _get_named_entities_with_transformer(
        self,
        text_to_anonymize: Union[str, list],
        batch_size: int,
    ):
        """Uses a custom NER transformer-based model to identify the sensitive
        information in text_to_anonymize.

        Args:
            text_to_anonymize (Union[str, list]): Content to be anonymized.
            batch_size (int): Number of samples per batch.

        Returns:
            list: list of sensitive entities found in the text(s) given as input.
        """
        inference_agent = TransformerInferenceAgentForNER(
            text_to_anonymize,
            self.model,
            self.tokenizer,
            device=self.device,
            max_length=self.tokens_per_sample,
            batch_size=batch_size,
            accelerator=self.accelerator,
            lowercase=self.lowercase,
        )
        ner_predictions = inference_agent.get_ner_predictions()

        return ner_predictions

    def _anonymize(self, text_to_anonymize: str, found_nes: list):
        """Anonymizes the text in text_to_anonymize.

        Args:
            text_to_anonymize (str): Text to be anonymized.
            found_nes (list): List of sensitive entities found in the text.

        Returns:
            str: Anonymized version of the text.
        """

        anonymized_text = custom_anonymizer(
            text_to_anonymize,
            found_nes,
            replacement_strategy=self.replacement_strategy,
        )

        return anonymized_text

    def anonymize_text(self, text_to_anonymize: str, found_nes: list):
        found_nes = self._collapse_tags(found_nes)
        sentences = tokenize_text_to_sentences(text_to_anonymize)
        anonymized_text = ""
        for sentence_id, sentence in enumerate(sentences):
            anonymized_sentence = self._anonymize(sentence, found_nes[sentence_id])
            anonymized_text = anonymized_text + anonymized_sentence + "\n"

        return anonymized_text

    def _get_named_entities(self, samples: list, batch_size: int):

        return self._get_named_entities_with_transformer(
            samples,
            batch_size=batch_size,
        )

    def _anonymize_with_ner(self, content_to_anonymize: list, batch_size: int):
        found_nes = self._get_named_entities(
            content_to_anonymize,
            batch_size=batch_size,
        )

        anonymized_content = []
        for doc_id, doc_to_anonymize in tqdm(
            enumerate(content_to_anonymize),
            desc="Anonymizing data...",
            colour="magenta",
        ):
            doc_nes = [ne[0:-1] for ne in found_nes if ne[-1]["DOCUMENT"] == doc_id]
            logger.info(f"Found {len(doc_nes)} named entities in document {doc_id}")
            doc_nes = self._join_predictions_for_the_same_sentence(doc_nes)
            anonymized_content.append(self.anonymize_text(doc_to_anonymize, doc_nes))

        return anonymized_content

    def metric(
        self,
        content_to_anonymize: Union[str, list],
        batch_size: int = 1,
    ):
        content_to_anonymize = self._assert_list_input(content_to_anonymize)
        found_nes = self._get_named_entities(
            content_to_anonymize,
            batch_size=batch_size,
        )

        anonymized_content = []
        for doc_id, doc_to_anonymize in tqdm(
            enumerate(content_to_anonymize),
            desc="Anonymizing data...",
            colour="magenta",
        ):
            doc_nes = [ne[0:-1] for ne in found_nes if ne[-1]["DOCUMENT"] == doc_id]
            logger.info(f"Found {len(doc_nes)} named entities in document {doc_id}")
            anonymized_content.append(len(doc_nes))

        # return sum of all nes
        return sum(anonymized_content)

    def anonymize_content(
        self,
        content_to_anonymize: Union[str, list],
        batch_size: int = 1,
    ):
        """Anonymizes the content given as input, according to a pre-defined
        method.

        Args:
            content_to_anonymize (Union[str, list]): Text or list of texts given as input to be anonymized.
            batch_size (int): Number of samples to compose each inference batch. Defaults to 1.

        Returns:
            list: Anonymized version of the text or texts given as input.
        """
        content_to_anonymize = self._assert_list_input(content_to_anonymize)

        return self._anonymize_with_ner(content_to_anonymize, batch_size)
