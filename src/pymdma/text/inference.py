import random
from typing import Union

import numpy as np
import torch
from accelerate import Accelerator
from loguru import logger
from tqdm import tqdm
from transformers import BertForTokenClassification, BertTokenizerFast

from .data.ner.dataloaders import DataLoaderForTokenClassification
from .data.ner.preprocess import PreProcessorForNERInference


class TransformerInferenceAgentForNER:
    def __init__(
        self,
        text_to_anonymize: Union[str, list],
        model: BertForTokenClassification,
        tokenizer: BertTokenizerFast,
        accelerator: Accelerator,
        batch_size: int = 1,
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lowercase: bool = True,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.text_to_anonymize = text_to_anonymize
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.accelerator = accelerator
        self.lowercase = lowercase

    def establish_inference_environment(self):
        """Sets up the environment for inference."""
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def preprocess_input_data(self):
        """Processes the data provided as input.

        Returns:
            tuple: A NER dataset for inference, the word spans and the word ids associated with each word and token, respectively, and the doc_ids linking each sentence with its original document.
        """
        preprocessor = PreProcessorForNERInference(self.tokenizer, self.max_length, lowercase=self.lowercase)
        preprocessed_data, word_spans, word_ids, doc_ids, sentence_ids = preprocessor.preprocess_for_inference(
            self.text_to_anonymize,
        )

        return preprocessed_data, word_spans, word_ids, doc_ids, sentence_ids

    def build_dataloader(self, preprocessed_inference_data):
        """Builds a dataloader using the data provided for inference.

        Args:
            preprocessed_inference_data (object): NER dataset for inference.

        Returns:
            object: Dataloader.
        """
        dataloader_builder = DataLoaderForTokenClassification(preprocessed_inference_data, self.tokenizer)
        inference_dataloader = dataloader_builder.get_dataloader(
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            chunked_dataset=False,
            workers=0,
        )

        return inference_dataloader

    def get_ner_predictions(self):
        """Processes the data and produces NER predictions for each sample.

        Returns:
            list: NER predictions.
        """
        self.establish_inference_environment()

        logger.info("Pre-processing data...")
        processed_data, word_spans, word_ids, doc_ids, sentence_ids = self.preprocess_input_data()
        logger.info("Pre-processing complete.")

        logger.info("Building dataloader...")
        inference_dataloader = self.build_dataloader(processed_data)
        logger.info("Dataloader complete.")

        logger.info("Accelerating...")
        # self.model, self.tokenizer, inference_dataloader = self.accelerator.prepare(
        #     self.model,
        #     self.tokenizer,
        #     inference_dataloader,
        # )
        logger.info("Elements accelerated.")

        logger.info("Generating predictions...")
        predictor = TransformerPredictorForNER(self.model, self.device)
        predictions = predictor.get_ner_predictions(
            inference_dataloader,
            processed_data["sentences"],
            word_ids,
            word_spans,
            doc_ids,
            sentence_ids,
            self.batch_size,
        )
        logger.info("Predictions generated.")

        return predictions


class TransformerPredictorForNER:
    def __init__(
        self,
        ner_model: object,
        device: str,
    ):
        super().__init__()
        self.ner_model = ner_model
        self.device = device

    def structure_ner_predictions(self, original_text, word_predictions, word_scores, spans):
        """Takes word-level predictions and structures them into a traditional
        NER format.

        Args:
            original_text (list): Tokenized sentence.
            word_predictions (list): List of word-level predictions.
            word_scores (list): List of word-level confidence scores.
            spans (list): List of spans associated with each word of the original text.

        Returns:
            list: Final predictions in typical NER fashion.
        """
        final_predictions = []
        for idx, word in enumerate(original_text):
            if word_predictions[idx] not in ["0"]:
                prediction = {
                    "TEXT": word,
                    "START": spans[idx][0],
                    "END": spans[idx][1],
                    "SCORE": word_scores[idx],
                    "CATEGORY": word_predictions[idx],
                }
                final_predictions.append(prediction)

        return final_predictions

    def get_word_level_predictions(self, token_predictions, token_scores, original_text, word_ids, word_spans):
        """Takes token-level predictions and converts them to word-level
        predictions.

        Args:
            token_predictions (list): List of of token-level predictions.
            token_scores (list): List of token-level confidence scores.
            original_text (list): Tokenized sentence.
            word_ids (list): List of word ids associated with each token classified.
            word_spans (list): List of word spans associated with the original text.

        Returns:
            list: List of structured, word-level predictions.
        """
        seen_word_ids = []
        word_predictions = []
        word_scores = []
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id not in seen_word_ids:
                    seen_word_ids.append(word_id)
                    word_predictions.append(
                        token_predictions[idx],
                    )  # We are currently ignoring post-first-token predictions. Is this wise?
                    word_scores.append(token_scores[idx])

        return self.structure_ner_predictions(original_text, word_predictions, word_scores, word_spans)

    def get_ner_predictions(self, test_data, original_text, word_ids, word_spans, doc_ids, sentence_ids, batch_size):
        """Produces predictions for a given batch of inference samples and
        converts them into a typical word-level NER format.

        Args:
            test_data (object): Inference dataloader.
            original_text (list): Original content provided for inference.
            word_ids (list): Word ids mapping each token to its respective word for every sample of the original content.
            word_spans (list): Word spans for every sample of the original content.
            doc_ids (list): IDs linking each sentence to its original document.
            sentence_ids (list): IDs linking each text to its original sentence.
            batch_size (int): Number of samples per batch.

        Returns:
            list: Final NER predictions.
        """
        self.ner_model.eval()

        all_predictions = []
        logger.info("Inference ongoing...")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Number of batches: {len(test_data)}")
        logger.info(f"Number of samples: {test_data}")

        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(test_data), "Inference ongoing...", colour="blue"):

                batch.to(self.device)

                logits = self.ner_model(**batch)["logits"]

                probabilities = torch.nn.functional.softmax(logits, dim=2)
                token_predictions = probabilities.argmax(dim=2).tolist()
                token_scores = probabilities.max(dim=2)[0].tolist()

                token_predictions = [
                    [self.ner_model.config.id2label[value] for value in predictions]
                    for predictions in token_predictions
                ]

                for predictions_idx, predictions in enumerate(token_predictions):
                    sample_idx = batch_idx * batch_size + predictions_idx
                    word_level_predictions = self.get_word_level_predictions(
                        predictions,
                        token_scores[predictions_idx],
                        original_text[sample_idx],
                        word_ids[sample_idx],
                        word_spans[sample_idx],
                    )

                    word_level_predictions.append({"SENTENCE": sentence_ids[sample_idx]})
                    word_level_predictions.append({"DOCUMENT": doc_ids[sample_idx]})

                    all_predictions.append(word_level_predictions)

        return all_predictions
