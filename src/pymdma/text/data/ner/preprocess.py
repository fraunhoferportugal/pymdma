from typing import Dict, Union

from datasets import Dataset
from tqdm import tqdm

from ...utils.text_processing import tokenize_text_to_sentences, tokenize_with_indexes


class PreProcessorForNERInference:
    def __init__(
        self,
        tokenizer,
        max_length,
        lowercase: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lowercase = lowercase

    def _lowercase_sentence(self, input_text: str):
        """Based on the value of self.lowercase, it converts the content in
        input_text into lowercase or not."""
        if self.lowercase:
            return input_text.lower()
        else:
            return input_text

    def tokenize_raw_sentence(self, input_text: str):
        """Tokenizes a raw sentence. Each token is a word.

        Args:
            input_text (str): Sentence to be tokenized.

        Returns:
            list: Tokens with spans associated.
        """
        return tokenize_with_indexes(input_text)

    def get_token_classification_dataset_inference(self, sentences_dict: dict):
        """Creates a dataset for token classification inference.

        Args:
            sentences_dict (dict): Dict of sentences, input_ids and attention_masks.

        Returns:
            Dataset: Dataset for NER inference.
        """
        return Dataset.from_dict(
            {
                "sentences": sentences_dict["sentences"],
                "input_ids": sentences_dict["input_ids"],
                "attention_mask": sentences_dict["attention_mask"],
            },
        )

    def tokenize_inputs(self, tokenized_sentences: list):
        """Tokenizes the inference dataset.

        Args:
            tokenized_sentences (list): Tokenized sentences for inference.

        Returns:
            Dataset: Tokenized NER Dataset for inference.
        """
        tokenized_content = {
            "sentences": tokenized_sentences,
            "input_ids": [],
            "attention_mask": [],
            "word_ids": [],
        }

        for sentence in tqdm(tokenized_sentences, desc="Tokenizing sentences for inference...", colour="red"):
            tokenized_inputs = self.tokenizer.__call__(
                sentence,
                truncation=False,
                is_split_into_words=True,
            )

            tokenized_content["input_ids"].append(tokenized_inputs["input_ids"])
            tokenized_content["attention_mask"].append(tokenized_inputs["attention_mask"])
            tokenized_content["word_ids"].append(tokenized_inputs.word_ids())

        return tokenized_content

    def preprocess_sentences(self, input_text: str, doc_id=int):
        """Splits a chunk of text into sentences and tokenizes each sentence,
        saving the spans of each word and the doc_id.

        Args:
            input_text (str): Text to be pre-processed.
            doc_id (int): ID linking each sentence to its respective document. Defaults to int.

        Returns:
            tuple: Tokenized sentences along with the spans of each original word and an id mapping each sentence to the document where it was extracted.
        """
        sentences = tokenize_text_to_sentences(input_text)

        tokenized_raw_sentences = []
        spans = []
        doc_ids = []
        for sentence in sentences:
            doc_ids.append(doc_id)
            sentence = self._lowercase_sentence(sentence)
            tokenized_content = self.tokenize_raw_sentence(sentence)
            tokenized_raw_sentences.append([token[0] for token in tokenized_content])
            spans.append([(token[1], token[2]) for token in tokenized_content])

        return tokenized_raw_sentences, spans, doc_ids

    def get_non_intermediate_token_id(self, word_ids: list, split_token_id: int):
        """Checks whether a certain split token is an intermediate token of a
        word. If so, returns a non-intermediate token. If not, returns the same
        token.

        Args:
            word_ids (list): Word ids list.
            split_token_id (int): Token where the split is set to occur.

        Returns:
            int: Token where the split is set to occur (correct).
        """
        while word_ids[split_token_id] == word_ids[split_token_id - 1]:
            split_token_id -= 1

        return split_token_id

    def split_long_sentences(self, input_sentences: dict, spans: list, doc_ids: list, max_length: int = 100):
        new_samples_dict: Dict[str, list] = {}
        for key in input_sentences.keys():
            new_samples_dict[key] = []

        new_spans = []
        new_doc_ids = []
        sentence_ids = []

        for idx in tqdm(range(len(input_sentences["sentences"])), desc="Splitting long sentences...", colour="yellow"):
            sentence_length = len(input_sentences["input_ids"][idx])  # number of tokens in the current sentence
            word_counter = 0
            if sentence_length > max_length:
                temp_dict = {}
                for key in input_sentences.keys():
                    temp_dict[key] = input_sentences[key][idx]
                temp_spans = spans[idx]

                while len(temp_dict["input_ids"]) > max_length:
                    split_token_id = self.get_non_intermediate_token_id(
                        temp_dict["word_ids"],
                        max_length,
                    )

                    new_samples_dict["input_ids"].append(temp_dict["input_ids"][0:split_token_id])
                    new_samples_dict["attention_mask"].append(temp_dict["attention_mask"][0:split_token_id])
                    new_samples_dict["word_ids"].append(temp_dict["word_ids"][0:split_token_id])
                    first_word_id = temp_dict["word_ids"][0]
                    last_word_id = new_samples_dict["word_ids"][-1][-1] + 1
                    new_samples_dict["sentences"].append(temp_dict["sentences"][first_word_id:last_word_id])
                    word_counter += len(temp_dict["sentences"][first_word_id:last_word_id])
                    new_spans.append(temp_spans[first_word_id:last_word_id])
                    new_doc_ids.append(doc_ids[idx])
                    sentence_ids.append(idx)

                    # remove processed content from the temporary variables
                    temp_dict["word_ids"] = temp_dict["word_ids"][split_token_id::]
                    temp_dict["attention_mask"] = temp_dict["attention_mask"][split_token_id::]
                    temp_dict["input_ids"] = temp_dict["input_ids"][
                        split_token_id::
                    ]  # Care! This absolutely need to be the last thing being updated, since the loop is dependent on this variable

                # Add the content from the last sentence
                new_samples_dict["input_ids"].append(temp_dict["input_ids"])
                new_samples_dict["attention_mask"].append(temp_dict["attention_mask"])
                new_samples_dict["word_ids"].append(temp_dict["word_ids"])
                new_samples_dict["sentences"].append(temp_dict["sentences"][last_word_id::])
                word_counter += len(temp_dict["sentences"][last_word_id::])
                new_spans.append(temp_spans[last_word_id::])
                new_doc_ids.append(doc_ids[idx])
                sentence_ids.append(idx)

            else:
                for key in input_sentences.keys():
                    new_samples_dict[key].append(input_sentences[key][idx])
                new_spans.append(spans[idx])
                new_doc_ids.append(doc_ids[idx])
                sentence_ids.append(idx)

        return new_samples_dict, new_spans, new_doc_ids, sentence_ids

    def preprocess_for_inference(self, input_content: Union[str, list]):
        """Preprocesses a NER Dataset for inference.

        Args:
            input_content (Union[str, list]): Input content. It can be either a single string, or a list of strings.

        Returns:
            tuple: Tokenized NER Dataset for inference along with the spans of each original word, the ids matching each token to a word, and the ids mapping each sentence of the dataset to the document where it was extracted.
        """
        if isinstance(input_content, str):
            all_tokenized_raw_sentences, all_spans, all_doc_ids = self.preprocess_sentences(input_content, 0)
        else:
            all_tokenized_raw_sentences = []
            all_spans = []
            all_doc_ids = []
            for doc_id, document in tqdm(enumerate(input_content), desc="Preprocessing notes for inference..."):
                tokenized_raw_sentences, spans, doc_ids = self.preprocess_sentences(document, doc_id)
                all_tokenized_raw_sentences.extend(tokenized_raw_sentences)
                all_spans.extend(spans)
                all_doc_ids.extend(doc_ids)

        tokenized_content = self.tokenize_inputs(all_tokenized_raw_sentences)
        splitted_content, all_spans, all_doc_ids, all_sentence_ids = self.split_long_sentences(
            tokenized_content,
            all_spans,
            all_doc_ids,
            max_length=self.max_length,
        )
        word_ids = splitted_content["word_ids"]

        token_classification_dataset = self.get_token_classification_dataset_inference(splitted_content)

        return token_classification_dataset, all_spans, word_ids, all_doc_ids, all_sentence_ids
