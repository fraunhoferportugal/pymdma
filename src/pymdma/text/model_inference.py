import os

import torch
from config import data_interim_dir
from loguru import logger
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .utils.read_save_utils import read_json


class ModelLoader:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        repo_name: str = "bert-base-cased",
        dict_dir: str = data_interim_dir,
        dict_name: str = "categories.json",
        n_blocks_to_freeze: int = 0,
    ):
        super().__init__()
        self.device = device
        logger.info(f"Loading model inside model loader init {repo_name}")
        self.repo_name = repo_name
        self.model_name = repo_name.split("/")[-1]
        self.dict_dir = dict_dir
        self.dict_name = dict_name
        self.n_blocks_to_freeze = n_blocks_to_freeze

    def load_mapping_dicts(self):
        """Loads the dicts mapping integer ids to labels and vice-versa.

        Returns:
            tuple: a dict mapping labels to ids and a dict mapping ids to labels.
        """

        labels2ids_dict = read_json(self.dict_dir, self.dict_name)["LABELS"]
        ids2labels_dict = {value: key for key, value in labels2ids_dict.items()}

        return labels2ids_dict, ids2labels_dict

    def load_pretrained_model(self):
        """Loads a pretrained model from huggingface.

        Returns:
            object: The tokenizer associated with the name of the huggingface repository provided.
            object: The model associated with the name of the huggingface repository provided.
        """
        labels2ids, ids2labels = self.load_mapping_dicts()
        logger.info(f"Loading model inside model loader {self.repo_name}")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.repo_name)
        model = AutoModelForTokenClassification.from_pretrained(
            self.repo_name,
            num_labels=len(ids2labels),
            id2label=ids2labels,
            label2id=labels2ids,
            ignore_mismatched_sizes=True,
        )

        return tokenizer, model

    def load_model_for_inference(self, model_file_name):
        """Loads a pre-trained model for NER.

        Args:
            model_file_name (str): Name of the saved model's file.

        Returns:
            tuple: tokenizer and model for NER.
        """
        tokenizer, model = self.load_pretrained_model()
        # save model into a folder with the model name
        state_dict = torch.load(os.path.join(self.repo_name, model_file_name), map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        return tokenizer, model
