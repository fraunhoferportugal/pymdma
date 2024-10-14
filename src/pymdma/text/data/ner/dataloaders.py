from datasets import Dataset, concatenate_datasets
from torch.utils import data
from transformers import DataCollatorForTokenClassification


class DataLoaderForTokenClassification:
    def __init__(
        self,
        ner_data_generator: object,
        tokenizer: object,
        processed_data_dir_name: str = "splitted_dataset",
    ):
        super().__init__()
        self.ner_data_generator = ner_data_generator
        self.tokenizer = tokenizer

    def remove_sentence_column(self, dataset: Dataset):
        """Removes the sentence column from the dataset."""
        return dataset.remove_columns("sentences")

    def _concat_datasets(self):
        return concatenate_datasets([Dataset.from_pandas(dataset) for dataset in self.ner_data_generator])

    def get_data_collator(self):
        """Initializes the data collator for token classification.

        Returns:
            object: Data collator for token classification.
        """
        return DataCollatorForTokenClassification(tokenizer=self.tokenizer)

    def get_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = True,
        workers: int = 1,
        pin_memory: bool = True,
        drop_last: bool = True,
        chunked_dataset: bool = True,
    ):
        """Creates a dataloader.

        Args:
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle data samples or not.
            workers (int): Number of workers requested. Defaults to 1.
            pin_memory (bool): Pin memory. Defaults to True.
            drop_last (bool): Whether to drop the last batch or not. Defaults to True.
            chunked_dataset(bool): Whether or not to process the dataloader in batches.

        Returns:
            object: Dataloader.
        """

        if chunked_dataset:
            full_dataset = self._concat_datasets()
        else:
            full_dataset = self.ner_data_generator
        full_dataset = self.remove_sentence_column(full_dataset)
        data_collator = self.get_data_collator()

        return data.DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=workers,
            drop_last=drop_last,
            collate_fn=data_collator,
        )
