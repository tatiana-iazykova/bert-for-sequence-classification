import torch
from typing import List, Union, Tuple, Any
import numpy as np
from transformers import PreTrainedTokenizerBase


class Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            texts: Union[List[str], np.ndarray],
            targets: Union[List[Any], np.ndarray]
    ):

        self.texts = texts
        self.targets = targets
        self.length = len(texts)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        ids = self.texts[index]
        y = self.targets[index]

        return ids, y


class Collator():

    def __init__(self, maxlen: int, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def collate(self, batch: Tuple[List[str], List[int]]) -> Tuple[torch.LongTensor, torch.LongTensor]:
        texts, targets = zip(*batch)

        texts = self.tokenizer(
            list(texts), max_length=self.maxlen, truncation=True, padding=True, return_tensors='pt')
        targets = torch.LongTensor(targets)

        return texts['input_ids'], targets
