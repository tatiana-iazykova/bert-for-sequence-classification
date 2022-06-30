import torch
from typing import List, Union, Tuple, Any
import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            maxlen: int,
            texts: Union[List[str], np.ndarray],
            targets: Union[List[Any], np.ndarray]
    ):
        self.tokenizer = tokenizer
        self.texts = [torch.LongTensor(self.tokenizer.encode(
            t,
            truncation=True,
            max_length=maxlen
        )) for t in texts]
        self.texts = torch.nn.utils.rnn.pad_sequence(self.texts, batch_first=True,
                                                     padding_value=self.tokenizer.pad_token_id)

        self.length = len(texts)

        self.target = torch.LongTensor(targets)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        ids = self.texts[index]
        y = self.target[index]

        return ids, y
