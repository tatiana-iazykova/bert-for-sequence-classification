from typing import Dict

import torch

from bert_clf.src.core import BaseCLF
import torch.nn as nn


class BertCLF(BaseCLF):

    def __init__(
            self,
            pretrained_model_name: str,
            id2label: Dict[int, str],
            dropout: float,
    ):
        super().__init__(pretrained_model_name=pretrained_model_name, id2label=id2label, dropout=dropout)
        out_bert = self.pretrained_model.config.hidden_size
        self.fc = nn.Linear(out_bert, len(self.mapper))

    def forward(self, texts: torch.Tensor) -> torch.Tensor:
        mask = (texts != self.tokenizer.pad_token_id).long()

        hidden = self.pretrained_model(texts, attention_mask=mask)[0]

        dense_outputs = self.fc(self.drop(hidden[:, 0]))
        outputs = self.act(dense_outputs)

        return outputs
