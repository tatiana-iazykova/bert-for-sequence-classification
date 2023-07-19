from typing import Dict

import torch
import torch.nn as nn

from bert_clf.src.core import BaseCLF


class EncoderCLF(BaseCLF):

    def __init__(
            self,
            pretrained_model_name: str,
            id2label: Dict[int, str],
            dropout: float,
    ):
        super().__init__(pretrained_model_name=pretrained_model_name, id2label=id2label, dropout=dropout)
        out = self.pretrained_model.config.d_model
        self.fc = nn.Linear(out, len(self.mapper))
        self.pretrained_model = self.pretrained_model.encoder

    def forward(self, texts: torch.Tensor) -> torch.Tensor:
        mask = (texts != self.tokenizer.pad_token_id).long()

        hidden = self.pretrained_model(texts, attention_mask=mask)['last_hidden_state']
        hidden = hidden * mask.unsqueeze(-1)
        hidden = hidden.sum(dim=1) / mask.sum(dim=1).unsqueeze(-1)

        dense_outputs = self.fc(self.drop(hidden))
        outputs = self.act(dense_outputs)

        return outputs
