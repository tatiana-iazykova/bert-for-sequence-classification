import json
import os
from typing import Dict, Optional

import torch
from transformers import AutoModel, AutoConfig

from bert_clf.src.core import BaseCLF
import torch.nn as nn


class BertCLF(BaseCLF):

    def __init__(
            self,
            pretrained_model_name: str,
            id2label: Optional[Dict[int, str]] = None,
            dropout: Optional[float] = 1e-6,
    ):
        super().__init__(pretrained_model_name=pretrained_model_name, dropout=dropout)

        if os.path.exists(pretrained_model_name):
            self.pretrained_model = AutoModel.from_config(
                AutoConfig.from_pretrained(self.tokenizer.name_or_path)
            )
            out_bert = self.pretrained_model.config.hidden_size

            with open(os.path.join(pretrained_model_name, "id2label.json")) as f:
                self.mapper = json.load(f)
                self.mapper = {int(k): v for k, v in self.mapper.items()}
            self.fc = nn.Linear(out_bert, len(self.mapper))
            self.load_state_dict(
                torch.load(
                    os.path.join(pretrained_model_name, "state_dict.pth"), map_location='cpu'
                )
            )

        else:
            self.mapper = id2label
            self.pretrained_model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name)
            out_bert = self.pretrained_model.config.hidden_size
            self.fc = nn.Linear(out_bert, len(self.mapper))

    def forward(self, texts: torch.Tensor) -> torch.Tensor:
        mask = (texts != self.tokenizer.pad_token_id).long()

        hidden = self.pretrained_model(texts, attention_mask=mask)[0]

        dense_outputs = self.fc(self.drop(hidden[:, 0]))
        outputs = self.act(dense_outputs)

        return outputs
