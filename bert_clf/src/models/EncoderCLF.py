import json
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import wget
from transformers import AutoModel, AutoConfig

from bert_clf.src.core import BaseCLF
from bert_clf.src.core.utils import SUPPORTED_MODELS


class EncoderCLF(BaseCLF):

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
            ).encoder
            out = self.pretrained_model.config.d_model

            with open(os.path.join(pretrained_model_name, "id2label.json")) as f:
                self.mapper = json.load(f)
                self.mapper = {int(k): v for k, v in self.mapper.items()}
            self.fc = nn.Linear(out, len(self.mapper))
            self.load_state_dict(
                torch.load(
                    os.path.join(pretrained_model_name, "state_dict.pth"), map_location='cpu'
                )
            )

        elif pretrained_model_name in SUPPORTED_MODELS:
            self.pretrained_model = AutoModel.from_config(
                AutoConfig.from_pretrained(self.tokenizer.name_or_path)
            ).encoder

            out = self.pretrained_model.config.d_model

            id2label_path = os.path.expanduser("~/.cache/huggingface/language_identification_id2label.json")
            state_dict_path = os.path.expanduser("~/.cache/huggingface/language_identification_state_dict.pth")

            wget.download(
                SUPPORTED_MODELS[pretrained_model_name]['id2label'],
                id2label_path
            )

            with open(id2label_path) as f:
                self.mapper = json.load(f)
                self.mapper = {int(k): v for k, v in self.mapper.items()}
            self.fc = nn.Linear(out, len(self.mapper))

            wget.download(
                SUPPORTED_MODELS[pretrained_model_name]['state_dict'],
                state_dict_path
            )

            self.load_state_dict(torch.load(state_dict_path, map_location='cpu'))

        else:
            self.mapper = id2label
            self.pretrained_model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name).encoder
            out = self.pretrained_model.config.d_model
            self.fc = nn.Linear(out, len(self.mapper))

    def forward(self, texts: torch.Tensor) -> torch.Tensor:
        mask = (texts != self.tokenizer.pad_token_id).long()

        hidden = self.pretrained_model(texts, attention_mask=mask)['last_hidden_state']
        hidden = hidden * mask.unsqueeze(-1)
        hidden = hidden.sum(dim=1) / mask.sum(dim=1).unsqueeze(-1)

        dense_outputs = self.fc(self.drop(hidden))
        outputs = self.act(dense_outputs)

        return outputs
