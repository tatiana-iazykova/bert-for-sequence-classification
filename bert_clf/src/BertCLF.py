from typing import Dict

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class BertCLF(nn.Module):

    def __init__(
            self,
            pretrained_model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase,
            id2label: Dict[int, str],
            dropout: float,
            device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrained_model = pretrained_model
        self.drop = nn.Dropout(dropout)
        self.act = nn.LogSoftmax(1)
        self.mapper = id2label
        self.device = device

        out_bert = self.pretrained_model.config.hidden_size
        self.fc = nn.Linear(out_bert, len(self.mapper))

    def forward(self, texts: torch.Tensor) -> torch.Tensor:
        mask = (texts != self.tokenizer.pad_token_id).long()

        hidden = self.pretrained_model(texts, attention_mask=mask)[0]

        dense_outputs = self.fc(self.drop(hidden[:, 0]))
        outputs = self.act(dense_outputs)

        return outputs

    def _predict(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer.encode(text, return_tensors="pt", truncation=True)
        outputs = self(inputs)
        return outputs

    def predict(self, text: str) -> str:
        outputs = self._predict(text=text)
        pred = outputs.argmax(1).item()
        pred_text = self.mapper[pred]
        return pred_text

    def predict_proba(self, text: str) -> Dict[str, float]:
        outputs = self._predict(text=text)
        probas = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().tolist()
        probas_dict = {self.mapper[i]: proba for i, proba in enumerate(probas)}
        return probas_dict
