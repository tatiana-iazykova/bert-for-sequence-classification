import torch
import torch.nn as nn
from typing import Dict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel


class BertCLF(nn.Module):

    def __init__(
            self,
            pretrained_model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase,
            id2label: Dict[int, str],
            dropout: float,
            tiny: bool = True,
            device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.pretrained_model = pretrained_model
        self.drop = nn.Dropout(dropout)
        self.act = nn.LogSoftmax(1)
        self.mapper = id2label
        self.device = device

        out_bert = 768
        if tiny:
            out_bert = 312
        self.fc = nn.Linear(out_bert, len(self.mapper))

    def forward(self, texts: torch.Tensor) -> torch.Tensor:
        mask = (texts != self.tokenizer.pad_token_id).long()

        hidden = self.pretrained_model(texts, attention_mask=mask)[0]

        dense_outputs = self.fc(self.drop(hidden[:, 0]))
        outputs = self.act(dense_outputs)

        return outputs

    def predict(self, text: str) -> str:
        inputs = self.tokenizer.encode(text, return_tensors="pt", truncation=True)
        outputs = self(inputs)
        pred = outputs.argmax(1).item()
        pred_text = self.mapper[pred]
        return pred_text
