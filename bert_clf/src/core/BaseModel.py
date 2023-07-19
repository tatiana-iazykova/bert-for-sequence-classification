from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from abc import abstractmethod


class BaseCLF(nn.Module):

    def __init__(
            self,
            pretrained_model_name: str,
            id2label: Dict[int, str],
            dropout: float,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name)
        self.pretrained_model = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name)
        self.drop = nn.Dropout(dropout)
        self.act = nn.LogSoftmax(1)
        self.mapper = id2label


    @abstractmethod
    def forward(self, texts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _predict(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer.encode(text, return_tensors="pt", truncation=True)
        outputs = self(inputs.to(self.pretrained_model.device))
        return outputs

    def predict(self, text: str) -> str:
        self.eval()
        with torch.no_grad():
            outputs = self._predict(text=text)
            pred = outputs.argmax(1).item()
            pred_text = self.mapper[pred]
        return pred_text

    def predict_proba(self, text: str) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            outputs = self._predict(text=text)
            probas = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy().tolist()[0]
            probas_dict = {self.mapper[i]: proba for i, proba in enumerate(probas)}
        return probas_dict
