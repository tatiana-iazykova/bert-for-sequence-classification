import json
import os
from typing import Dict, Optional

import requests
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from abc import abstractmethod


class BaseCLF(nn.Module):

    def __init__(
            self,
            pretrained_model_name: str,
            dropout: Optional[float] = 1e-6,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name)
        self.drop = nn.Dropout(dropout)
        self.act = nn.LogSoftmax(1)

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

    def save_pretrained(self, path: str):
        self.tokenizer.save_pretrained(path)
        self.pretrained_model.config.save_pretrained(path)
        torch.save(
            self.state_dict(),
            os.path.join(path, "state_dict.pth")
        )
        with open(os.path.join(path, 'id2label.json'), mode='w', encoding='utf-8') as f:
            json.dump(self.mapper, f, indent=4, ensure_ascii=False)
