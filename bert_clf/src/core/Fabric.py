from typing import Dict, Any

from bert_clf.src.core import BaseCLF
from bert_clf.src.models import BertCLF, EncoderCLF


class CLFFabric:

    @staticmethod
    def getter(
            pretrained_model_name: str,
            id2label: Dict[int, str],
            dropout: float,
            config: Dict[str, Any]
    ) -> BaseCLF:

        assert isinstance(config['transformer_model']['encoder'], bool), "encoder parameter should be of boolean type"

        if config['transformer_model']['encoder'] is True:
            model = EncoderCLF(
                pretrained_model_name=pretrained_model_name,
                id2label=id2label,
                dropout=dropout,
            )
            return model

        model = BertCLF(
                pretrained_model_name=pretrained_model_name,
                id2label=id2label,
                dropout=dropout,
            )
        return model
