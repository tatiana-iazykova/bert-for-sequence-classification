import shutil

from bert_clf.pipeline import train
from bert_clf.utils import load_config
from bert_clf import prepare_data, prepare_data_notebook, EncoderCLF, BertCLF
from unittest import TestCase
from pathlib import Path
import torch
from parameterized import parameterized
import pandas as pd
import json


class TestDocParser(TestCase):
    request_data_dir = Path(__file__).parent / "request"
    response_data_dir = Path(__file__).parent / "response"

    device = "cpu"

    model_path = Path(__file__).parent / "results"
    model = BertCLF(model_path.as_posix())

    model.eval()
    config_path = Path(__file__).parent / "base_case_config.yaml"
    config = load_config(config_path.as_posix())

    @parameterized.expand([
        "test.csv",
    ])
    def test_pred(self, test_id):

        true = pd.read_csv(self.response_data_dir / test_id)["target"].to_list()
        df = pd.read_csv(Path(__file__).parent / "data" / test_id)
        pred = df["text"].apply(self.model.predict).to_list()

        self.assertListEqual(
            true,
            pred
        )

    def test_prepare_df(self):

        df = pd.DataFrame({
            'text': ['aaaaa', 'bbbbb', 'cccc', 'ddddd'],
            'target': [1, 1, 0, 0]
        })

        id2label_, train_texts, valid_texts, train_targets, valid_targets = prepare_data_notebook(
            config=self.config,
            train_df=df
        )

        with open(self.response_data_dir / 'id2label_df.json') as f:
            id2label_true = json.load(f)

        with open(self.response_data_dir / 'train_texts_df.json') as f:
            train_texts_true = json.load(f)

        with open(self.response_data_dir / 'valid_texts_df.json') as f:
            valid_texts_true = json.load(f)

        with open(self.response_data_dir / 'train_targets_df.json') as f:
            train_targets_true = json.load(f)

        with open(self.response_data_dir / 'valid_targets_df.json') as f:
            valid_targets_true = json.load(f)

        id2label = {}
        for k, v in id2label_.items():
            id2label[str(k)] = int(v)

        true = [id2label_true, train_texts_true, valid_texts_true, train_targets_true, valid_targets_true]
        pred = [id2label, train_texts, valid_texts, train_targets, valid_targets]

        self.assertListEqual(
            true,
            pred
        )

    def test_dataframe(self):
        with open(Path(__file__).parent / 'response/id2label.json') as f:
            id2label_true = json.load(f)

        with open(Path(__file__).parent / 'response/train_texts.json') as f:
            train_texts_true = json.load(f)

        with open('response/valid_texts.json') as f:
            valid_texts_true = json.load(f)

        with open('response/train_targets.json') as f:
            train_targets_true = json.load(f)

        with open('response/valid_targets.json') as f:
            valid_targets_true = json.load(f)

        id2label_, train_texts, valid_texts, train_targets, valid_targets = prepare_data(config=self.config)

        id2label = {}
        for k, v in id2label_.items():
            id2label[str(k)] = int(v)

        true = [id2label_true, train_texts_true, valid_texts_true, train_targets_true, valid_targets_true]
        pred = [id2label, train_texts, valid_texts, train_targets, valid_targets]

        self.assertListEqual(
            true,
            pred
        )

    def test_model(self):
        train("base_case_config.yaml")

        model_weights = self.model.state_dict()

        config = load_config("base_case_config.yaml")
        model_trained = BertCLF(config['training']['output_dir'])

        model_trained.eval()

        model_trained_weights = model_trained.state_dict()

        for k in model_weights:
            self.assertTrue(torch.allclose(model_weights[k], model_trained_weights[k]))

        shutil.rmtree(config['training']['output_dir'])

    def test_model_encoder(self):
        train("encoder_case_config.yaml")

        config = load_config("encoder_case_config.yaml")
        model_trained = EncoderCLF(config['training']['output_dir'])

        model_trained.eval()

        model_trained_weights = model_trained.state_dict()

        for k in model_trained_weights:
            self.assertTrue(not torch.all(model_trained_weights[k].eq(torch.rand(model_trained_weights[k].shape))))
        shutil.rmtree(config['training']['output_dir'])
