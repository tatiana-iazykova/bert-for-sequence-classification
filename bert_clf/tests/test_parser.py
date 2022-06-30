from bert_clf.pipeline import train
from bert_clf.utils import load_config
from bert_clf import prepare_data, prepare_data_notebook
from unittest import TestCase
from pathlib import Path
import torch
from parameterized import parameterized
import pandas as pd
import json
import os


class TestDocParser(TestCase):

    request_data_dir = Path(__file__).parent / "request"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(
        "results/model.pth", map_location=device
    )

    model.eval()
    config = load_config("base_case_config.yaml")

    @parameterized.expand([
        "test.csv",
    ])
    def test_pred(self, test_id):

        true = pd.read_csv(f'response/{test_id}')["target"].to_list()
        df = pd.read_csv(f'data/{test_id}')
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

        with open('response/id2label_df.json') as f:
            id2label_true = json.load(f)

        with open('response/train_texts_df.json') as f:
            train_texts_true = json.load(f)

        with open('response/valid_texts_df.json') as f:
            valid_texts_true = json.load(f)

        with open('response/train_targets_df.json') as f:
            train_targets_true = json.load(f)

        with open('response/valid_targets_df.json') as f:
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
        with open('response/id2label.json') as f:
            id2label_true = json.load(f)

        with open('response/train_texts.json') as f:
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

        model_trained = torch.load(
            os.path.join(self.config['training']['output_dir'], 'model.pth'), map_location=self.device
        )

        model_trained.eval()

        model_trained_weights = model_trained.state_dict()

        for k in model_weights:
            self.assertTrue(torch.allclose(model_weights[k], model_trained_weights[k]))

        os.remove(os.path.join(self.config['training']['output_dir'], 'model.pth'))
