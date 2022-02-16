import os

from src.utils import train, evaluate, predict_metrics
from src.pandas_dataset.PandasDataset import PandasDataset
from src.dataset import Dataset
from src.BertCLF import BertCLF
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch
import torch.optim as optim
import torch.nn as nn
from utils import load_config
import json
from typing import Dict, Any, List, Tuple


def prepare_data(
        config: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[int, str], List[str], List[str], List[int], List[int]]:
    """
    loads data and encodes targets

    :param config: config with  all the necessary information for loading data
    :return: mapper and separated texts and labels
    """

    df = PandasDataset(
        train_data_path=config['data']['train_data_path'],
        test_data_path=config['data']['test_data_path'],
        random_state=config['data']['random_state'],
        text_label=config['data']['text_column'],
        target_label=config['data']['target_column'],
        test_size=config['data']['test_size']
    )

    id2label = {i: l for i, l in enumerate(df.train[config['data']['target_column']].unique())}
    label2id = {l: i for i, l in id2label.items()}
    train_texts = df.train[config['data']['text_column']].to_list()
    valid_texts = df.test[config['data']['text_column']].to_list()
    train_targets = df.train[config['data']['target_column']].map(label2id).to_list()
    valid_targets = df.test[config['data']['target_column']].map(label2id).to_list()

    return id2label, train_texts, valid_texts, train_targets, valid_targets


def prepare_dataset(
        tokenizer: PreTrainedTokenizerBase,
        train_texts: List[str],
        train_targets: List[int],
        valid_texts: List[str],
        valid_targets: List[int],
        config: Dict[str, Dict[str, Any]]
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """

    :param tokenizer: tokenizer from transformers
    :param train_texts: list of texts selected for training
    :param train_targets: list of targets for texts selected for training
    :param valid_texts: list of texts selected for evaluation
    :param valid_targets: list of targets for texts selected for evaluation
    :param config:  config with all the necessary information to set up a model

    :return: Dataloader  objects for training and evaluatuon
    """

    training_set = Dataset(
        tokenizer=tokenizer,
        texts=train_texts,
        targets=train_targets,
        maxlen=config['transformer_model']['maxlen']
    )

    training_generator = torch.utils.data.DataLoader(
        training_set,
        batch_size=config['transformer_model']['batch_size'],
        shuffle=config['transformer_model']['shuffle']
    )

    valid_set = Dataset(
        tokenizer=tokenizer,
        texts=valid_texts,
        targets=valid_targets,
        maxlen=config['transformer_model']['maxlen']
    )

    valid_generator = torch.utils.data.DataLoader(
        valid_set,
        batch_size=config['transformer_model']['batch_size'],
        shuffle=config['transformer_model']['shuffle']
    )

    return training_generator, valid_generator


def train_evaluate(
        model: BertCLF,
        config: Dict[str, Dict[str, Any]],
        training_generator: torch.utils.data.DataLoader,
        valid_generator: torch.utils.data.DataLoader,
        criterion: torch.optim,
        optimizer: torch.nn
):
    """
    Training and evaluation process
    :param model: architecture you want to fine-tune
    :param config: config file with all the necessary information for training
    :param training_generator: training data
    :param valid_generator: evaluation data
    :param criterion: loss from torch losses
    :param optimizer: optimizer from torch optimizers
    :return: fine-tuned model
    """
    for i in range(config['training']['num_epocs']):

        print(f'==== Epoch {i+1} ====')
        tr = train(
            model=model,
            iterator=training_generator,
            optimizer=optimizer,
            criterion=criterion,
            average=config['training']['average_f1']
        )

        evl = evaluate(
            model=model,
            iterator=valid_generator,
            criterion=criterion,
            average=config['training']['average_f1']
        )

        print(f'Train F1: {tr}\nEval F1: {evl}')

    print()
    print('Final metrics')
    predict_metrics(model, valid_generator)
    return model


def main():
    config = load_config("config.yaml")

    os.makedirs(config['training']['output_dir'], exist_ok=True)

    device = torch.device(config['transformer_model']['device'])
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config['transformer_model']["model"]
    )
    model_bert = AutoModel.from_pretrained(
        pretrained_model_name_or_path=config['transformer_model']["model"]
    ).to(device)

    id2label, train_texts, valid_texts, train_targets, valid_targets = prepare_data(config=config)

    model = BertCLF(
        pretrained_model=model_bert,
        tokenizer=tokenizer,
        id2label=id2label,
        dropout=config['transformer_model']['dropout'],
        tiny=config['transformer_model']['tiny_bert']
    )

    if not config['transformer_model']["path_to_state_dict"]:
        model.load_state_dict(
            torch.load(config['transformer_model']["path_to_state_dict"], map_location=device),
            strict=False
        )

    optimizer = optim.Adam(model.parameters(), lr=float(config['transformer_model']['learning_rate']))
    criterion = nn.NLLLoss()

    training_generator, valid_generator = prepare_dataset(
        tokenizer=tokenizer,
        train_texts=train_texts,
        train_targets=train_targets,
        valid_texts=valid_texts,
        valid_targets=valid_targets,
        config=config
    )

    model = train_evaluate(
        model=model,
        config=config,
        training_generator=training_generator,
        valid_generator=valid_generator,
        criterion=criterion,
        optimizer=optimizer
    )
    torch.save(model.state_dict(), os.path.join(config['training']['output_dir'], "model"))
    with open(os.path.join(config['training']['output_dir'], 'label_mapper.json'), mode='w', encoding='utf-8') as f:
        json.dump(model.mapper, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()



