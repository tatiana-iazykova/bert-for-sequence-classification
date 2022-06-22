import pandas as pd
from bert_clf.src.pandas_dataset.PandasDataset import PandasDataset
from bert_clf.src.pandas_dataset.SimpleDataFrame import SimpleDataFrame
from typing import Dict, Any, List, Tuple, Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from bert_clf.src.dataset import Dataset
import torch
import warnings


def prepare_data(
        config: Dict[str, Dict[str, Any]],
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

    return get_mapper_and_separated_data(config=config, df=df)


def prepare_data_notebook(
        config: Dict[str, Dict[str, Any]],
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None
) -> Tuple[Dict[int, str], List[str], List[str], List[int], List[int]]:
    """
    loads data and encodes targets

    :param config: config with  all the necessary information for loading data
    :param train_df: bert_clf dataframe
    :param test_df: testing dataframe
    :return: mapper and separated texts and labels
    """

    df = SimpleDataFrame(
        train_data=train_df,
        test_data=test_df,
        random_state=config['data']['random_state'],
        text_label=config['data']['text_column'],
        target_label=config['data']['target_column'],
        test_size=config['data']['test_size']
    )

    return get_mapper_and_separated_data(config=config, df=df)


def get_mapper_and_separated_data(
        config: Dict[str, Dict[str, Any]],
        df: Union[SimpleDataFrame, PandasDataset]
) -> Tuple[Dict[int, str], List[str], List[str], List[int], List[int]]:
    """
    separates texts and targets and encodes the latter

    :param config: config with  all the necessary information for handling the data
    :param df: class object that has train and valid data

    :return: mapper and separated texts and labels
    """

    id2label = {i: l for i, l in enumerate(df.train[config['data']['target_column']].unique())}
    label2id = {l: i for i, l in id2label.items()}
    train_texts = df.train[config['data']['text_column']].to_list()
    valid_texts = df.test[config['data']['text_column']].to_list()
    train_targets = df.train[config['data']['target_column']].map(label2id).to_list()

    if len(set(df.test[config['data']['target_column']].to_list()).intersection(label2id)) != len(label2id):
        raise ValueError('Some labels in test are not present in train')

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
    :param train_texts: list of texts selected for bert_clf
    :param train_targets: list of targets for texts selected for bert_clf
    :param valid_texts: list of texts selected for evaluation
    :param valid_targets: list of targets for texts selected for evaluation
    :param config:  config with all the necessary information to set up a model

    :return: Dataloader  objects for bert_clf and evaluatuon
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
