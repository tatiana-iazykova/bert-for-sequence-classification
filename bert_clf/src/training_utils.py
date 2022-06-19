import torch
from sklearn.metrics import classification_report
from bert_clf.src.BertCLF import BertCLF
from bert_clf.src.early_stopping import EarlyStopping
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from typing import Dict, Any, Optional, Union, List, Tuple


def predict_metrics(
        model: BertCLF,
        iterator: torch.utils.data.DataLoader,
):

    """
    :param model: model architecture that you want to fine-tune
    :param iterator: iterator with data reserved for evaluating
    """

    true = []
    pred = []

    model.eval()
    with torch.no_grad():
        for texts, ys in tqdm(iterator, total=len(iterator), desc="Computing final metrics..."):
            predictions = model(texts.to(model.device)).squeeze()
            preds = predictions.detach().to('cpu').numpy().argmax(1).tolist()
            y_true = ys.tolist()
            true.extend(y_true)
            pred.extend(preds)

    true = [model.mapper[x] for x in true]
    pred = [model.mapper[x] for x in pred]

    print(classification_report(true, pred, zero_division=0))


def train(
        model: BertCLF,
        iterator: torch.utils.data.DataLoader,
        optimizer: torch.optim,
        criterion: torch.nn,
        average: str = 'macro',
        other_metrics: Optional[Union[str, List[str]]] = None
) -> Tuple[float, Optional[Dict[str, float]]]:
    """
    :param model: model architecture that you want to fine-tune
    :param iterator: iterator with data reserved for bert_clf
    :param optimizer: torch optimizer
    :param criterion: instance of torch-like loses
    :param average: type of averaging for f1 sklearn metric. Possible types are: 'micro', 'macro', 'weighted'
    :return: mean metric for the bert_clf loop
    """

    epoch_loss = []
    epoch_f1 = []
    metrics = None
    if other_metrics is not None:
        metrics = {}
        if type(other_metrics) == str:
            metrics[other_metrics] = []
        elif type(other_metrics) == list:
            for m in other_metrics:
                metrics[m] = []
        else:
            raise ValueError("Other metrics can be only in list or str format")

    model.train()

    for texts, ys in tqdm(iterator, total=len(iterator), desc='Training loop'):

        optimizer.zero_grad()
        predictions = model(texts.to(model.device))
        loss = criterion(predictions, ys.to(model.device))

        loss.backward()
        optimizer.step()
        preds = predictions.detach().to('cpu').numpy().argmax(1).tolist()
        y_true = ys.tolist()

        epoch_loss.append(loss.item())
        epoch_f1.append(f1_score(y_true, preds, average=average))
        if metrics is not None:
            for k in metrics:
                metrics[k].append(f1_score(y_true, preds, average=k))

    if metrics is not None:
        for k, v in metrics.items():
            metrics[k] = np.mean(v)
    return np.mean(epoch_f1), metrics


def evaluate(
        model: BertCLF,
        iterator: torch.utils.data.DataLoader,
        criterion: torch.nn,
        average: str = 'macro',
        other_metrics: Optional[Union[str, List[str]]] = None
) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
    """
    :param model: trained model (instance of class model.CLF)
    :param iterator: instance of torch.utils.data.DataLoader
    :param criterion: instance of torch-like loses
    :param average: type of averaging for f1 sklearn metric. Possible types are: 'micro', 'macro', 'weighted'
    :param other_metrics: other metrics you would like to track. NOTE: they wouldn't affect the training provess
    :return: mean metric for the evaluating loop
    """

    epoch_loss = []
    epoch_f1 = []
    metrics = None

    if average not in ['micro', 'macro', 'weighted']:
        raise ValueError(f"average parameter can only be 'micro', 'macro', 'weighted', got '{average}'")

    if other_metrics is not None:
        metrics = {}
        if type(other_metrics) == str:
            if other_metrics not in ['micro', 'macro', 'weighted']:
                raise ValueError(f"other_metrics parameter can only be 'micro',"
                                 f" 'macro', 'weighted', got '{other_metrics}'")
            metrics[other_metrics] = []
        elif type(other_metrics) == list:
            for m in other_metrics:
                if m not in ['micro', 'macro', 'weighted']:
                    raise ValueError(f"other_metrics parameter can only be 'micro',"
                                     f" 'macro', 'weighted', got '{m}'")
                metrics[m] = []
        else:
            raise ValueError("Other metrics can be only in list or str format")

    model.eval()
    with torch.no_grad():
        for texts, ys in tqdm(iterator, total=len(iterator), desc='Evaluating loop'):
            predictions = model(texts.to(model.device))
            loss = criterion(predictions, ys.to(model.device))
            preds = predictions.detach().to('cpu').numpy().argmax(1).tolist()
            y_true = ys.tolist()

            epoch_loss.append(loss.item())
            epoch_f1.append(f1_score(y_true, preds, average=average))
            if metrics is not None:
                for k in metrics:
                    metrics[k].append(f1_score(y_true, preds, average=k))

    if metrics is not None:
        for k, v in metrics.items():
            metrics[k] = np.mean(v)

    return np.mean(epoch_f1), metrics


def train_evaluate(
        model: BertCLF,
        training_generator: torch.utils.data.DataLoader,
        valid_generator: torch.utils.data.DataLoader,
        criterion: torch.optim,
        optimizer: torch.nn,
        num_epochs: int,
        average: str,
        config: Dict[str, Union[Dict[str, Any], List[str]]]
):
    """
    Training and evaluation process
    :param model: architecture you want to fine-tune
    :param training_generator: bert_clf data
    :param valid_generator: evaluation data
    :param criterion: loss from torch losses
    :param optimizer: optimizer from torch optimizers
    :param num_epochs: number of epochs,
    :param config: configuration file with all the parameters needed for training the model,
    :param average: f1-averaging

    :return: fine-tuned model
    """

    stopper = EarlyStopping(
        config=config,
    )

    for i in range(num_epochs):

        print(f"==== Epoch {i+1} out of {num_epochs} ====")
        tr, metrics_train = train(
            model=model,
            iterator=training_generator,
            optimizer=optimizer,
            criterion=criterion,
            average=average,
            other_metrics=config['training']['other_metrics']
        )

        evl, metrics_eval = evaluate(
            model=model,
            iterator=valid_generator,
            criterion=criterion,
            average=average,
            other_metrics=config['training']['other_metrics']
        )

        if config['training']['early_stopping'] is True:
            stopper(model=model, val_loss=evl)

            if stopper.early_stop:
                print('Early stopping')
                print(f'Train F1: {tr}\nEval F1: {evl}')
                print("\n\n")
                predict_metrics(
                    model=model,
                    iterator=valid_generator
                )
                return model

        print(f'Train F1: {tr}\nEval F1: {evl}')
        print()
        if metrics_train is not None and metrics_eval is not None:
            for m in metrics_train:
                print(f'Train F1 {m}: {metrics_train[m]}\nEval F1 {m}: {metrics_eval[m]}')
                print()

    print()
    predict_metrics(
        model=model,
        iterator=valid_generator
    )
    return model
