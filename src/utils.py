import torch
from sklearn.metrics import classification_report
from src.BertCLF import BertCLF
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm


def predict_metrics(
        model: BertCLF,
        iterator: torch.utils.data.DataLoader,
        device: str = 'cpu'
):

    """
    :param model: model architecture that you want to fine-tune
    :param iterator: iterator with data reserved for evaluating
    :param device: cpu or cuda
    """

    device = torch.device(device)
    true = []
    pred = []

    model.eval()
    with torch.no_grad():
        for texts, ys in iterator:
            predictions = model(texts.to(device)).squeeze()
            preds = predictions.detach().to('cpu').numpy().argmax(1).tolist()
            y_true = ys.tolist()
            true.extend(y_true)
            pred.extend(preds)

    true = [model.mapper[x] for x in true]
    pred = [model.mapper[x] for x in pred]

    print(classification_report(true, pred))


def train(
        model: BertCLF,
        iterator: torch.utils.data.DataLoader,
        optimizer: torch.optim,
        criterion: torch.nn,
        device: str = 'cpu',
        average: str = 'macro'
) -> float:
    """
    :param model: model architecture that you want to fine-tune
    :param iterator: iterator with data reserved for training
    :param optimizer: torch optimizer
    :param criterion: instance of torch-like loses
    :param device: cpu or cuda
    :param average: type of averaging for f1 sklearn metric. Possible types are: 'micro', 'macro', 'weighted'
    :return: mean metric for the training loop
    """

    device = torch.device(device)
    epoch_loss = []
    epoch_f1 = []

    model.train()

    for texts, ys in tqdm(iterator, total=len(iterator), desc='Training loop'):

        optimizer.zero_grad()
        predictions = model(texts.to(device))
        loss = criterion(predictions, ys.to(device))

        loss.backward()
        optimizer.step()
        preds = predictions.detach().to('cpu').numpy().argmax(1).tolist()
        y_true = ys.tolist()

        epoch_loss.append(loss.item())
        epoch_f1.append(f1_score(y_true, preds, average=average))

    return np.mean(epoch_f1)


def evaluate(
        model: BertCLF,
        iterator: torch.utils.data.DataLoader,
        criterion: torch.nn,
        device: str = 'cpu',
        average: str = 'macro'
) -> float:
    """
    :param model: trained model (instance of class model.CLF)
    :param iterator: instance of torch.utils.data.DataLoader
    :param criterion: instance of torch-like loses
    :param device: cpu or cuda
    :param average: type of averaging for f1 sklearn metric. Possible types are: 'micro', 'macro', 'weighted'
    :return: mean metric for the evaluating loop
    """
    device = torch.device(device)
    epoch_loss = []
    epoch_f1 = []

    model.eval()
    with torch.no_grad():
        for texts, ys in tqdm(iterator, total=len(iterator), desc='Evaluating loop'):
            predictions = model(texts.to(device))
            loss = criterion(predictions, ys.to(device))
            preds = predictions.detach().to('cpu').numpy().argmax(1).tolist()
            y_true = ys.tolist()

            epoch_loss.append(loss.item())
            epoch_f1.append(f1_score(y_true, preds, average=average))

    return np.mean(epoch_f1)
