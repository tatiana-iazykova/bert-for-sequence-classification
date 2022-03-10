import torch
from sklearn.metrics import classification_report
from bert_clf.src.BertCLF import BertCLF
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm


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

    print(classification_report(true, pred))


def train(
        model: BertCLF,
        iterator: torch.utils.data.DataLoader,
        optimizer: torch.optim,
        criterion: torch.nn,
        average: str = 'macro'
) -> float:
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

    return np.mean(epoch_f1)


def evaluate(
        model: BertCLF,
        iterator: torch.utils.data.DataLoader,
        criterion: torch.nn,
        average: str = 'macro'
) -> float:
    """
    :param model: trained model (instance of class model.CLF)
    :param iterator: instance of torch.utils.data.DataLoader
    :param criterion: instance of torch-like loses
    :param average: type of averaging for f1 sklearn metric. Possible types are: 'micro', 'macro', 'weighted'
    :return: mean metric for the evaluating loop
    """
    epoch_loss = []
    epoch_f1 = []

    model.eval()
    with torch.no_grad():
        for texts, ys in tqdm(iterator, total=len(iterator), desc='Evaluating loop'):
            predictions = model(texts.to(model.device))
            loss = criterion(predictions, ys.to(model.device))
            preds = predictions.detach().to('cpu').numpy().argmax(1).tolist()
            y_true = ys.tolist()

            epoch_loss.append(loss.item())
            epoch_f1.append(f1_score(y_true, preds, average=average))

    return np.mean(epoch_f1)


def train_evaluate(
        model: BertCLF,
        training_generator: torch.utils.data.DataLoader,
        valid_generator: torch.utils.data.DataLoader,
        criterion: torch.optim,
        optimizer: torch.nn,
        num_epochs: int,
        average: str
):
    """
    Training and evaluation process
    :param model: architecture you want to fine-tune
    :param training_generator: bert_clf data
    :param valid_generator: evaluation data
    :param criterion: loss from torch losses
    :param optimizer: optimizer from torch optimizers
    :param num_epocs: number of epochs,
    :param device: device,
    :param average: f1-averaging

    :return: fine-tuned model
    """
    for i in range(num_epochs):

        print(f"==== Epoch {i+1} out of {num_epochs} ====")
        tr = train(
            model=model,
            iterator=training_generator,
            optimizer=optimizer,
            criterion=criterion,
            average=average
        )

        evl = evaluate(
            model=model,
            iterator=valid_generator,
            criterion=criterion,
            average=average
        )

        print(f'Train F1: {tr}\nEval F1: {evl}')
        print()

    print()
    predict_metrics(
        model=model,
        iterator=valid_generator
    )
    return model
