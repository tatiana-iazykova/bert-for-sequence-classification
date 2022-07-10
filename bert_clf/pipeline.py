import json
import os

import numpy as np
import torch
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel, AutoTokenizer

from bert_clf import BertCLF, train_evaluate, prepare_data, prepare_dataset
from bert_clf.utils import load_config, get_argparse, set_global_seed, str_to_class


def train(path_to_config: str):
    """
    path_to_config: path to yaml config file with all the information concerning the training
    """
    config = load_config(path_to_config)

    try:
        loss_func = str_to_class(module_name='torch.nn', class_name=config["training"]["loss"])
    except AttributeError:
        raise ImportError("Couldn't find your loss function in torch.nn module, please check that you spelled your loss"
                          "function correctly and that it exists in this version of PyTorch")

    set_global_seed(seed=config['data']['random_state'])

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
        device=device
    )

    model = model.to(device)

    if config['transformer_model']["path_to_state_dict"]:
        model.load_state_dict(
            torch.load(config['transformer_model']["path_to_state_dict"], map_location=device),
            strict=False
        )

    optimizer = optim.Adam(model.parameters(), lr=float(config['transformer_model']['learning_rate']))

    if config['training']['class_weight']:
        class_weight = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_targets),
            y=train_targets,
        )

        class_weight = torch.Tensor(class_weight).to(device)
        criterion = loss_func(weight=class_weight)

    else:
        criterion = loss_func()

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
        training_generator=training_generator,
        valid_generator=valid_generator,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['training']['num_epochs'],
        average=config['training']['average_f1'],
        config=config
    )

    if config['training']['save_state_dict']:
        torch.save(
            model.state_dict(),
            os.path.join(config["training"]["output_dir"], "model.pth"),
        )

        with open(
                os.path.join(config["training"]["output_dir"], 'label_mapper.json'),
                mode='w',
                encoding='utf-8'
        ) as f:
            json.dump(model.mapper, f, indent=4, ensure_ascii=False)
    else:
        torch.save(
            model,
            os.path.join(config["training"]["output_dir"], "model.pth"),
        )


def main():
    parser = get_argparse()
    args = parser.parse_args()

    train(path_to_config=args.path_to_config)


if __name__ == "__main__":
    main()
