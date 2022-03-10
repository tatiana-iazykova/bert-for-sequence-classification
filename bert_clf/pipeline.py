import os
import json
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from bert_clf.utils import load_config, get_argparse
from bert_clf.src.training_utils import train_evaluate
from bert_clf.src.BertCLF import BertCLF
from bert_clf.src.preparing_data_utils import prepare_data, prepare_dataset


def train(path_to_config: str):
    """
    path_to_config: path to yaml config file with all the information concerning the training
    """
    config = load_config(path_to_config)

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
        tiny=config['transformer_model']['tiny_bert'],
        device=device
    )

    model = model.to(device)

    if config['transformer_model']["path_to_state_dict"]:
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
        training_generator=training_generator,
        valid_generator=valid_generator,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['training']['num_epochs'],
        average=config['training']['average_f1']
    )

    torch.save(model.state_dict(), os.path.join(config['training']['output_dir'], "model"))
    with open(os.path.join(config['training']['output_dir'], 'label_mapper.json'), mode='w', encoding='utf-8') as f:
        json.dump(model.mapper, f, indent=4, ensure_ascii=False)


def main():
    parser = get_argparse()
    args = parser.parse_args()

    train(path_to_config=args.path_to_config)


if __name__ == "__main__":
    main()
