import os
from src.training_utils import train_evaluate
from src.BertCLF import BertCLF
from transformers import AutoModel
from transformers import AutoTokenizer
import torch
import torch.optim as optim
import torch.nn as nn
from utils import load_config
import json
from src.preparing_data_utils import prepare_data, prepare_dataset


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
