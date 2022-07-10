[![pypi version](https://img.shields.io/pypi/v/bert-for-sequence-classification)](https://pypi.org/project/bert-for-sequence-classification)
[![pypi downloads](https://img.shields.io/pypi/dm/bert-for-sequence-classification)](https://pypi.org/project/bert-for-sequence-classification)

# bert-for-sequence-classification
Pipeline for easy fine-tuning of BERT architecture for sequence classification

## Quick Start

### Installation

1. Install the library
```
pip install bert-for-sequence-classification
```
   
2. If you want to train you model on GPU, please install pytorch version compatible with your device.

To find the version compatible with the cuda installed on your GPU, check 
[Pytorch website](https://pytorch.org/get-started/previous-versions/).
You can learn CUDA version installed on your device by typing `nvidia-smi` in console or
`!nvidia-smi` in a notebook cell.

### CLI Use

```
bert-clf-train --path_to_config <path to yaml file>
```

Example config file can be found [here](config.yaml)

### Jupyter notebook

Example notebook can be found [here](example/pipeline_example.ipynb)

### Inference mode

When using your trained model for inference it depends on how you saved your model

if path_to_state_dict in [config](config.yaml) is equal to false, 
then if you have the library installed:

```python

import torch
import pandas as pd

device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")

model = torch.load(
    "path_to_saved_model", map_location=device
)
    
model.eval()

df = pd.read_csv("path_to_some_df")

df["target_column"] = df["text_column"].apply(model.predict)
```

Otherwise:

```python

import torch
import json
import pandas as pd
from bert_clf.src.BertCLF import BertCLF
from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="pretrained_model_name_or_path"
    )
model_bert = AutoModel.from_pretrained(
    pretrained_model_name_or_path="pretrained_model_name_or_path"
).to(device)

id2label = json.load(open("path/to/saved/mapper")) # mapper is saved with the state dict

model = BertCLF(
    pretrained_model=model_bert,
    tokenizer=tokenizer,
    id2label=id2label,
    dropout="some number",
    device=device
)

model.load_state_dict(
    torch.load(
    "path_to_state_dict", map_location=device
    ),
    strict=False
)

model.eval()
    
df = pd.read_csv("path_to_some_df")

df["target_column"] = df["text_column"].apply(model.predict)
```
