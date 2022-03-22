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
bert-clf-train --path_to_config <path to yaml file>\
```

Example config file can be found [here](config.yaml)

### Jupyter notebook

Example notebook can be found [here](example/pipeline_example.ipynb)

