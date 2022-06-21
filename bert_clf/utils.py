import random
from argparse import ArgumentParser

import numpy as np
import torch
import yaml
import importlib
from typing import ClassVar


def load_config(path: str):
    with open(path, "r") as yamlfile:
        data = yaml.safe_load(yamlfile)
        return data


def get_argparse() -> ArgumentParser:
    """Get argument parser.
    Returns:
        ArgumentParser: Argument parser.
    """

    parser = ArgumentParser(prog="bert-clf-train")
    parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        help="Path to config",
    )

    return parser


def set_global_seed(seed: int):
    """
    Set global seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str_to_class(module_name: str, class_name: str) -> ClassVar:
    """
    Convert string to Python class object.
    https://stackoverflow.com/questions/1176136/convert-string-to-python-class-object
    """

    # load the module, will raise ImportError if module cannot be loaded
    module = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    cls = getattr(module, class_name)
    return cls
