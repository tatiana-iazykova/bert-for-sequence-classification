import random
from argparse import ArgumentParser
from typing import Any, Dict

import numpy as np
import torch
import yaml


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


def import_loss(config: Dict[str, Dict[str, Any]]) -> None:
    """
    import loss specified in config file
    """
    import_line = f'from torch.nn import {config["training"]["loss"]} as loss_func'
    exec(import_line)
