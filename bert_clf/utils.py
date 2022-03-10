import yaml
from argparse import ArgumentParser


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
