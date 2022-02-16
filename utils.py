import yaml
import pandas as pd


def load_config(path: str):
    with open(path, "r") as yamlfile:
        data = yaml.safe_load(yamlfile)
        return data
