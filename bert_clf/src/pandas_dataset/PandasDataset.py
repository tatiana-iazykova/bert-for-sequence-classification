import os

import pandas as pd

from bert_clf.src.pandas_dataset.BaseDataset import BaseDataset


class PandasDataset(BaseDataset):
    """
    Dataset class for datasets that have simple structure
    """

    def __init__(self,
                 train_data_path: str,
                 test_data_path: str = None,
                 random_state: int = 42,
                 text_label: str = '',
                 target_label: str = '',
                 test_size: float = 0.3
                 ):

        self.valid_data_types = {
            '.csv': self._read_csv,
            '.tsv': self._read_csv,
            '.xls': self._read_excel,
            '.xlsx': self._read_excel,
            '.json': self._read_json,
            '.jsonl': self._read_json
        }

        super(PandasDataset, self).__init__(
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            random_state=random_state,
            text_label=text_label,
            target_label=target_label,
            test_size=test_size
        )

    def read_data(self, path: str) -> pd.DataFrame:
        """
        Given the path to the file returns extension of that file

        Example:
        path: "../input/some_data.csv"
        :return: ".csv"
        """
        _, extension = os.path.splitext(path)
        if extension.lower() in self.valid_data_types:
            return self.valid_data_types[extension](path=path, extension=extension)
        else:
            raise ValueError(f"Your data type ({extension}) is not supported, please convert your dataset "
                             f"to one of the following formats {list(self.valid_data_types.keys())}.")

    @staticmethod
    def _read_csv(path: str, extension: str) -> pd.DataFrame:
        """
        Reads a csv file given its path
        :param path: "../../some_file.csv"
        :return: dataframe
        """
        sep = ','
        if extension == '.tsv':
            sep = '\t'

        return pd.read_csv(filepath_or_buffer=path, sep=sep, encoding="utf-8")

    @staticmethod
    def _read_excel(path: str, extension: str) -> pd.DataFrame:
        """
        Reads a xls or xlsx file given its path
        :param path: "../../some_file.xlsx"
        :return: dataframe
        """
        engine = 'openpyxl'
        if extension == '.xls':
            engine = None

        return pd.read_excel(io=path, engine=engine)

    @staticmethod
    def _read_json(path: str, extension: str) -> pd.DataFrame:
        """
        Reads a json or jsonl file given its path
        :param path: "../../some_file.jsonl"
        :return: dataframe
        """
        lines = False
        if extension == '.jsonl':
            lines = True
        return pd.read_json(path_or_buf=path, lines=lines, encoding="utf-8")
