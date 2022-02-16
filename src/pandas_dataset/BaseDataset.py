import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


class BaseDataset:

    def __init__(
            self,
            train_data_path: str,
            test_data_path: str = None,
            random_state: int = 42,
            text_label: str = '',
            target_label: str = '',
            test_size: float = 0.3,
    ):
        self.train = self.read_data(Path(repr(train_data_path)[1:-1]).as_posix())[[text_label, target_label]]
        self.train.dropna(inplace=True)
        self.test = self.read_data(
            Path(repr(test_data_path)[1:-1]).as_posix()
        )[[text_label, target_label]].dropna() if test_data_path is not None else None

        if not self.test:
            self.__train_test_split(
                random_state=random_state,
                target_label=target_label,
                test_size=test_size
            )

    def read_data(self, path: str) -> pd.DataFrame:
        raise NotImplementedError

    def __train_test_split(self, random_state: int, target_label: str, test_size: float):
        self.train, self.test = train_test_split(self.train,
                                                 test_size=test_size,
                                                 random_state=random_state,
                                                 stratify=self.train[target_label]
                                                 )
