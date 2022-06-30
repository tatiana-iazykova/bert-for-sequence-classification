import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union


class SimpleDataFrame:
    """
    wrapper to have one class for all dataframes
    """
    def __init__(
            self,
            train_data: pd.DataFrame,
            test_data: pd.DataFrame = None,
            random_state: int = 42,
            text_label: str = '',
            target_label: str = '',
            test_size: float = 0.3,
            stratify: Union[bool, str, pd.Series] = True
    ):
        self.train = train_data[[text_label, target_label]]
        self.train.dropna(inplace=True)
        self.test = test_data[[text_label, target_label]].dropna() if test_data is not None else None

        if self.test is None:
            self.__train_test_split(
                random_state=random_state,
                target_label=target_label,
                test_size=test_size,
                stratify=stratify
            )

    def __train_test_split(
            self,
            random_state: int,
            target_label: str,
            test_size: float,
            stratify: Union[bool, str, pd.Series] = True
    ):
        if stratify is True:
            stratify = self.train[target_label]
        elif type(stratify) == str:
            if stratify in self.train.columns:
                stratify = self.train[stratify]
            else:
                raise ValueError('Column not found in data')
        elif type(stratify) == pd.Series:
            stratify = stratify
        else:
            stratify = None
        self.train, self.test = train_test_split(self.train,
                                                 test_size=test_size,
                                                 random_state=random_state,
                                                 stratify=stratify
                                                 )
