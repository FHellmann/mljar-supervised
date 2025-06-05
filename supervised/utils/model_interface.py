from typing import Protocol
import numpy as np
import pandas as pd


class AbstractModel(Protocol):
    """
    Inherit this class to enable the use of the model for classification.
    """

    def fit(self, x_train, y_train):
        ...

    def predict(self, x: pd.Series) -> np.ndarray:
        ...

    def predict_proba(self, x: pd.Series) -> np.ndarray:
        ...
