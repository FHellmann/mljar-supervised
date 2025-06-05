from typing import Union, Optional
import numpy as np
import pandas as pd

import torch


class DataPreparer:
    """
    A utility class for preparing input and output data for PyTorch models.
    Supports classification (binary & multiclass) and regression.
    """

    @staticmethod
    def prepare_input(x: Union[np.ndarray, pd.DataFrame, pd.Series]) -> torch.Tensor:
        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x.values
        return torch.tensor(x, dtype=torch.float32)

    @staticmethod
    def prepare_output(
        y: Union[np.ndarray, pd.Series, list],
        task_type: str = "binary",  # binary | multiclass | regression
        target_specification: Optional = None,
    ) -> torch.Tensor:
        """
        Prepare target variables for training with PyTorch.

        :param y: Target variable.
        :param task_type: Task type – binary, multiclass, or regression.
        :param target_specification: For binary classification only: target specification of the target variable.
        :return: Torch tensor of the target variable in the appropriate format.
        """

        if isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, list):
            y = np.array(y)

        if task_type == "binary":
            if target_specification is None:
                raise ValueError(
                    "For binary classification, 'target_specification' must be provided."
                )
            y = np.array(
                [1 if label == target_specification else 0 for label in y],
                dtype=np.float32,
            )
            return torch.tensor(y, dtype=torch.float32)

        elif task_type == "multiclass":
            y = np.asarray(y, dtype=np.int64)
            return torch.tensor(y, dtype=torch.long)

        elif task_type == "regression":
            y = np.asarray(y, dtype=np.float32)
            return torch.tensor(y, dtype=torch.float32)

        else:
            raise ValueError(
                f"Unknown task_type '{task_type}'. Expected: 'binary', 'multiclass', 'regression'."
            )
