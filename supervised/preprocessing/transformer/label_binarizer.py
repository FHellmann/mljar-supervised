from typing import List, Dict, Any, Callable

import numpy as np
from pandas import DataFrame

from supervised.utils.attribute_serializer import AttributeSerializer
from supervised.preprocessing.base_transformer import BaseTransformer


class LabelBinarizer(BaseTransformer, AttributeSerializer):
    def __init__(self):
        super(LabelBinarizer, self).__init__("label_binarizer")
        self._new_columns = []
        self._uniq_values = None
        self._old_column = None
        self._old_column_dtype = None

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs):
        column = kwargs['column']
        self._old_column = column
        self._old_column_dtype = str(X[column].dtype)
        self._uniq_values = np.unique(X[column].values)
        # self._uniq_values = [str(u) for u in self._uniq_values]

        if len(self._uniq_values) == 2:
            self._new_columns.append(column + "_" + str(self._uniq_values[1]))
        else:
            for v in self._uniq_values:
                self._new_columns.append(column + "_" + str(v))

    def transform(self, X: DataFrame, **kwargs):
        column = kwargs['column']
        if len(self._uniq_values) == 2:
            X[column + "_" + str(self._uniq_values[1])] = (
                    X[column] == self._uniq_values[1]
            ).astype(int)
        else:
            for v in self._uniq_values:
                X[column + "_" + str(v)] = (X[column] == v).astype(int)

        X.drop(column, axis=1, inplace=True)
        return X

    def inverse_transform(self, X: DataFrame, **kwargs) -> DataFrame:
        if self._old_column is None:
            return X

        old_col = (X[self._new_columns[0]] * 0).astype(self._old_column_dtype)

        for unique_value in self._uniq_values:
            new_col = f"{self._old_column}_{unique_value}"
            if new_col not in self._new_columns:
                old_col[:] = unique_value
            else:
                old_col[X[new_col] == 1] = unique_value

        X[self._old_column] = old_col
        X.drop(self._new_columns, axis=1, inplace=True)
        return X

    def to_dict(self, exclude_callables_nones: bool = True, exclude_attributes: List[str] = None,
                **attribute_encoders: Callable[[Any], Any]) -> Dict[str, Any] | None:
        return super().to_dict(exclude_callables_nones, exclude_attributes,
                               _uniq_values=lambda x: [str(i) for i in list(x)], **attribute_encoders)

    def from_dict(self, data_json: Dict[str, Any], **attribute_decoders: Callable[[Any], Any]) -> None:
        super().from_dict(data_json,
                          _uniq_values=lambda x: [False, True] if "True" in x and "False" in x and len(x) == 2 else x,
                          **attribute_decoders)
