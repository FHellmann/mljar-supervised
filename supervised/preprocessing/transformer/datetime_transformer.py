from typing import Dict, Any, Callable

import numpy as np
import pandas as pd
from pandas import DataFrame

from supervised.utils.attribute_serializer import AttributeSerializer
from supervised.preprocessing.base_transformer import BaseTransformer


class DateTimeTransformer(BaseTransformer, AttributeSerializer):
    def __init__(self):
        super(DateTimeTransformer, self).__init__("datetime")
        self._new_columns = []
        self._old_column = None
        self._min_datetime = None
        self._transforms = []

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs):
        column = kwargs["column"]
        self._old_column = column
        self._min_datetime = np.min(X[column])

        values = X[column].dt.year
        if len(np.unique(values)) > 1:
            self._transforms += ["year"]
            new_column = column + "_Year"
            self._new_columns += [new_column]

        values = X[column].dt.month
        if len(np.unique(values)) > 1:
            self._transforms += ["month"]
            new_column = column + "_Month"
            self._new_columns += [new_column]

        values = X[column].dt.day
        if len(np.unique(values)) > 1:
            self._transforms += ["day"]
            new_column = column + "_Day"
            self._new_columns += [new_column]

        values = X[column].dt.weekday
        if len(np.unique(values)) > 1:
            self._transforms += ["weekday"]
            new_column = column + "_WeekDay"
            self._new_columns += [new_column]

        values = X[column].dt.dayofyear
        if len(np.unique(values)) > 1:
            self._transforms += ["dayofyear"]
            new_column = column + "_DayOfYear"
            self._new_columns += [new_column]

        values = X[column].dt.hour
        if len(np.unique(values)) > 1:
            self._transforms += ["hour"]
            new_column = column + "_Hour"
            self._new_columns += [new_column]

        values = (X[column] - self._min_datetime).dt.days
        if len(np.unique(values)) > 1:
            self._transforms += ["days_diff"]
            new_column = column + "_Days_Diff_To_Min"
            self._new_columns += [new_column]

    def transform(self, X: DataFrame, **kwargs):
        column = self._old_column

        if "year" in self._transforms:
            new_column = column + "_Year"
            X[new_column] = X[column].dt.year

        if "month" in self._transforms:
            new_column = column + "_Month"
            X[new_column] = X[column].dt.month

        if "day" in self._transforms:
            new_column = column + "_Day"
            X[new_column] = X[column].dt.day

        if "weekday" in self._transforms:
            new_column = column + "_WeekDay"
            X[new_column] = X[column].dt.weekday

        if "dayofyear" in self._transforms:
            new_column = column + "_DayOfYear"
            X[new_column] = X[column].dt.dayofyear

        if "hour" in self._transforms:
            new_column = column + "_Hour"
            X[new_column] = X[column].dt.hour

        if "days_diff" in self._transforms:
            new_column = column + "_Days_Diff_To_Min"
            X[new_column] = (X[column] - self._min_datetime).dt.days

        X.drop(column, axis=1, inplace=True)
        return X

    def from_dict(self, params: Dict[str, Any], **attribute_decoders: Callable[[Any], Any]) -> None:
        super().from_dict(params, _min_datetime=lambda x: None if x is None else pd.to_datetime(x),
                          **attribute_decoders)
