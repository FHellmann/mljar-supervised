from typing import List, Callable, Any, Dict

import numpy as np
import pandas as pd

from supervised.preprocessing.base_transformer import BaseTransformer
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.utils.attribute_serializer import AttributeSerializer


class MissingValuesTransformer(BaseTransformer, AttributeSerializer):
    FILL_NA_MIN = "na_fill_min_1"
    FILL_NA_MEAN = "na_fill_mean"
    FILL_NA_MEDIAN = "na_fill_median"
    FILL_DATETIME = "na_fill_datetime"

    NA_EXCLUDE = "na_exclude"
    MISSING_VALUE = "_missing_value_"
    REMOVE_COLUMN = "remove_column"

    def __init__(self, columns=[], na_fill_method=FILL_NA_MEDIAN):
        self._columns = columns
        # fill method
        self._na_fill_method = na_fill_method
        # fill parameters stored as a dict, feature -> fill value
        self._na_fill_params = {}
        self._datetime_columns = []

    def fit(self, X):
        self._fit_na_fill(X)

    def _fit_na_fill(self, X):
        for column in self._columns:
            if np.sum(pd.isnull(X[column]) == True) == 0:
                continue
            self._na_fill_params[column] = self._get_fill_value(X[column])
            if PreprocessingUtils.get_type(X[column]) == PreprocessingUtils.DATETIME:
                self._datetime_columns += [column]

    def _get_fill_value(self, x):
        # categorical type
        if PreprocessingUtils.get_type(x) == PreprocessingUtils.CATEGORICAL:
            if self._na_fill_method == MissingValuesTransformer.FILL_NA_MIN:
                return (
                    MissingValuesTransformer.MISSING_VALUE
                )  # add new categorical value
            return PreprocessingUtils.get_most_frequent(x)
        # datetime
        if PreprocessingUtils.get_type(x) == PreprocessingUtils.DATETIME:
            return PreprocessingUtils.get_most_frequent(x)
        # text
        if PreprocessingUtils.get_type(x) == PreprocessingUtils.TEXT:
            return MissingValuesTransformer.MISSING_VALUE

        # numerical type
        if self._na_fill_method == MissingValuesTransformer.FILL_NA_MIN:
            return PreprocessingUtils.get_min(x) - 1.0
        if self._na_fill_method == MissingValuesTransformer.FILL_NA_MEAN:
            return PreprocessingUtils.get_mean(x)
        return PreprocessingUtils.get_median(x)

    def transform(self, X):
        X = self._transform_na_fill(X)
        # this is additional run through columns,
        # in case of transforming data with new columns with missing values
        # X = self._make_sure_na_filled(X) # disbaled for now
        return X

    def _transform_na_fill(self, X):
        for column, value in self._na_fill_params.items():
            ind = pd.isnull(X.loc[:, column])
            X.loc[ind, column] = value
        return X

    def _make_sure_na_filled(self, X):
        self._fit_na_fill(X)
        return self._transform_na_fill(X)

    def to_dict(self, exclude_callables_nones: bool = True, exclude_attributes: List[str] = None,
                **attribute_encoders: Callable[[Any], Any]) -> Dict[str, Any] | None:
        if len(self._na_fill_params) == 0:
            return {}
        return super().to_dict(exclude_callables_nones, exclude_attributes,
                               _datetime_columns=lambda x: list(x),
                               _na_fill_params=lambda x: {key: str(value) for key, value in x.items()},
                               **attribute_encoders)

    def from_dict(self, params: Dict[str, Any], **attribute_decoders: Callable[[Any], Any]) -> None:
        if params is not None:
            super().from_dict(params,
                              _na_fill_params=lambda x: {key: pd.to_datetime(value) for key, value in x.items()},
                              **attribute_decoders)
        else:
            self._na_fill_method, self._na_fill_params = None, None
            self._datetime_columns = []
