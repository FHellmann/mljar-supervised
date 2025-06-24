import numpy as np
import pandas as pd
import json

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


    def to_json(self) -> str:

        # Using the to_dict method of AttributeSerializer
        data_dict = self.to_dict()

        for key, value in data_dict.items():
            if isinstance(value, (np.integer, np.floating)):
                data_dict[key] = value.item()
            elif isinstance(value, np.ndarray):
                data_dict[key] = value.tolist()
        return json.dumps(data_dict)
    @classmethod
    def from_json(cls, json_string: str) -> "MissingValuesTransformer":
        data_dict = json.loads(json_string)
        return cls.create_from_dict(cls, data_dict)
