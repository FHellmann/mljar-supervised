from typing import List, Callable, Any, Dict

import pandas as pd
from pandas import DataFrame

from supervised.utils.attribute_serializer import AttributeSerializer
from supervised.preprocessing.base_transformer import BaseTransformer
from supervised.preprocessing.transformer.label_binarizer import LabelBinarizer
from supervised.preprocessing.transformer.label_encoder import LabelEncoder
from supervised.preprocessing.transformer.loo_encoder import LooEncoder
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils


class PreprocessingCategorical(BaseTransformer, AttributeSerializer):
    CONVERT_ONE_HOT = "categorical_to_onehot"
    CONVERT_INTEGER = "categorical_to_int"
    CONVERT_LOO = "categorical_to_loo"

    FEW_CATEGORIES = "few_categories"
    MANY_CATEGORIES = "many_categories"

    def __init__(self, columns=[], method=CONVERT_INTEGER):
        self._convert_method = method
        self._convert_params = {}
        self._columns = columns
        self._enc = None

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs):
        if (
            self._convert_method == PreprocessingCategorical.CONVERT_LOO
            and self._columns
        ):
            self._enc = LooEncoder(cols=self._columns)
            self._enc.fit(X, y)
        else:
            self._fit_categorical_convert(X)

    def _fit_categorical_convert(self, X):
        # Precompute unique counts for all columns to avoid repeated calculations
        unique_counts = {col: X[col].nunique(dropna=False) for col in self._columns}

        for column in self._columns:
            if PreprocessingUtils.get_type(X[column]) != PreprocessingUtils.CATEGORICAL:
                # no need to convert, already a number
                continue
            # limit categories - it is needed when doing one hot encoding
            # this code is also used in predict.py file
            # and transform_utils.py
            # TODO it needs refactoring !!!
            too_many_categories = unique_counts[column] > 200
            if (
                self._convert_method == PreprocessingCategorical.CONVERT_ONE_HOT
                and not too_many_categories
            ):
                lbl = LabelBinarizer()
            else:
                lbl = LabelEncoder()

            lbl.fit(X=X[column])
            self._convert_params[column] = lbl.to_dict()

    def transform(self, X: DataFrame, **kwargs):
        if (
            self._convert_method == PreprocessingCategorical.CONVERT_LOO
            and self._columns
        ):
            return self._enc.transform(X)
        else:
            for column, lbl_params in self._convert_params.items():
                if "unique_values" in lbl_params and "new_columns" in lbl_params:
                    # convert to one hot
                    lbl = LabelBinarizer()
                    lbl.from_json(lbl_params)
                    X = lbl.transform(X, column)
                else:
                    # convert to integer
                    lbl = LabelEncoder()
                    lbl.from_json(lbl_params)
                    transformed_values = lbl.transform(X.loc[:, column])
                    # check for pandas FutureWarning: Setting an item
                    # of incompatible dtype is deprecated and will raise
                    # in a future error of pandas.
                    if transformed_values.dtype != X.loc[:, column].dtype and \
                        (X.loc[:, column].dtype == bool or X.loc[:, column].dtype == int):
                        X = X.astype({column: transformed_values.dtype})
                    if isinstance(X[column].dtype, pd.CategoricalDtype):
                        X[column] = X[column].astype('object')
                    X.loc[:, column] = transformed_values

            return X

    def inverse_transform(self, X: DataFrame, **kwargs):
        for column, lbl_params in self._convert_params.items():
            if "unique_values" in lbl_params and "new_columns" in lbl_params:
                # convert to one hot
                lbl = LabelBinarizer()
                lbl.from_json(lbl_params)
                X = lbl.inverse_transform(X, column)  # should raise exception
            else:
                # convert to integer
                lbl = LabelEncoder()
                lbl.from_json(lbl_params)
                transformed_values = lbl.inverse_transform(X.loc[:, column])
                # check for pandas FutureWarning: Setting an item
                # of incompatible dtype is deprecated and will raise
                # in a future error of pandas.
                if transformed_values.dtype != X.loc[:, column].dtype and \
                        (X.loc[:, column].dtype == bool or X.loc[:, column].dtype == int):
                        X = X.astype({column: transformed_values.dtype})
                X.loc[:, column] = transformed_values

        return X

    def to_dict(self, exclude_callables_nones: bool = True, exclude_attributes: List[str] = None,
                **attribute_encoders: Callable[[Any], Any]) -> Dict[str, Any] | None:
        super().to_dict(exclude_callables_nones=exclude_callables_nones, exclude_attributes=exclude_attributes,
                        _enc=lambda x: x.to_dict()
                        **attribute_encoders)

    def to_json(self):
        params = {}
        if (
            self._convert_method == PreprocessingCategorical.CONVERT_LOO
            and self._columns
        ):
            params = {
                "enc": self._enc.to_json(),
                "convert_method": self._convert_method,
                "columns": self._columns,
            }
        elif len(self._convert_params) > 0:
            params = {
                "convert_method": self._convert_method,
                "convert_params": self._convert_params,
                "columns": self._columns,
            }
        return params

    def from_json(self, params):
        if params is not None:
            self._convert_method = params.get("convert_method", None)
            self._columns = params.get("columns", [])
            if self._convert_method == PreprocessingCategorical.CONVERT_LOO:
                self._enc = LooEncoder()
                self._enc.from_json(params.get("enc", {}))
            else:
                self._convert_params = params.get("convert_params", {})

        else:
            self._convert_method, self._convert_params = None, None
            self._columns = []
