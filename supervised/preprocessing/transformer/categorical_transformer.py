import numpy as np
import pandas as pd
from pandas import DataFrame

from supervised.preprocessing.base_transformer import BaseTransformer
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.preprocessing.transformer.label_binarizer import LabelBinarizer
from supervised.preprocessing.transformer.label_encoder import LabelEncoder
from supervised.preprocessing.transformer.loo_encoder import LooEncoder


class CategoricalTransformer(BaseTransformer):
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
                self._convert_method == CategoricalTransformer.CONVERT_LOO
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
                    self._convert_method == CategoricalTransformer.CONVERT_ONE_HOT
                    and not too_many_categories
            ):
                lbl = LabelBinarizer()
            else:
                lbl = LabelEncoder()

            lbl.fit(X=X[column])
            self._convert_params[column] = lbl.to_dict()

    def transform(self, X: DataFrame, **kwargs):
        if (
                self._convert_method == CategoricalTransformer.CONVERT_LOO
                and self._columns
        ):
            return self._enc.transform(X)
        else:
            for column, lbl_params in self._convert_params.items():
                if "unique_values" in lbl_params and "new_columns" in lbl_params:
                    # convert to one hot
                    lbl = LabelBinarizer()
                    lbl.from_dict(lbl_params)
                    X = lbl.transform(X, column=column)
                else:
                    # convert to integer
                    lbl = LabelEncoder()
                    lbl.from_dict(lbl_params)
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

    @staticmethod
    def get_categorical_encoding(X, y, column):
        # return PreprocessingCategorical.CONVERT_LOO
        try:
            unique_cnt = len(np.unique(X.loc[~pd.isnull(X[column]), column]))
            if unique_cnt <= 20:
                return CategoricalTransformer.FEW_CATEGORIES
        except Exception as e:
            pass

        return CategoricalTransformer.MANY_CATEGORIES
        """
        if unique_cnt <= 2 or unique_cnt > 25:
            return PreprocessingCategorical.CONVERT_INTEGER

        return PreprocessingCategorical.CONVERT_ONE_HOT
        """
