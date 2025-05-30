import numpy as np
import pandas as pd

from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    REGRESSION,
)
from supervised.preprocessing.preprocessing_utils import PreprocessingUtils
from supervised.preprocessing.transformer.categorical_transformer import CategoricalTransformer


class DataInfo:
    @staticmethod
    def compute(X, y, machinelearning_task):
        columns_info = {}
        for col in X.columns:
            columns_info[col] = []
            #
            empty_column = np.sum(pd.isnull(X[col]) == True) == X.shape[0]
            if empty_column:
                columns_info[col] += ["empty_column"]
                continue
            #
            constant_column = len(np.unique(X.loc[~pd.isnull(X[col]), col])) == 1
            if constant_column:
                columns_info[col] += ["constant_column"]
                continue
            #
            if PreprocessingUtils.is_na(X[col]):
                columns_info[col] += ["missing_values"]
            #
            if PreprocessingUtils.is_categorical(X[col]):
                columns_info[col] += ["categorical"]
                columns_info[col] += [CategoricalTransformer.get_categorical_encoding(X, y, col)]
            elif PreprocessingUtils.is_datetime(X[col]):
                columns_info[col] += ["datetime_transform"]
            elif PreprocessingUtils.is_text(X[col]):
                columns_info[col] = ["text_transform"]  # override other transforms
            else:
                # numeric type, check if scale needed
                if PreprocessingUtils.is_scale_needed(X[col]):
                    columns_info[col] += ["scale"]

        target_info = []
        if machinelearning_task == BINARY_CLASSIFICATION:
            if not PreprocessingUtils.is_0_1(y):
                target_info += ["convert_0_1"]

        if machinelearning_task == REGRESSION:
            if PreprocessingUtils.is_log_scale_needed(y):
                target_info += ["scale_log"]
            elif PreprocessingUtils.is_scale_needed(y):
                target_info += ["scale"]

        num_class = None
        if machinelearning_task == MULTICLASS_CLASSIFICATION:
            num_class = PreprocessingUtils.num_class(y)

        return {
            "columns_info": columns_info,
            "target_info": target_info,
            "num_class": num_class,
        }
