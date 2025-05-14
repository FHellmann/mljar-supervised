import numpy as np
from pandas import DataFrame
from sklearn import preprocessing

from supervised.preprocessing.base_transformer import BaseTransformer
from supervised.utils.attribute_serializer import AttributeSerializer


class ScaleTransformer(BaseTransformer, AttributeSerializer):
    SCALE_NORMAL = "scale_normal"
    SCALE_LOG_AND_NORMAL = "scale_log_and_normal"

    def __init__(self, columns=[], scale_method=SCALE_NORMAL):
        self.scale_method = scale_method
        self.columns = columns
        self.scale = preprocessing.StandardScaler(
            copy=True, with_mean=True, with_std=True
        )
        self.X_min_values = None  # it is used in SCALE_LOG_AND_NORMAL

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs):
        if len(self.columns):
            for c in self.columns:
                X[c] = X[c].astype(float)

            if self.scale_method == self.SCALE_NORMAL:
                self.scale.fit(X[self.columns])
            elif self.scale_method == self.SCALE_LOG_AND_NORMAL:
                self.X_min_values = np.min(X[self.columns], axis=0)
                self.scale.fit(np.log(X[self.columns] - self.X_min_values + 1))

    def transform(self, X: DataFrame, **kwargs):
        if len(self.columns):
            for c in self.columns:
                X[c] = X[c].astype(float)
            if self.scale_method == self.SCALE_NORMAL:
                X.loc[:, self.columns] = self.scale.transform(X[self.columns])
            elif self.scale_method == self.SCALE_LOG_AND_NORMAL:
                X[self.columns] = np.log(
                    np.clip(
                        X[self.columns] - self.X_min_values + 1, a_min=1, a_max=None
                    )
                )
                X.loc[:, self.columns] = self.scale.transform(X[self.columns])
        return X

    def inverse_transform(self, X: DataFrame, **kwargs):
        if len(self.columns):
            if self.scale_method == self.SCALE_NORMAL:
                X.loc[:, self.columns] = self.scale.inverse_transform(X[self.columns])
            elif self.scale_method == self.SCALE_LOG_AND_NORMAL:
                X[self.columns] = X[self.columns].astype("float64")

                X[self.columns] = self.scale.inverse_transform(X[self.columns])
                X[self.columns] = np.exp(X[self.columns])

                X.loc[:, self.columns] += self.X_min_values - 1
        return X
