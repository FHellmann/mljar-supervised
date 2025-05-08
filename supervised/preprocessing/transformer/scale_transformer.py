from typing import List, Callable, Any, Dict

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

    def to_dict(self, exclude_callables_nones: bool = True, exclude_attributes: List[str] = None,
                **attribute_encoders: Callable[[Any], Any]) -> Dict[str, Any] | None:
        def scale_encoder(scale):
            return {
                "scale": list(scale.scale_),
                "mean": list(scale.mean_),
                "var": list(scale.var_),
                "n_samples_seen": int(scale.n_samples_seen_),
                "n_features_in": int(scale.n_features_in_),
            }

        if len(self.columns) == 0:
            return None
        return super().to_dict(exclude_callables_nones, exclude_attributes,
                        scale=scale_encoder, X_min_values=lambda x: list(x), **attribute_encoders)

    def from_dict(self, params: Dict[str, Any], **attribute_decoders: Callable[[Any], Any]) -> None:
        self.scale = preprocessing.StandardScaler(
            copy=True, with_mean=True, with_std=True
        )

        def scale_decoder(data):
            self.scale.scale_ = data.get("scale")
            if self.scale.scale_ is not None:
                self.scale.scale_ = np.array(self.scale.scale_)
            self.scale.mean_ = data.get("mean")
            if self.scale.mean_ is not None:
                self.scale.mean_ = np.array(self.scale.mean_)
            self.scale.var_ = data.get("var")
            if self.scale.var_ is not None:
                self.scale.var_ = np.array(self.scale.var_)
            self.scale.n_samples_seen_ = int(data.get("n_samples_seen"))
            self.scale.n_features_in_ = int(data.get("n_features_in"))
            self.scale.feature_names_in_ = data.get("columns", [])

        super().from_dict(params, scale=scale_decoder, **attribute_decoders)
