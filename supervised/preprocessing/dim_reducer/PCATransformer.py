from typing import Optional

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from supervised.preprocessing.base_transformer import BaseTransformer
from supervised.utils.attribute_storage import AttributeStorage


class PCATransformer(BaseTransformer, AttributeStorage):
    def __init__(self, results_path: Optional[str] = None, variance_threshold=0.95):
        super().__init__("pca", results_path)
        self._input_columns = None
        self._new_features = None
        self._error = None
        self._pca = None
        self._scale = None
        self._variance_threshold = variance_threshold  # Variance to be maintained

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs) -> None:

        self._input_columns = X.columns.tolist()
        self._scale = StandardScaler()
        X_scaled = self._scale.fit_transform(X)

        self._pca = PCA(n_components=self._variance_threshold)
        self._pca.fit(X_scaled)

        self._new_features = [f"PC_{i+1}" for i in range(len(self._pca.n_components_))]

    def transform(self, X: DataFrame, **kwargs) -> DataFrame:

        if self._pca is None:
            raise AttributeError("PCA is not fitted yet.")

        # only use the columns that were used in training
        X_scaled = self._scale.transform(X[self._input_columns])
        X_pca = self._pca.transform(X_scaled)

        X = X.copy()
        X[self._new_features] = X_pca

        return X
