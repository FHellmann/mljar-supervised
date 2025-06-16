from typing import Optional

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from supervised.preprocessing.base_transformer import BaseTransformer
from supervised.utils.attribute_storage import AttributeStorage


class SVDTransformer(BaseTransformer, AttributeStorage):
    def __init__(self, results_path: Optional[str] = None, n_components=None):
        super().__init__("svd", results_path)
        self._input_columns = None
        self._new_features = None
        self._error = None
        self._svd = None
        self._scale = None
        self._n_components = n_components

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs) -> None:

        self._input_columns = X.columns.tolist()
        self._scale = StandardScaler()
        X_scaled = self._scale.fit_transform(X)

        self._svd = TruncatedSVD(n_components=self._n_components)
        self._svd.fit(X_scaled)

        n_components_actual = self._svd.components_.shape[0]
        self._new_features = [f"SVD_{i+1}" for i in range(n_components_actual)]

    def transform(self, X: DataFrame, **kwargs) -> DataFrame:

        if self._svd is None:
            raise AttributeError("SVD is not fitted yet.")

        # only use the columns that were used in training
        X_scaled = self._scale.transform(X[self._input_columns])
        X_svd = self._svd.transform(X_scaled)

        if X_svd.ndim == 1:
            X_svd = X_svd.reshape(-1, 1)

        X_transformed = X.copy()
        for i, col in enumerate(self._new_features):
            X_transformed[col] = X_svd[:, i]

        return X_transformed
