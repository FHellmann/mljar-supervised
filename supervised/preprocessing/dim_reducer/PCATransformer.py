from typing import Optional
import pandas as pd

from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from supervised.preprocessing.base_transformer import BaseTransformer
from supervised.utils.attribute_storage import AttributeStorage


class PCATransformer(BaseTransformer, AttributeStorage):
    def __init__(self, results_path: Optional[str] = None, variance_threshold=0.95):
        super().__init__("pca", results_path)
        self.results_path = results_path
        self._input_columns = None
        self._new_features = None
        self._error = None
        self.pca = None
        self._scale = None
        self.variance_threshold = variance_threshold  # Variance to be maintained

    def fit(self, X: DataFrame, y: DataFrame = None, **kwargs) -> None:

        self._input_columns = X.columns.tolist()
        self._scale = StandardScaler()
        X_scaled = self._scale.fit_transform(X)

        self.pca = PCA(n_components=self.variance_threshold)
        self.pca.fit(X_scaled)

        n_components = self.pca.components_.shape[0]
        self._new_features = [f"PC_{i+1}" for i in range(n_components)]

    def transform(self, X: DataFrame, **kwargs) -> DataFrame:

        if self.pca is None:
            raise AttributeError("PCA is not fitted yet.")

        # only use the columns that were used in training
        X_scaled = self._scale.transform(X[self._input_columns])
        X_pca = self.pca.transform(X_scaled)

        if X_pca.ndim == 1:
            X_pca = X_pca.reshape(-1, 1)

        # X_transformed = X.copy()
        # for i, col in enumerate(self._new_features):
        #     X_transformed[col] = X_pca[:, i]
        #
        # print("transform method:", X_transformed.columns)
        #
        # return X_transformed

        X_transformed = pd.DataFrame(X_pca, columns=self._new_features, index=X.index)
        # *** ENDE DER ANPASSUNG ***

        print("DEBUG (PCATransformer.py; transform): X data after PCA: ", X_transformed.columns)

        return X_transformed

        # return pd.DataFrame(X_pca, columns=self._new_features, index=X.index)
