import time

import numpy as np
from pandas import DataFrame
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from supervised.exceptions import AutoMLException
from supervised.preprocessing.base_transformer import BaseTransformer


class KMeansTransformer(BaseTransformer):
    def __init__(self, results_path=None, model_name=None, k_fold=None):
        super(KMeansTransformer, self).__init__(f"kmeans_fold_{k_fold}", results_path)
        self._new_features = []
        self._input_columns = []
        self._error = None
        self._kmeans = None
        self._scale = None
        self._model_name = model_name
        self._k_fold = k_fold
        self.load()

    def fit(self, X: DataFrame, y: DataFrame = None, **fit_params):
        if self._new_features:
            return
        if self._error is not None and self._error:
            raise AutoMLException(
                "KMeans Features not created due to error (please check errors.md). "
                + self._error
            )
            return
        if X.shape[1] == 0:
            self._error = f"KMeans not created. No continous features. Input data shape: {X.shape}, {y.shape}"
            raise AutoMLException("KMeans Features not created. No continous features.")

        start_time = time.time()

        n_clusters = int(np.log10(X.shape[0]) * 8)
        n_clusters = max(8, n_clusters)
        n_clusters = min(n_clusters, X.shape[1])

        self._input_columns = X.columns.tolist()
        # scale data
        self._scale = StandardScaler(copy=True, with_mean=True, with_std=True)
        X = self._scale.fit_transform(X)

        # Kmeans
        self._kmeans = kmeans = MiniBatchKMeans(n_clusters=n_clusters, init="k-means++")
        self._kmeans.fit(X)
        self._create_new_features_names()

        # print(
        #    f"Created {len(self._new_features)} KMeans Features in {np.round(time.time() - start_time,2)} seconds."
        # )

    def _create_new_features_names(self):
        n_clusters = self._kmeans.cluster_centers_.shape[0]
        self._new_features = [f"Dist_Cluster_{i}" for i in range(n_clusters)]
        self._new_features += ["Cluster"]

    def transform(self, X: DataFrame) -> DataFrame:
        if self._kmeans is None:
            raise AutoMLException("KMeans not fitted")

        # scale
        X_scaled = self._scale.transform(X[self._input_columns])

        # kmeans
        distances = self._kmeans.transform(X_scaled)
        clusters = self._kmeans.predict(X_scaled)

        X[self._new_features[:-1]] = distances
        X[self._new_features[-1]] = clusters

        return X

    def load(self):
        if super(KMeansTransformer, self).load():
            self._create_new_features_names()
