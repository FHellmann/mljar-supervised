import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from supervised.preprocessing.undersampler.CCTransformer import CCTransformer
from supervised.automl import AutoML


def test_clustercentroids_transformer():
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        weights=[0.9, 0.1],
        class_sep=0.8,
        random_state=42,
    )

    X = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    y = pd.Series(y)

    transformer = CCTransformer()
    transformer.fit(X, y)
    X_res, y_res = transformer.transform(X, y)

    print("Original class distribution:\n", y.value_counts())
    print("Resampled class distribution:\n", y_res.value_counts())

    assert len(X_res) == len(y_res), "Feature and target length mismatch after ClusterCentroids"
    assert len(X_res) <= len(X), "ClusterCentroids should not increase the number of samples"
    assert y_res.value_counts()[0] == y_res.value_counts()[1], "ClusterCentroids should balance the classes"


def test_automl_clustercentroids():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        weights=[0.85, 0.15],
        class_sep=0.6,
        random_state=42,
    )

    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    automl = AutoML(
        mode="Explain",
        total_time_limit=120,
        ml_task="binary_classification",
        undersampling_method="cc",
    )

    automl.fit(X_train, y_train)

    predictions = automl.predict(X_test)
    print("Predictions:", predictions[:10])

    assert len(predictions) == len(X_test), "Prediction length does not match test data"

