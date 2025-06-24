import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from supervised.preprocessing.dim_reducer.PCATransformer import PCATransformer
from supervised.automl import AutoML


def test_pca_transformer():
    df = pd.DataFrame(np.random.randn(50, 4), columns=["a", "b", "c", "d"])
    transformer = PCATransformer(variance_threshold=0.9)
    transformer.fit(df)
    df_out = transformer.transform(df)

    # Output for control
    print("\nOriginal Columns:", df.columns.tolist())
    print("New Features:", transformer._new_features)
    print("Transformed Columns:", df_out.columns.tolist())
    print("Transformed Data (first rows):\n", df_out.head())

    assert df_out.shape[0] == df.shape[0]
    assert any(col.startswith("PC_") for col in df_out.columns)
    assert not df_out.isnull().values.any()


def test_automl_pca():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
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
        dim_reduction_method="pca",
    )

    automl.fit(X_train, y_train)

    predictions = automl.predict(X_test)

    print(predictions)
