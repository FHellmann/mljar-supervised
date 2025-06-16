import pandas as pd
import numpy as np
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


def test_automl_with_pca():
    np.random.seed(42)
    df = pd.DataFrame(np.random.randn(100, 5), columns=["a", "b", "c", "d", "e"])
    df["target"] = np.random.choice([0, 1], size=100)

    X = df.drop(columns=["target"])
    y = df["target"]

    automl = AutoML(algorithms=["Linear"], use_pca=False)

    automl.fit(X, y)

    features = automl._data_info.get_final_features()
    print("Features after PCA:", features)

    assert any(
        f.startswith("PC_") for f in features
    ), "No PCA components in feature set!"

    print("[OK] AutoML with use_pca=True generated and trained PCA components.")
