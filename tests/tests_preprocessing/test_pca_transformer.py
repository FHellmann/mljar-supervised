import pandas as pd
import numpy as np
from supervised.preprocessing.dim_reducer.PCATransformer import PCATransformer

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
