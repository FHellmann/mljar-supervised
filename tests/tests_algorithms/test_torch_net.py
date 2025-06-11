import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from supervised.algorithms.torch_net_algorithm import TorchAlgorithm


def test_torch_algorithm_basic():
    # Generate dummy data
    X = np.random.rand(100, 10).astype(np.float32)  # 100 Samples, 10 Features
    y = np.random.randint(0, 2, size=100)  # binary labels

    print("\n\nTarget labels: ", pd.unique(y))

    # Determine parameters
    params = {
        "ml_task": "binary",
        "lr": 0.01,
        "epochs": 5,
        "batch_size": 16,
        "hidden_config": {1: 64, 2: 32},
        "dropout": False,
        "dropout_rate": 0.5,
        "input_dim": 10,
        "output_dim": 1,
        "target_specification": 1,
    }

    # Initialize torch model / algorithm
    algo = TorchAlgorithm(params)

    # Training
    algo.fit(X, y)

    # Predictions
    preds = algo.predict(X)
    proba = algo.predict_proba(X)

    print("Predictions shape:", preds.shape)
    print("Probabilities shape:", proba.shape)

    assert preds.shape == (100,)
    assert proba.shape == (100, 1)

    print("Classification report:")
    print(classification_report(y, preds))

    print("Test was successful!")


if __name__ == "__main__":
    test_torch_algorithm_basic()
