import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from supervised.algorithms.encoder_net_algorithm import EncoderNetAlgorithm


class DummyAutoencoder:
    def __init__(self, input_dim=10, bottleneck_dim=5):
        self.bottleneck_dim = bottleneck_dim

        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, 8), nn.ReLU(), nn.Linear(8, bottleneck_dim), nn.ReLU()
        )


def test_encoder_algorithm_basic():
    # Dummy-Daten erzeugen
    X = np.random.rand(100, 10).astype(np.float32)
    y = np.random.randint(0, 5, size=100)  # labels

    print("\n\nTarget labels:", pd.unique(y))

    # Dummy-Autoencoder erzeugen
    autoencoder = DummyAutoencoder(input_dim=10, bottleneck_dim=5)

    # Parameter setzen
    params = {
        "ml_task": "binary",
        "lr": 0.01,
        "epochs": 5,
        "batch_size": 16,
        "freeze_layers": 1,
        "target_specification": 1,
        "device": "cpu",
        "autoencoder": autoencoder,
    }

    # Algorithmus initialisieren
    algo = EncoderNetAlgorithm(params)

    # Training
    algo.fit(X, y)

    # Vorhersagen
    preds = algo.predict(X)
    proba = algo.predict_proba(X)

    print("Predictions shape:", preds.shape)
    print("Probabilities shape:", proba.shape)

    assert preds.shape == (100,)
    assert proba.shape == (100, 2)
    print(
        "First five probabilities for other classes (0) and target class(1): \n",
        proba[:5],
    )

    y_binary = [1 if value == 1 else 0 for value in y]
    print("First five class labels in binary format: ", y_binary[:5])

    print("Classification report:")
    print(classification_report(y_binary, preds))

    print("Test was successful!")


if __name__ == "__main__":
    test_encoder_algorithm_basic()
