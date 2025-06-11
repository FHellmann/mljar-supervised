import numpy as np
import logging
from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.encoder_net import (
    EncoderClassificationNetwork,
)
from supervised.utils.config import LOG_LEVEL
from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    AlgorithmsRegistry,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class EncoderNetAlgorithm(BaseAlgorithm):
    algorithm_name = "EncoderNetwork"
    algorithm_short_name = "EncNet"

    def __init__(self, params):
        super().__init__(params)
        self.model = None
        self.is_model_fitted = False
        self.autoencoder = params.get("autoencoder")

    def fit(self, X, y, **kwargs):
        logger.debug("EncoderNetAlgorithm.fit")

        if self.autoencoder is None:
            raise ValueError("Autoencoder must be passed via params['autoencoder'].")

        output_dim = 1 if self.params.get("ml_task") == "binary" else len(np.unique(y))

        self.model = EncoderClassificationNetwork(
            autoencoder=self.autoencoder, output_dim=output_dim, params=self.params
        )

        self.model.fit(
            x_train=X,
            y_train=y,
            params=self.params,
        )

        self.is_model_fitted = True
        return self

    def predict(self, X):
        if not self.is_model_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_model_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict_proba(X)

    def is_fitted(self):
        return self.is_model_fitted

    def save(self, model_file_path):
        import torch

        torch.save(self.model.state_dict(), model_file_path)
        self.model_file_path = model_file_path

    def load(self, model_file_path):
        import torch

        if self.autoencoder is None:
            raise ValueError("Autoencoder must be set before loading.")
        output_dim = self.params.get("output_dim")
        if output_dim is None:
            raise ValueError("output_dim must be set in params.")
        self.model = EncoderClassificationNetwork(
            autoencoder=self.autoencoder,
            output_dim=output_dim,
            params=self.params,
        )
        self.model.load_state_dict(torch.load(model_file_path))
        self.model.eval()
        self.is_model_fitted = True
        self.model_file_path = model_file_path


# Parameter definition
encoder_net_params = {
    "lr": [0.01, 0.001, 0.0001],
    "epochs": [25, 50, 100],
    "batch_size": [16, 32, 64],
    "freeze_layers": [1, 2, 3, 4, 5],
    "target_specification": 1,
    "device": ["cpu", "cuda"],
}

default_encoder_net_params = {
    "lr": 0.001,
    "epochs": 50,
    "batch_size": 32,
    "freeze_layers": 2,
    "target_specification": 1,
    "device": "cpu",
}

required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "scale",
]

additional = {
    "max_rows_limit": 100000,
    "max_cols_limit": 200,
}

# Registration
AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    EncoderNetAlgorithm,
    encoder_net_params,
    required_preprocessing,
    additional,
    default_encoder_net_params,
)

AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    EncoderNetAlgorithm,
    encoder_net_params,
    required_preprocessing,
    additional,
    default_encoder_net_params,
)
