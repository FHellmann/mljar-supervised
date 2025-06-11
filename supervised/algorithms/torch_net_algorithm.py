import numpy as np
from supervised.algorithms.algorithm import BaseAlgorithm
from supervised.algorithms.torch_net import TorchNetwork
from supervised.utils.config import LOG_LEVEL
import logging

from supervised.algorithms.registry import (
    BINARY_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    AlgorithmsRegistry,
)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class TorchAlgorithm(BaseAlgorithm):
    algorithm_name = "TorchNetwork"
    algorithm_short_name = "TorchNet"

    def __init__(self, params):
        super().__init__(params)
        self.model = None
        self.is_model_fitted = False

    def fit(self, X, y, **kwargs):
        logger.debug("TorchAlgorithm.fit")

        # Initialize model if no model is initialized
        if self.model is None:
            input_dim = X.shape[1]
            output_dim = (
                1 if self.params.get("ml_task") == "binary" else len(np.unique(y))
            )

            torch_net_params = {
                "hidden_config": self.params.get(
                    "hidden_config", default_params["hidden_config"]
                ),
                "dropout": self.params.get("dropout", default_params["dropout"]),
                "dropout_rate": self.params.get(
                    "dropout_rate", default_params["dropout_rate"]
                ),
            }

            self.model = TorchNetwork(
                input_dim=input_dim,
                output_dim=output_dim,
                task_type=self.params.get("ml_task"),
                params=torch_net_params,
                target_specification=self.params.get("target_specification"),
            )

        # Training of torch network
        self.model.fit(
            x_train=X,
            y_train=y,
            epochs=self.params.get("epochs", 10),
            batch_size=self.params.get("batch_size", 32),
            lr=self.params.get("lr", 1e-3),
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

        input_dim = self.params.get("input_dim")
        output_dim = self.params.get("output_dim")
        if input_dim is None or output_dim is None:
            raise ValueError(
                "input_dim and output_dim must be set in params before loading"
            )
        self.model = TorchNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            task_type=self.params.get("ml_task"),
            params=self.params,
            target_specification=self.params.get("target_specification"),
        )
        self.model.load_state_dict(torch.load(model_file_path))
        self.model.network.eval()
        self.is_model_fitted = True
        self.model_file_path = model_file_path


torchnet_params = {
    # Training parameters
    "lr": [0.01, 0.001, 0.0001],
    "epochs": [50, 100],
    "batch_size": [16, 32, 64],
    # Architecture parameters
    "hidden_config": [{1: 128, 2: 64, 3: 32}, {1: 256, 2: 128, 3: 64, 4: 32}],
    "dropout": [True, False],
    "dropout_rate": [0.25, 0.5],
    # Classification parameters
    "target_specification": 1,
}

default_params = {
    "lr": 0.01,
    "epochs": 50,
    "batch_size": 32,
    "hidden_config": {1: 256, 2: 128, 3: 64, 4: 32},
    "dropout": False,
    "dropout_rate": 0.5,
    "target_specification": 1,
}


additional = {
    "max_rows_limit": 100000,
    "max_cols_limit": 200,
}

required_preprocessing = [
    "missing_values_inputation",
    "convert_categorical",
    "scale",
]

AlgorithmsRegistry.add(
    BINARY_CLASSIFICATION,
    TorchAlgorithm,
    torchnet_params,
    required_preprocessing,
    additional,
    default_params,
)

AlgorithmsRegistry.add(
    MULTICLASS_CLASSIFICATION,
    TorchAlgorithm,
    torchnet_params,
    required_preprocessing,
    additional,
    default_params,
)
