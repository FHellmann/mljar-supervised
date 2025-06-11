import logging
from typing import Union, Optional
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from supervised.preprocessing.torch_preparer.DataPreparer import DataPreparer
from supervised.utils.model_interface import AbstractModel
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class TorchNetwork(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        task_type: str,
        params,
        target_specification: Optional = None,
    ):

        super(TorchNetwork, self).__init__()
        logger.debug("TorchNetwork.__init__")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.task_type = task_type
        self.target_specification = target_specification
        hidden_config = params.get("hidden_config", {1: 256, 2: 128, 3: 64, 4: 32})
        dropout = params.get("dropout", False)
        dropout_rate = params.get("dropout_rate", 0.5) if dropout else 0.0

        layers = []
        in_dim = input_dim
        for i in sorted(hidden_config.keys()):
            # Fully connected layer
            layers.append(nn.Linear(in_dim, hidden_config[i]))
            # Activation function ReLu
            layers.append(nn.ReLU())
            if dropout:
                # Add dropout if wanted
                layers.append(nn.Dropout(dropout_rate))
            # Update dimension for the next layer
            in_dim = hidden_config[i]
        layers.append(nn.Linear(in_dim, output_dim))
        # Summarize the layers in a model
        self.network = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        """
        Routes the input through the network.

        :param x: The input data.
        :return: The output of the network.
        """
        logger.debug("TorchNetwork.forward")
        return self.network(x)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
    ):
        """
        Training of the network with specific data.

        :param x_train: Input data for training.
        :param y_train: Target data for training.
        :param epochs: Number of epochs for training.
        :param batch_size: Batch size for training.
        :param lr: Learning rate for training.
        """

        logger.debug("TorchNetwork.fit")

        # Start signal for the training
        self.network.train()
        # Definition of the loss function based on the classification task
        criterion = (
            nn.BCEWithLogitsLoss() if self.output_dim == 1 else nn.CrossEntropyLoss()
        )
        # Usage of Adam as an optimizer
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        # Convert data into a suited format (torch tensor) with a helper method
        x_train = DataPreparer.prepare_input(x_train)
        y_train = DataPreparer.prepare_output(
            y_train,
            task_type=self.task_type,
            target_specification=self.target_specification,
        )

        # Convert data into a format necessary for PyTorch model training
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Train network for a fixed number of epochs
        for epoch in range(epochs):
            # Separate data into Mini-Batches
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                # Return the predictions (logits); remove unnecessary dimensions
                outputs = self.network(x_batch).squeeze(dim=-1)
                # Calculation of loss
                loss = criterion(outputs, y_batch)
                # Backpropagation
                loss.backward()
                # Actualization of the weights
                optimizer.step()

        return self

    def predict(self, x: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Realization of predictions.

        :param x: Input data for prediction.
        :return: Predictions for the input data as a NumPy Array.
        """

        logger.debug("TorchNetwork.predict")

        # Switch network to the evaluation mode
        self.network.eval()
        # Prepare input with the helper method
        x = DataPreparer.prepare_input(x)
        # No calculation of gradients (saves memory and computing time)
        with torch.no_grad():
            # Raw output of the network; remove unnecessary dimensions
            logits = self.network(x).squeeze()

        # Usage of different activation functions based on the classification task
        if self.output_dim == 1:
            return (torch.sigmoid(logits) > 0.5).int().numpy()
        else:
            return torch.argmax(logits, dim=1).numpy()
        # Usage of numpy() for compatibility with sklearn

    def predict_proba(self, x: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Prediction of network output as probabilities.

        :param x: Input data for prediction.
        :return: Probabilities for the input data as a NumPy Array.
        """

        logger.debug("TorchNetwork.predict_proba")

        # Switch network to evaluation mode
        self.network.eval()
        # Prepare input data with the helper method
        x = DataPreparer.prepare_input(x)
        # Do not calculate gradients
        with torch.no_grad():
            # Calculate raw output
            logits = self.network(x).squeeze()
            # Calculate probabilities based on the classification task.
            if self.output_dim == 1:
                probs = torch.sigmoid(logits)
                return probs.unsqueeze(1).numpy()
            else:
                return torch.softmax(logits, dim=1).numpy()
            # Usage of numpy() for compatibility with sklearn
