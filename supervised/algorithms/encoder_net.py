import logging
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from supervised.preprocessing.dim_reducer import AutoencoderModel
from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class EncoderClassificationNetwork(nn.Module):

    def __init__(self, autoencoder: AutoencoderModel, output_dim: int, params):
        """
        :param autoencoder: Autoencoder model that provides encoder layers and a bottleneck layer for the
        classification network.
        :param output_dim: Output dimension of the encoder network. 1 if binary classification task otherwise
                           number of classes.
        :param params: Dictionary of hyperparameters.
        """
        logger.debug("EncoderClassificationNetwork.__init__")

        super().__init__()
        self.device = params.get("device", "cpu")
        self.autoencoder = autoencoder
        self.output_dim = output_dim
        self.freeze_layers = int(params.get("freeze_layers", 1))

        # Classification layer
        self.classifier = nn.Linear(
            self.autoencoder.bottleneck_dim, 1 if output_dim == 1 else output_dim
        ).to(self.device)

        # Freeze layers
        self._freeze_layers(self.freeze_layers)

        # Loss and optimizer (initialization in fit method)
        self.criterion = None
        self.optimizer = None

    def _freeze_layers(self, freeze_layers: int):
        """
        Helper function to freeze a fixed number of layers.

        :param freeze_layers: Number of layers that will be "frozen" (0 = all trainable)
        """
        logger.debug("EncoderClassificationNetwork.freeze_layers")

        # Number of freeze layers must be less than or equal to encoder layers
        num_layers = len(list(self.autoencoder.encoder_net.children()))
        if freeze_layers > num_layers:
            logger.warning(
                "Number of freeze layers is greater than number of encoder layers."
            )
            freeze_layers = num_layers
            logger.info(
                f"Freeze layers set to {freeze_layers}. This corresponds to the number of encoder layers."
            )

        # First: Freeze all layers
        for param in self.autoencoder.encoder_net.parameters():
            param.requires_grad = False

        # Second: "Unfreezing" the specific number of layers
        if freeze_layers > 0:
            children = list(self.autoencoder.encoder_net.children())
            for layer in children[-freeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x):
        """
        Routes the data through the network.

        :param x: Input data.
        :return: The output of the network.
        """
        logger.debug("EncoderClassificationNetwork.forward")

        x = self.autoencoder.encoder_net(x)
        logits = self.classifier(x)
        return logits

    def fit(
        self,
        x_train,
        y_train,
        params,
        target_specification: Optional = None,
    ):
        """
        Training of the network.

        :param x_train: Input data for training.
        :param y_train: Target data for training.
        :param params: Dictionary of hyperparameters.
        :param target_specification: Target specification for training.
        """
        logger.debug("EncoderClassificationNetwork.fit")

        batch_size = params.get("batch_size", 32)
        epochs = params.get("epochs", 10)

        # Switch network to training mode
        self.train()
        # Prepare input and output data (convert NumPy Array to PyTorch tensor)
        x_tensor = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(
            self._label_output(y_train, target_specification),
            dtype=torch.float32 if self.output_dim == 1 else torch.long,
        ).to(self.device)

        # Build a dataloader for Mini-Batch
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Loss function depends on the classification task
        self.criterion = (
            nn.BCEWithLogitsLoss() if self.output_dim == 1 else nn.CrossEntropyLoss()
        )
        params_to_optimize = filter(lambda p: p.requires_grad, self.parameters())
        # Optimize algorithm Adam
        self.optimizer = Adam(params_to_optimize, lr=0.001)

        # Loop over epoch and Mini-Batches
        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                # Deletion of gradients
                self.optimizer.zero_grad()
                outputs = self(batch_x)
                # Fit data for the binary classification task and the BCEWithLogitsLoss
                if self.output_dim == 1:
                    outputs = outputs.view(-1)
                    loss = self.criterion(outputs, batch_y)
                else:
                    loss = self.criterion(outputs, batch_y)
                # Backpropagation
                loss.backward()
                # Parameter update
                self.optimizer.step()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        :param x: Input data for prediction.
        :return: Predictions for the input data as a NumPy Array.
        """
        logger.debug("EncoderClassificationNetwork.predict")

        # Switch model to evaluation mode
        self.eval()
        # Bring x data into the right format (Torch tensor)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        # No calculation of the gradient
        with torch.no_grad():
            # Calculate raw output data
            logits = self(x_tensor)

        # Process raw output data based on the classification task
        if self.output_dim == 1:
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            # Define a threshold
            return (probs > 0.5).astype(int)
        else:
            # Choose class with the highest value
            probs = torch.argmax(logits, dim=1).cpu().numpy()
            return probs

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate probabilities for the input data.

        :param x: Input data for prediction.
        :return: Probabilities for the input data as a NumPy Array.
        """
        logger.debug("EncoderClassificationNetwork.predict_proba")

        # Switch network to evaluation mode
        self.eval()
        # Bring x data into the right format
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        # Calculate raw output data
        with torch.no_grad():
            logits = self(x_tensor)

        # Process raw output based on the classification task.
        if self.output_dim == 1:
            probs = torch.sigmoid(logits).cpu().numpy()
            # Probabilities for class [0, 1]
            return np.hstack([1 - probs, probs])
        else:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs


encoder_net_params = {
    "freeze_layers": [1, 2, 3, 4, 5],
    "device": ["cpu", "cuda"],
    "epochs": [10, 25, 50, 100],
    "batch_size": [16, 32, 64, 128],
}


default_encoder_net_params = {
    "freeze_layers": 1,
    "device": "cpu",
    "epochs": [10],
    "batch_size": [32],
}
