import logging

import torch
from torch import nn

from supervised.utils.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


class Autoencoder(nn.Module):
    """Autoencoder class for dimensionality reduction."""

    def __init__(self, input_dim: int, params):
        logger.debug("Autoencoder.__init__")
        # Call constructor of nn.Module (required for PyTorch Models)
        super().__init__()

        self.input_dim = input_dim
        self.layer_config = params.get("layer_config", [512, 256, 128])
        self.bottleneck_dim = params.get("bottleneck_dim", 64)

        if self.bottleneck_dim >= self.layer_config[-1]:
            logger.warning(
                "Bottleneck dim (%d) is greater than or equal to last encoder layer (%d). Reducing...",
                self.bottleneck_dim,
                self.layer_config[-1],
            )
            self.bottleneck_dim = self.layer_config[-1] // 2
            logger.info("New Bottleneck dim: %d", self.bottleneck_dim)

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        # Add all the layers to the encoder
        for units in self.layer_config:
            # Fully connected layers
            encoder_layers.append(nn.Linear(in_dim, units))
            # ReLu activation function
            encoder_layers.append(nn.ReLU())
            # Update the input dimension for the next layer
            in_dim = units
        # Last layer that reduces the dimensionality to the bottleneck_dim
        encoder_layers.append(nn.Linear(in_dim, self.bottleneck_dim))
        # Activation function for the bottleneck layer
        encoder_layers.append(nn.ReLU())
        # Pack the layers into a Sequential module to get the "encoder"
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = self.bottleneck_dim
        for units in reversed(self.layer_config):
            # Fully connected layers
            decoder_layers.append(nn.Linear(in_dim, units))
            # Activation function ReLu
            decoder_layers.append(nn.ReLU())
            # Update the dimension for the next layer
            in_dim = units
        # Last layer that maps the bottleneck dimension back to the input dimension
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        # Pack the layers into a Sequential module to get the "decoder"
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        Send data through the encoder and decoder.

        :return: The reconstructed data.
        """
        logger.debug("Autoencoder.forward")
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def encode(self, x: torch.Tensor):
        logger.debug("Autoencoder.encode")
        return self.encoder(x)

    # Getter for the encoder
    @property
    def encoder_net(self):
        return self.encoder


autoencoder_params = {
    "layer_config": [[512, 256, 128], [512, 256, 128, 64], [512, 256, 128, 64, 32]],
    "bottleneck_dim": [64, 32, 16, 8],
}
