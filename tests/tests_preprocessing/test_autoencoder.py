import unittest
import torch
import numpy as np

from supervised.preprocessing.dim_reducer.AutoencoderModel import Autoencoder


class AutoencoderTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.input_dim = 512
        cls.params = {
            "layer_config": [256, 128],
            "bottleneck_dim": 64,
            "epochs": 2,
            "batch_size": 16,
            "lr": 0.001,
            "device": "cpu",
            "verbose": False,
        }
        cls.model = Autoencoder(cls.input_dim, cls.params)
        cls.X = np.random.rand(100, cls.input_dim).astype(np.float32)

    def test_init(self):
        self.assertTrue(hasattr(self.model, "encoder"))
        self.assertTrue(hasattr(self.model, "decoder"))

    def test_encode_dimension(self):
        x_tensor = torch.tensor(self.X)
        encoded = self.model.encode(x_tensor)
        self.assertEqual(encoded.shape[1], self.params["bottleneck_dim"])

    def test_forward_output_shape(self):
        x_tensor = torch.tensor(self.X)
        output = self.model(x_tensor)
        self.assertEqual(output.shape[1], self.input_dim)
        self.assertEqual(output.shape[0], self.X.shape[0])

    def test_train_model_runs(self):
        try:
            self.model.train_model(self.X, self.params)
        except Exception as e:
            self.fail(f"train_model raised an exception {e}")


if __name__ == "__main__":
    unittest.main()
