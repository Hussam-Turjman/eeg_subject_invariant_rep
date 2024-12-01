import torch
import numpy as np


class GaussianNoise(object):
    """
    Independent and identically distributed Gaussian noise is added to the signal
    Reference : https://arxiv.org/pdf/2007.04871.pdf
    Section 3.2

    Args:
        low (float): lower bound of the standard deviation of the Gaussian noise
        high (float): upper bound of the standard deviation of the Gaussian noise
    """

    def __init__(self, low=0.0, high=0.2):
        self.std_range = [low, high]

    @property
    def summarize(self):
        return {
            "name": "GaussianNoise",
            "params": {
                "low": self.std_range[0],
                "high": self.std_range[1],

            }
        }

    def __call__(self, x: torch.Tensor):
        """
        Args: X (tensor): Tensor eeg signal of size (batch_size, channels, features)
        """
        assert type(x) is torch.Tensor
        x_aug = x
        std_val = np.random.uniform(*self.std_range)
        noise = np.random.normal(0, std_val, size=x_aug.shape).astype(np.float32)

        return x_aug + torch.from_numpy(noise)


__all__ = ["GaussianNoise"]
