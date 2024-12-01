import torch
import numpy as np

from numpy.random import default_rng
import math


class SensorDropout(object):
    """
    A random subset of sensors is replaced with zeros.
    Reference : https://arxiv.org/pdf/2007.04871.pdf
    Section 3.2

    Args:
        sensor_dropout (float): fraction of sensors to be dropped

    """

    def __init__(self, fraction=0.5):
        assert 0.0 <= fraction <= 1.0, "Dropout ratio must be between 0 and 1"
        self.sensor_dropout = fraction

    @property
    def summarize(self):
        return {
            "name": "SensorDropout",
            "params": {
                "sensor_dropout": self.sensor_dropout,

            }
        }

    def __call__(self, x: torch.Tensor):
        """
        Args: X (tensor): Tensor eeg signal of size (batch_size, channels, features)
        """
        assert type(x) is torch.Tensor
        x_aug = x
        num_sensors = x.size(1)
        rng = default_rng()

        a = rng.choice(num_sensors, size=math.ceil(num_sensors * self.sensor_dropout), replace=False)
        assert len(a) == np.unique(a).shape[0]

        for channel in a:
            x_aug[:, channel, :] = 0.0

        return x_aug


__all__ = ["SensorDropout"]
