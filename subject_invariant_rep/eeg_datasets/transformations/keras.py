import sys

from tensorflow.keras import layers

import numpy as np

from scipy.signal import iirnotch, lfilter
import tensorflow as tf
from numpy.random import default_rng
import math


class SensorDropout(layers.Layer):
    """
    A random subset of sensors is replaced with zeros.
    Reference : https://arxiv.org/pdf/2007.04871.pdf
    Section 3.2

    Args:
        sensor_dropout (float): fraction of sensors to be dropped

    """

    def __init__(self, fraction=0.5, channels=8):
        super(SensorDropout, self).__init__()
        assert 0.0 <= fraction <= 1.0, "Dropout ratio must be between 0 and 1"
        self.sensor_dropout = fraction
        self.channels = channels

    @property
    def summarize(self):
        return {
            "name": "SensorDropout",
            "params": {
                "sensor_dropout": self.sensor_dropout,

            }
        }

    def call(self, x):
        x_aug = x
        rng = default_rng()
        size = math.ceil(self.channels * self.sensor_dropout)
        a = rng.choice(self.channels, size=size, replace=False)
        assert len(a) == np.unique(a).shape[0]

        for channel in a:
            # scatter and nd all values for each channel
           #  x_aug = tf.tensor_scatter_nd_update(x_aug, [[0, channel, 0, 0]], [0.0])


            exit()
            # x_aug = x_aug[:, channel, :, :].assign(0.0)

        return x_aug


class GaussianNoise(layers.Layer):
    """
    Independent and identically distributed Gaussian noise is added to the signal
    Reference : https://arxiv.org/pdf/2007.04871.pdf
    Section 3.2

    Args:
        low (float): lower bound of the standard deviation of the Gaussian noise
        high (float): upper bound of the standard deviation of the Gaussian noise
    """

    def __init__(self, low=0.0, high=0.2):
        super(GaussianNoise, self).__init__()
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

    def call(self, x):
        x_aug = x
        std_val = np.random.uniform(*self.std_range)
        noise = tf.random.normal(shape=tf.shape(x_aug), stddev=std_val)

        return x_aug + noise


class BandStopFilter(layers.Layer):
    """
    The signal content at a randomly selected frequency band is filtered out using a bandstop filter.
    Reference : https://arxiv.org/pdf/2007.04871.pdf
    Section 3.2

    Args:
        sfreq (float): sampling frequency of the signal
        low_freq (float): lower bound of the frequency band
        high_freq (float): upper bound of the frequency band
        bw (float): bandwidth of the bandstop filter
    """

    def __init__(self, sfreq, low_freq=0.5, high_freq=30.0, bw=5):
        super(BandStopFilter, self).__init__()
        self.freq_range = [low_freq, high_freq]
        self.bw = bw
        self.sfreq = sfreq

    @property
    def summarize(self):
        return {
            "name": "BandStopFilter",
            "params": {
                "low_freq": self.freq_range[0],
                "high_freq": self.freq_range[1],
                "bw": self.bw,
                "sfreq": self.sfreq,

            }
        }

    def call(self, signals):
        batch_size = tf.shape(signals)[0]
        channels = tf.shape(signals)[1]
        samples = tf.shape(signals)[2]
        x_aug = tf.squeeze(signals)
        augmentation_val = np.random.uniform(*self.freq_range)
        augmentation_val = np.min([augmentation_val, (self.sfreq / 2) - 1])
        b, a = iirnotch(augmentation_val, augmentation_val / self.bw, self.sfreq)
        x_aug = lfilter(b, a, x_aug)
        return tf.expand_dims(x_aug, -1)
