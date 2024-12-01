import torch
import numpy as np

from scipy.signal import iirnotch, lfilter


class BandStopFilter(object):
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

    def __init__(self, sfreq, low_freq=2.0, high_freq=82.5, bw=5):
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

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: X (tensor): Tensor eeg signal of size (batch_size, channels, features)
        """
        assert type(x) is torch.Tensor
        x_aug = x
        augmentation_val = np.random.uniform(*self.freq_range)
        augmentation_val = np.min([augmentation_val, (self.sfreq / 2) - 1])
        b, a = iirnotch(augmentation_val, augmentation_val / self.bw, self.sfreq)
        x_aug = lfilter(b, a, x_aug)
        return torch.tensor(x_aug, dtype=torch.float32)


__all__ = ["BandStopFilter"]
