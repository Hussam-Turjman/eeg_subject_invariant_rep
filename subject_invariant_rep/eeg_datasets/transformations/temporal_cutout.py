import torch

import numpy as np


class TemporalCutout(object):
    """
    A random contiguous section of the time-series signal (cutout window) is
    replaced with zeros.
    Reference :
    https://arxiv.org/abs/1708.04552
    https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    """

    def __init__(self, sfreq: float, cutout_max_len=None):
        """
        Args:
            sfreq (float): sampling frequency
            cutout_max_len (int): maximum length of the cutout window
        """
        if cutout_max_len is None:
            cutout_max_len = sfreq // 2
        self.sfreq = sfreq
        self.cutout_max_len = cutout_max_len
        self.mask_start_range = [0, int(sfreq)]
        self.mask_end_range = [0, int(cutout_max_len)]

    @property
    def summarize(self):
        return {
            "name": "TemporalCutout",
            "params": {
                "sfreq": self.sfreq,
                "cutout_max_len": self.cutout_max_len
            }
        }

    def __call__(self, x: torch.Tensor):
        """
        Args: X (tensor): Tensor eeg signal of size (batch_size, channels, features)
        """
        assert type(x) is torch.Tensor
        x_aug = x
        num_features = x_aug.size(2)

        if num_features < self.mask_end_range[1]:
            self.mask_end_range[1] = num_features - 1

        if num_features < self.mask_start_range[1]:
            self.mask_start_range[1] = num_features - 1

        start = np.random.randint(*self.mask_start_range)
        end = np.random.randint(*self.mask_end_range)
        start_idx = np.min([start, end])
        end_idx = np.max([start, end])
        x_aug[:, :, start_idx:end_idx] = 0.0
        return x_aug


__all__ = ["TemporalCutout"]
