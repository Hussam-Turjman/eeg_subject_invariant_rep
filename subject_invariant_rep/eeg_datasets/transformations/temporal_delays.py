import torch
import numpy as np


class TemporalDelays(object):
    """
    The time-series data is randomly delayed in time
    Reference : https://arxiv.org/pdf/2007.04871.pdf
    Section 3.2
    """

    def __init__(self, sfreq, range_duration, step=0.5, fixed_duration=None):
        self.sampling_rate = sfreq
        self.time_shift_range = range_duration
        self.fixed_duration = fixed_duration
        if self.fixed_duration is None:
            assert self.time_shift_range is not None
            assert range_duration > 0
            assert self.sampling_rate > 0
            assert step > 0
            self.time_shift_range = np.arange(-range_duration, step + range_duration, step)
        else:
            self.time_shift_range = [self.fixed_duration]

    @property
    def summarize(self):
        return {
            "name": "TemporalDelays",
            "params": {
                "start": self.time_shift_range[0],
                "end": self.time_shift_range[-1],
                "fixed_duration": self.fixed_duration,
                "sampling_rate": self.sampling_rate,
            }
        }

    def __call__(self, x: torch.Tensor):
        """
        Args: X (tensor): Tensor eeg signal of size (batch_size, channels, features)
        """
        assert type(x) is torch.Tensor

        augmentation_val = np.random.choice(self.time_shift_range)

        number_of_samples = x.size(-1)
        window_len = number_of_samples / self.sampling_rate
        if augmentation_val > window_len:
            augmentation_val = 0

        random_idx = int(augmentation_val * self.sampling_rate)

        if random_idx != 0:
            new_data = torch.zeros_like(x)
            if augmentation_val < 0:
                new_data[:, :, :random_idx] = x[:, :, np.abs(random_idx):]
                new_data[:, :, random_idx:] = x[:, :, :np.abs(random_idx)]
            else:
                new_data[:, :, random_idx:] = x[:, :, :-random_idx]
                new_data[:, :, :random_idx] = x[:, :, -random_idx:]
            x_aug = new_data
        else:
            x_aug = x
        return x_aug


__all__ = ["TemporalDelays"]
