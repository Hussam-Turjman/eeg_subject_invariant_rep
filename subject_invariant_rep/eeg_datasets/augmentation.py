from __future__ import annotations
from typing import List, Callable

import numpy as np
import torch
from random import shuffle


class EEGAugmentation(object):

    def __init__(self, transformations: List[Callable],
                 return_original: bool = False,
                 augmentation_count: int = 2,
                 random_order: bool = True
                 ):
        assert len(transformations) > 0, "Expected at least one transformation"
        assert augmentation_count > 0, "Expected to return at least one augmentation"
        self._transformations = transformations
        self._return_original = return_original
        self._augmentation_count = augmentation_count
        self._random_order = random_order

    @property
    def summarize(self):
        return {
            "return_original": self._return_original,
            "augmentation_count": self._augmentation_count,
            "random_order": self._random_order,
            "transformations": [t.summarize for t in self._transformations]
        }

    def _apply_transformation(self, x: torch.Tensor):
        if len(x.shape) > 3:
            raise ValueError(f"Expected 3D tensor  --> (samples,channels,features) . Got {x.shape}")
        squeeze = False
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            squeeze = True
        x_aug = x.clone()

        assert len(x_aug.shape) == 3, f"Expected 3D tensor  --> (samples,channels,features) . Got {x_aug.shape}"

        if self._random_order:
            shuffle(self._transformations)

        for transformation in self._transformations:
            x_aug = transformation(x_aug)
            assert type(x_aug) is torch.Tensor, f"Expected tensor. Got {type(x_aug)}"

        if squeeze:
            x_aug = x_aug.squeeze(0)
        return x_aug

    def __call__(self, sample):
        X, y, subject_id = sample
        if type(X) is np.ndarray:
            X = torch.tensor(X, dtype=torch.float32)

        x_augmented = []

        if self._return_original:
            x_augmented.append(X)

        for _ in range(self._augmentation_count):
            x_aug = self._apply_transformation(X)
            x_augmented.append(x_aug)

        return x_augmented, y, subject_id


__all__ = ["EEGAugmentation"]
