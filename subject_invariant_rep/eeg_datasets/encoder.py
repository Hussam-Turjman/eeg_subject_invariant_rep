from __future__ import annotations
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from typing import Literal, List

import numpy as np


class LabelsEncoder(object):
    encoder: OneHotEncoder | LabelEncoder | OrdinalEncoder

    def __init__(self, encoder_type: Literal['onehot', 'label', 'ordinal']):
        if encoder_type == 'onehot':
            self.encoder = OneHotEncoder(dtype=np.float32)
        elif encoder_type == 'label':
            self.encoder = LabelEncoder()
        elif encoder_type == 'ordinal':
            self.encoder = OrdinalEncoder(dtype=np.float32)
        else:
            raise ValueError(f"Unknown encoder type {encoder_type}")

        self.encoder_type = encoder_type

    @property
    def classes(self):
        if self.encoder_type == 'onehot' or self.encoder_type == 'ordinal':
            classes = self.encoder.categories_
        elif self.encoder_type == 'label':
            classes = self.encoder.classes_
        if type(classes[0]) is np.ndarray:
            classes = classes[0].tolist()
        return classes

    def fit(self, labels: List[str]):
        if self.encoder_type == 'onehot' or self.encoder_type == 'ordinal':
            self.encoder.fit(np.expand_dims(labels, axis=1))
        elif self.encoder_type == 'label':
            self.encoder.fit(labels)
        return self

    def transform(self, y: List[str] | np.ndarray | List[List[str]]):
        if type(y) is list:
            y = np.array(y)

        if len(y.shape) == 2:
            y = np.expand_dims(y, axis=2)
            y = np.concatenate(y, axis=0)
        elif len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)
        if self.encoder_type == "label":
            y = y.ravel()

        transformed = self.encoder.transform(y)
        if self.encoder_type == 'onehot':
            transformed = transformed.toarray()
        return transformed

    def inverse_transform(self, encoded: np.ndarray):
        return self.encoder.inverse_transform(encoded)


__all__ = ['LabelsEncoder']
