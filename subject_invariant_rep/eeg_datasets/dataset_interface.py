from __future__ import annotations

import pickle
import typing
from abc import ABC, abstractmethod
from typing import List, Literal

import mne
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .encoder import LabelsEncoder
from ..constants import *
from sklearn.model_selection import train_test_split, StratifiedKFold


def do_getitem(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    X = self.X[idx]
    Y = self.Y[idx]
    subject_id = self.subjects_ids[idx]
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.int64)
    subject_id = torch.tensor(subject_id, dtype=torch.int64)
    sample = (X, Y, subject_id)
    if self.transform is not None:
        sample = self.transform(sample)
    return sample


class EEGSubset(Dataset):
    def __init__(self, X, Y, subjects_ids, transform):
        assert len(X) == len(Y) == len(subjects_ids)
        self.X = X
        self.Y = Y
        self.subjects_ids = subjects_ids
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return do_getitem(self, idx)

    def dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, **kwargs)


class EEGDatasetInterface(Dataset, ABC):
    def __init__(self, subjects: List[int | str],
                 root_dir: str = None,
                 subjects_limit=None,
                 selected_classes: List[str] = None,
                 window_len=WINDOW_LEN,
                 sampling_rate=SAMPLING_RATE,
                 l_freq=LOW_FREQ,
                 h_freq=HIGH_FREQ,
                 scalings="mean",
                 transform=None,
                 merge_channels=False,
                 grid=False,
                 subject_id_encoder_type: Literal['label', 'onehot', 'ordinal'] = 'onehot',
                 class_encoder_type: Literal['label', 'onehot', 'ordinal'] = 'onehot',

                 features_order: Literal['TC', 'CT'] = "CT"  # TC - time x channels, CT - channels x time
                 ):
        if grid and merge_channels:
            raise ValueError("Cannot merge channels if grid is True.")
        if grid:
            if not np.sqrt(window_len * sampling_rate).is_integer():
                raise ValueError("Window length * sampling rate must be a square number.")
        if subjects_limit is not None:
            assert subjects_limit > 0, "Subjects limit must be greater than 0."
            subjects = subjects[:subjects_limit]

        self.__loaded = False
        self._montage = None
        self.features_order = features_order
        self.subjects_limit = subjects_limit

        self.class_encoder_type = class_encoder_type
        self.subject_id_encoder_type = subject_id_encoder_type
        self.root_dir = root_dir
        self.subjects = subjects
        self.window_len = window_len
        self.sampling_rate = sampling_rate
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.scalings = scalings
        self.transform = transform

        self.merge_channels = merge_channels
        self.grid = grid

        self.X = []
        self.Y = []
        self.subjects_ids = []
        self._subjects_ranges = None
        self._task_ranges = None

        self._selected_classes = selected_classes
        self._labels = None

        self.class_encoder = LabelsEncoder(encoder_type=self.class_encoder_type)
        self.class_encoder.fit(self.labels)

        self.subject_id_encoder = LabelsEncoder(encoder_type=self.subject_id_encoder_type)
        self.subject_id_encoder.fit(self.subject_id_labels)

        # self.raw_info = mne.create_info(self.channels_names, self.sampling_rate, ["eeg"] * self.channels_count)
        # self.raw_info.set_montage('standard_1020')
        # print(f"ALL : {self.raw_info}")
        self.raw_info = None

        self._load()

        self.scaler = mne.decoding.Scaler(info=self.raw_info,
                                          scalings=self.scalings)
        self.X = np.array(self.X, dtype='object')
        self.X = self.X.astype('float32')
        self.Y = np.array(self.Y)
        self.Y = self.class_encoder.transform(self.Y)
        self.Y = self.Y.astype('int32')
        self.subjects_ids = np.array(self.subjects_ids)
        self.subjects_ids = self.subject_id_encoder.transform(self.subjects_ids)
        self.subjects_ids = self.subjects_ids.astype('int32')

        # original_shape = self.X.shape
        # data = np.hstack(self.X)
        # data = self.scaler.fit_transform(data.T).T[0]
        # self.X = data.reshape(original_shape)

        if self.features_order == "CT":
            pass
        elif self.features_order == "TC":
            self.X = np.swapaxes(self.X, 1, 2)
        else:
            raise ValueError("Features order must be 'CT' or 'TC'.")
        self.__loaded = True
        print(self)

    @property
    def montage(self):
        return self._montage

    @property
    def subjects_ranges(self):
        if self._subjects_ranges is not None:
            return self._subjects_ranges
        self._subjects_ranges = []
        return self._subjects_ranges

    @property
    @abstractmethod
    def task_ranges(self) -> dict:
        pass

    @property
    def summarize(self):
        return {
            "subjects": self.subjects,
            "root_dir": self.root_dir,
            "subjects_limit": self.subjects_limit,
            "window_len": self.window_len,
            "sampling_rate": self.sampling_rate,
            "l_freq": self.l_freq,
            "h_freq": self.h_freq,
            "scalings": self.scalings,
            "transform": self.transform if self.transform is None else self.transform.summarize,
            "class_distribution": self.class_distribution,
            "merge_channels": self.merge_channels,
            "grid": self.grid,
            "labels": self.labels,
            "task_2_label": self.task_2_label,
            "subject_id_labels": self.subject_id_labels,
            "subject_id_2_label": self.subject_id_2_label,
            "class_encoder_type": self.class_encoder_type,
            "subject_id_encoder_type": self.subject_id_encoder_type,
        }

    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    @classmethod
    def load(cls, path):
        return pickle.load(open(path, "rb"))

    @abstractmethod
    def _load(self):
        pass

    @property
    def y_as_labels(self):
        return self.class_encoder.inverse_transform(self.Y).flatten()

    def inverse_class_labels(self, y):
        return self.class_encoder.inverse_transform(y).flatten()

    @property
    def subject_id_as_labels(self):
        return self.subject_id_encoder.inverse_transform(self.subjects_ids).flatten()

    def inverse_subject_id_labels(self, y):
        return self.subject_id_encoder.inverse_transform(y).flatten()

    @property
    def class_distribution(self):
        labels = self.y_as_labels
        unique, counts = np.unique(labels, return_counts=True)
        dist = {class_name: 0 for class_name in self.labels}
        for idx, label in enumerate(unique):
            dist[label] = int(counts[idx])
        return dist

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def __str__(self):
        return "\n" \
               f"Channels : {self.channels_count}\n" \
               f"Chunk size : {self.chunk_size}\n" \
               f"Window length : {self.window_len}\n" \
               f"Sampling rate : {self.sampling_rate}\n" \
               f"Bandpass Low frequency : {self.l_freq}\n" \
               f"Bandpass High frequency : {self.h_freq}\n" \
               f"Samples : {self.samples_count}\n" \
               f"Subjects : {self.subjects_count}\n" \
               f"Labels : {self.labels}\n" \
               f"Subject id labels : {self.subject_id_labels}\n" \
               f"Class distribution : {self.class_distribution}\n"

    @property
    def last_dimension_size(self):
        return self.X.shape[-1]

    @property
    def subjects_count(self):
        return len(self.subjects)

    @property
    def samples_count(self):
        return len(self.X)

    @property
    def chunk_size(self):
        return int(self.window_len * self.sampling_rate)

    @property
    def num_classes(self) -> int:
        return len(self.class_encoder.classes)

    @property
    @abstractmethod
    def channels_names(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def task_2_label(self) -> dict:
        pass

    @property
    @abstractmethod
    def labels(self) -> List[str]:
        pass

    @property
    def subject_id_labels(self) -> List[str]:
        return [f"sub_{i}" for i in range(self.subjects_count)]

    @property
    def subject_id_2_label(self) -> dict:
        mapping = {i: f"sub_{i}" for i in range(self.subjects_count)}
        return mapping

    def get_task_data(self, task: str, trials: typing.List[int] = None):
        task_range = self.task_ranges[task]

        out = []
        for subject_id in task_range.keys():
            if trials is None:
                for trial in task_range[subject_id]:
                    out.extend(trial)
            else:
                for trial_idx in trials:
                    trial = task_range[subject_id][trial_idx]
                    out.extend(trial)
        task_range = out

        return self.X[task_range], self.Y[task_range], self.subjects_ids[task_range]

    def get_subject_data(self, subject_idx: int):
        assert 0 <= subject_idx < self.subjects_count, f"Subject index must be less than {self.subjects_count} got {subject_idx}"

        subject_range = self.subjects_ranges[subject_idx]
        X = self.X[subject_range["start"]:subject_range["end"]]
        Y = self.Y[subject_range["start"]:subject_range["end"]]
        subjects_ids = self.subjects_ids[subject_range["start"]:subject_range["end"]]
        return X, Y, subjects_ids

    def get_subject_task_data(self, subject_idx: int, task: str, trials: typing.List[int] = None):
        X, Y, subjects_ids = self.get_task_data(task, trials=trials)
        encoded_subject_id = self.subject_id_encoder.transform(np.array([self.subject_id_labels[subject_idx]]))

        indices = np.where(subjects_ids == encoded_subject_id.flatten()[0])[0]
        return X[indices], Y[indices], subjects_ids[indices]

    @property
    @abstractmethod
    def channels_count(self) -> int:
        pass

    @property
    def grid_shape(self):
        return int(np.sqrt(self.chunk_size)), int(np.sqrt(self.chunk_size))

    def _get_segment(self, data, start_idx, end_idx) -> np.ndarray | None:
        segment = data[:, start_idx:end_idx]
        if self.merge_channels:
            segment = segment.flatten()
            if segment.shape[0] != (self.chunk_size * self.channels_count):
                return None
        else:
            if segment.shape[1] != self.chunk_size:
                return None
        if self.grid:
            segment = segment.reshape(segment.shape[0], self.grid_shape[0], self.grid_shape[1])
        return segment

    def __len__(self):
        return self.samples_count

    def __getitem__(self, idx):
        return do_getitem(self, idx)

    def dataloader(self, *args, **kwargs):
        return DataLoader(self, *args, **kwargs)

    def kfold(self, k=5, validation_size=0.30, shuffle=True, random_state=42, transform_test_set=False):
        cv = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)
        for train_idx, test_idx in cv.split(self.X, self.y_as_labels):
            X_train = self.X[train_idx]
            Y_train = self.Y[train_idx]
            subjects_ids_train = self.subjects_ids[train_idx]

            X_train, X_val, y_train, y_val, subjects_ids_train, subjects_ids_val = train_test_split(
                X_train, Y_train, subjects_ids_train,
                test_size=validation_size,
                shuffle=True,
                random_state=random_state)

            X_test = self.X[test_idx]
            Y_test = self.Y[test_idx]
            subjects_ids_test = self.subjects_ids[test_idx]

            train_dataset = EEGSubset(X_train, y_train, subjects_ids_train, self.transform)
            val_dataset = EEGSubset(X_val, y_val, subjects_ids_val, self.transform)
            test_dataset = EEGSubset(X_test, Y_test, subjects_ids_test,
                                     transform=self.transform if transform_test_set else None)
            yield train_dataset, val_dataset, test_dataset

    def leave_one_out(self, validation_size=0.30, random_state=42, transform_test_set=False, shuffle=True):
        for idx, test_subject_range in enumerate(self.subjects_ranges):
            train_ranges = self.subjects_ranges.copy()
            train_ranges.pop(idx)

            X_train = self.X.copy()
            Y_train = self.Y.copy()
            subjects_ids_train = self.subjects_ids.copy()
            to_remove = np.arange(test_subject_range["start"], test_subject_range["end"])

            X_train = np.delete(X_train, to_remove, axis=0)
            Y_train = np.delete(Y_train, to_remove, axis=0)
            subjects_ids_train = np.delete(subjects_ids_train, to_remove, axis=0)

            X_train, X_val, y_train, y_val, subjects_ids_train, subjects_ids_val = train_test_split(
                X_train, Y_train, subjects_ids_train,
                test_size=validation_size,
                shuffle=True,
                random_state=random_state)

            X_test = self.X[test_subject_range["start"]:test_subject_range["end"]]
            Y_test = self.Y[test_subject_range["start"]:test_subject_range["end"]]
            subjects_ids_test = self.subjects_ids[test_subject_range["start"]:test_subject_range["end"]]

            train_dataset = EEGSubset(X_train, y_train, subjects_ids_train, self.transform)
            val_dataset = EEGSubset(X_val, y_val, subjects_ids_val, self.transform)
            test_dataset = EEGSubset(X_test, Y_test, subjects_ids_test,
                                     transform=self.transform if transform_test_set else None)

            yield train_dataset, val_dataset, test_dataset


__all__ = ["EEGDatasetInterface", "EEGSubset"]
