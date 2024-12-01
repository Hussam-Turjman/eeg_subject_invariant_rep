# EEG Motor Movement/Imagery Dataset
# https://physionet.org/content/eegmmidb/1.0.0/
from __future__ import annotations

from typing import List, Literal

import mne
import numpy as np
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import read_raw_edf

from .dataset_interface import EEGDatasetInterface
from ..constants import *


class MMIDataset(EEGDatasetInterface):
    all_subjects = list(range(1, 110))

    runs = [
        1,  # Baseline, eyes open
        2,  # Baseline, eyes closed
        3,  # Task 1 (open and close left or right fist)
        4,  # Task 2 (imagine opening and closing left or right fist)
        5,  # Task 3 (open and close both fists or both feet)
        6,  # Task 4 (imagine opening and closing both fists or both feet)
    ]
    task_duration = 90  # seconds
    baseline_duration = 60  # seconds

    def __init__(self,
                 subjects: List[int | str],
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

                 features_order="CT"):
        super(MMIDataset, self).__init__(subjects=subjects,
                                         root_dir=root_dir,
                                         subjects_limit=subjects_limit,
                                         selected_classes=selected_classes,
                                         window_len=window_len,
                                         sampling_rate=sampling_rate,
                                         l_freq=l_freq,
                                         h_freq=h_freq,
                                         scalings=scalings,
                                         transform=transform,

                                         merge_channels=merge_channels,
                                         grid=grid,
                                         subject_id_encoder_type=subject_id_encoder_type,
                                         class_encoder_type=class_encoder_type,

                                         features_order=features_order)

    @property
    def channels_count(self) -> int:
        return 64

    @property
    def name(self) -> str:
        return "MMIDataset"

    @property
    def labels(self) -> List[str]:
        if self._labels is not None:
            return self._labels

        labels = ["open", "closed", "t1", "t2", "t3", "t4"]

        if self._selected_classes is not None:
            self._selected_classes = list(set(self._selected_classes))
            assert set(self._selected_classes) <= {"open", "closed", "t1", "t2", "t3", "t4"}
            labels = self._selected_classes
        self._labels = sorted(labels)
        return self._labels

    @property
    def channels_names(self) -> List[str]:
        return ['Fp1', 'Fpz', 'Fp2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFz', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', 'F9',
                'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'F10', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz',
                'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10',
                'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P9', 'P7', 'P5', 'P3',
                'P1', 'Pz']

    @property
    def task_ranges(self) -> dict:
        if self._task_ranges is not None:
            return self._task_ranges
        self._task_ranges = {
            "open": {},
            "closed": {},
            "t1": {},
            "t2": {},
            "t3": {},
            "t4": {}
        }
        return self._task_ranges

    @property
    def task_2_label(self) -> dict:
        mapping = {
            1: "open",
            2: "closed",
            3: "t1",
            4: "t2",
            5: "t3",
            6: "t4"
        }
        return mapping

    def _load(self):
        runs = self.runs.copy()

        subject_id = 0
        for subject in self.subjects:
            print(f"Loading data for subject [{subject_id + 1}/{self.subjects_count}] ..")
            subject_range = {"start": len(self.X)}
            raw_fnames = eegbci.load_data(subject, runs,
                                          path=self.root_dir,
                                          update_path=self.root_dir is not None,
                                          verbose="CRITICAL")
            raws = [read_raw_edf(f, preload=True, verbose="CRITICAL") for f in raw_fnames]
            montage = make_standard_montage("standard_1020")
            for task, raw in zip(runs, raws):
                eegbci.standardize(raw)
                raw.set_montage(montage)
                raw.filter(self.l_freq, self.h_freq, fir_design="firwin",
                           skip_by_annotation="edge", verbose="CRITICAL")
                raw.resample(sfreq=self.sampling_rate, verbose="ERROR")

                raw: mne.io.Raw
                data = raw.get_data(picks="eeg")
                if task == 1 or task == 2:
                    tirm_duration = self.baseline_duration
                else:
                    tirm_duration = self.task_duration
                data = data[:, :int(tirm_duration * self.sampling_rate)]
                self.raw_info = raw.info
                self.raw_info.set_montage('standard_1020')

                start_idx = 0
                end_idx = self.chunk_size
                subject_id_label = self.subject_id_2_label[subject_id]
                task_label = self.task_2_label[task]

                segment_start = len(self.X)

                while start_idx < data.shape[1]:
                    segment = self._get_segment(data, start_idx, end_idx)
                    if segment is None:
                        break

                    if task_label in self.labels:
                        self.Y.append(task_label)
                        self.X.append(segment)
                        self.subjects_ids.append(subject_id_label)

                    start_idx += self.chunk_size
                    end_idx += self.chunk_size
                segment_end = len(self.X)
                if subject_id_label not in self.task_ranges[task_label].keys():
                    self.task_ranges[task_label][subject_id_label] = []
                if segment_end - segment_start > 0:
                    self.task_ranges[task_label][subject_id_label].append(
                        np.arange(segment_start, segment_end).tolist())
            subject_id += 1
            subject_range["end"] = len(self.X)

            self.subjects_ranges.append(subject_range)
        assert len(self.X) == len(self.Y)
        if len(self.Y) == 0:
            raise ValueError("No data loaded. Check your configuration.")
        print(f"Done loading data for {self.subjects_count} subjects.")


__all__ = ["MMIDataset"]
