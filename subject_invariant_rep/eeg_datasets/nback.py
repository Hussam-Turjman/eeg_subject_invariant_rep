from __future__ import annotations

import re
from pathlib import Path as PathLib
from typing import List, Literal

import mne
import numpy as np

from .dataset_interface import EEGDatasetInterface
from .eeg_dataframe import EEGDataframe
from ..constants import *
from ..utils.logger import logger
from ..utils.path import Path


def _get_df_timestamps(df):
    markers = np.array(df.markers).flatten()
    start_block_indices = df.marker_indices("start_block")
    task_indices = start_block_indices + 1
    end_block_indices = df.marker_indices("end_block")
    tasks = []
    pattern = re.compile(r"n=\d")
    for task_level in markers[task_indices]:
        if pattern.match(task_level) is not None:
            out = re.findall(r"\d", task_level)
            tasks.append(int(out[0]))
        else:
            raise ValueError("Task level not found in marker stream.")
    start_steps = df.relative_timestamps_markers[start_block_indices].tolist()
    end_steps = df.relative_timestamps_markers[end_block_indices].tolist()

    baseline_start = df.marker_relative_timestamps("start_baseline")
    baseline_end = df.marker_relative_timestamps("end_baseline")
    if len(baseline_start) == 1 and len(baseline_end) == 1:
        start_steps.append(baseline_start[0])
        end_steps.append(baseline_end[0])
        tasks.append(BASELINE_LABEL)
    return start_steps, end_steps, tasks


def _get_task(start, end, start_steps, end_steps, tasks):
    for start_step, end_step, task in zip(start_steps, end_steps, tasks):
        if start_step <= start <= end_step and start_step <= end <= end_step:
            return task
    return BREAK_LABEL


def collect_nback_subjects(root_dir: str, exclude_dirs=None,exclude_files=None,extension=".xdf"):
    out_files = []
    if exclude_dirs is None:
        exclude_dirs = []

    if exclude_files is None:
        exclude_files = []

    def walk_callback(root, subdirs, files):
        for file in files:
            if file.lower().endswith(extension):
                subdirs_check = [Path(str(x)).file_name() for x in PathLib(root).parents]
                subdirs_check = set(subdirs_check)
                ignore = len(subdirs_check.intersection(set(exclude_dirs)))
                f = Path(root, file)
                if file in exclude_files:
                    ignore += 1
                if ignore > 0:
                    print(f"Ignoring : {f}", flush=True)
                    continue

                assert f.exists(), f"File {f} does not exist."
                out_files.append(str(f))

    if not Path(root_dir).exists():
        raise ValueError(f"Path {root_dir} does not exist.")
    if not Path(root_dir).walk(walk_callback):
        raise ValueError(f"Path {root_dir} is empty or not a directory.")
    files = sorted(out_files)
    return files


class NBackDataset(EEGDatasetInterface):
    task_duration = 60  # seconds
    baseline_duration = 60  # seconds

    def __init__(self,
                 subjects: List[int | str] | None,
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
        if subjects is None and root_dir is None:
            raise ValueError("Either subjects or root_dir must be provided.")

        if root_dir is not None:
            logger.info(f"Collecting nback subjects from {root_dir}")
            subjects = collect_nback_subjects(root_dir, exclude_dirs=["incomplete"])
        super(NBackDataset, self).__init__(
            subjects=subjects,
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
            class_encoder_type=class_encoder_type,
            subject_id_encoder_type=subject_id_encoder_type,
            root_dir=root_dir,
            features_order=features_order
        )

    @property
    def channels_count(self) -> int:
        return 8

    @property
    def labels(self) -> List[str]:
        if self._labels is not None:
            return self._labels

        labels = ["n0", "n1", "n2", "n3", "break", "baseline"]

        if self._selected_classes is not None:
            self._selected_classes = list(set(self._selected_classes))
            assert set(self._selected_classes) <= {"n0", "n1", "n2", "n3", "break", "baseline"}
            labels = self._selected_classes
        self._labels = sorted(labels)
        return self._labels

    @property
    def name(self) -> str:
        return "NBackDataset"

    @property
    def task_ranges(self) -> dict:
        if self._task_ranges is not None:
            return self._task_ranges
        self._task_ranges = {
            "n0": {},
            "n1": {},
            "n2": {},
            "n3": {},
            "break": {},
            "baseline": {}
        }
        return self._task_ranges

    @property
    def task_2_label(self) -> dict:
        mapping = {
            0: "n0",
            1: "n1",
            2: "n2",
            3: "n3",
            BREAK_LABEL: "break",
            BASELINE_LABEL: "baseline"
        }
        return mapping

    def _split_segment(self, segment_data, subject_id, task):
        start_idx = 0
        end_idx = self.chunk_size
        subject_id_label = self.subject_id_2_label[subject_id]
        task_label = self.task_2_label[task]

        segment_start = len(self.X)
        while start_idx < segment_data.shape[1]:
            segment = self._get_segment(segment_data, start_idx, end_idx)
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
            self.task_ranges[task_label][subject_id_label].append(np.arange(segment_start, segment_end).tolist())

    @property
    def channels_names(self) -> List[str]:
        return HEADSET_CHANNELS

    def _load(self):
        subject_id = 0
        for subject_file in self.subjects:
            print(f"Loading subject [{subject_id + 1}/{self.subjects_count}]")
            subject_range = {"start": len(self.X)}
            df = EEGDataframe(filename=subject_file)
            eeg_raw = df.load()
            self._montage = eeg_raw.get_montage()
            start_steps, end_steps, tasks = _get_df_timestamps(df)

            eeg_raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, verbose="CRITICAL", picks="eeg")
            eeg_raw.resample(sfreq=self.sampling_rate, verbose="CRITICAL")
            data = eeg_raw.get_data()

            self.raw_info = eeg_raw.info
            self.raw_info.set_montage('standard_1020')

            break_indices = np.arange(0, data.shape[1])
            tasks_indices = []

            for start, end, task in zip(start_steps, end_steps, tasks):
                start = round(start, 2)
                end = round(end, 2)
                duration = round(end - start, 2)
                if task == BASELINE_LABEL:
                    tirm_duration = self.baseline_duration
                elif task != BREAK_LABEL:
                    tirm_duration = self.task_duration
                else:
                    tirm_duration = duration
                start_idx = int(start * self.sampling_rate)
                end_idx = int(end * self.sampling_rate)

                assert start_idx < data.shape[1]
                assert end_idx < data.shape[1]
                assert end_idx - start_idx > 0
                assert start_idx < end_idx
                segment_data = data[:, start_idx:end_idx]
                segment_data = segment_data[:, :int(tirm_duration * self.sampling_rate)]
                end_idx = start_idx + segment_data.shape[1]
                tasks_indices.extend(np.arange(start_idx, end_idx).tolist())

                # print(
                #     f"start  : {start}, end : {end}, task : {task}, duration : {duration},"
                #     f" start_idx : {start_idx}, end_idx : {end_idx}")

                self._split_segment(segment_data, subject_id, task)

            break_indices = np.delete(break_indices, tasks_indices)
            self._split_segment(data[:, break_indices], subject_id, BREAK_LABEL)

            subject_range["end"] = len(self.X)
            self.subjects_ranges.append(subject_range)
            subject_id += 1
        assert len(self.X) == len(self.Y) == len(self.subjects_ids)
        if len(self.Y) == 0:
            raise ValueError("No data loaded. Check your configuration.")


__all__ = ["NBackDataset", "collect_nback_subjects"]
