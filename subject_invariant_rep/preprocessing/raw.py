from ..utils.path import Path
import numpy as np
from .utils import resampling


class Raw(object):
    def __init__(self, base_path: str, subject: int, training: bool, sfreq: float, window_len: float):
        self.base_path = base_path
        self.subject = subject
        self.training = training
        self.sfreq = sfreq
        self.window_len = window_len
        subject_id = str(subject).zfill(3)
        file_type = "T" if training else "E"
        self.file_path = Path(base_path, f"S{subject_id}" + f"{file_type}.npz")
        assert self.file_path.exists(), f"File {self.file_path} does not exist"
        self._X = None
        self._y = None
        self._subject_ids = None
        self._read()

    @property
    def trials(self):
        return self._X.shape[0]

    @property
    def chunk_size(self):
        return self._X.shape[2]

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def subject_ids(self):
        return self._subject_ids

    def _read(self):
        data = np.load(self.file_path.path)
        self._X = data["x"]
        self._y = data["y"]
        self._subject_ids = data["subject_ids"]
        assert len(self.X) == len(self.y) == len(self.subject_ids), "Data length mismatch"
        return self

    def load(self, start=None, stop=None):
        if start is not None and stop is not None:
            start_time = int(start * self.sfreq)
            stop_time = int(stop * self.sfreq)
            self._X = self._X[:, :, start_time:stop_time]
        return self


__all__ = ["Raw"]
