from __future__ import annotations
import pyxdf
import numpy as np
from ..utils.logger import logger
import mne


class EEGDataframe(object):
    original_raw: mne.io.RawArray | None
    #                    ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
    eeg_channels_names = ['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'Oz', 'PO8']

    def __init__(self, filename: str):
        self.relative_timestamps_markers = None
        self.absolute_timestamps_markers = None
        self.markers = None
        self.relative_timestamps = None
        self.absolute_timestamps = None
        self.sampling_rate = None
        self.marker_stream = None
        self.data_stream = None
        self.filename = filename
        self.streams = None
        self.header = None
        self.original_info = None
        self.original_raw = None

    def load(self) -> mne.io.RawArray:
        self.streams, self.header = pyxdf.load_xdf(self.filename)
        if len(self.streams) > 2:
            logger.warning("Warning: There are more than 2 streams in the xdf file.")
        for s in self.streams:
            if s['info']['type'][0] == 'Data' and s['info']['effective_srate'] != 0:
                self.data_stream = s
            if s['info']['type'][0] == 'Markers':
                self.marker_stream = s
        self.sampling_rate = float(self.data_stream["info"]["nominal_srate"][0])
        self.absolute_timestamps = self.data_stream['time_stamps']
        self.relative_timestamps = np.array(
            [i * (1 / self.sampling_rate) for i in range(len(self.absolute_timestamps))])
        self.markers = self.marker_stream['time_series']
        self.absolute_timestamps_markers = self.marker_stream['time_stamps']
        self.relative_timestamps_markers = self.get_relative_marker_timestamps()

        stream = self.data_stream
        data = stream["time_series"].T[:8]
        sfreq = self.sampling_rate

        info = mne.create_info(self.eeg_channels_names, sfreq, ["eeg"] * 8)
        info.set_montage('standard_1020')
        # scaler = mne.decoding.Scaler(info=info, scalings="median")
        # data = scaler.fit_transform(data.T).T[0]
        raw = mne.io.RawArray(data, info, verbose='error')
        self.original_info = info
        self.original_raw = raw
        return raw

    def create_raw_from_data(self, data: np.ndarray, sfreq=None) -> mne.io.RawArray:
        if sfreq is None:
            sfreq = self.sampling_rate

        info = mne.create_info(self.eeg_channels_names, sfreq, ["eeg"] * 8)
        info.set_montage('standard_1020')
        raw = mne.io.RawArray(data, info, verbose='error')
        return raw

    @property
    def eeg_raw(self) -> mne.io.RawArray:
        raw = self.original_raw.copy()
        raw: mne.io.RawArray
        raw.pick_types(eeg=True)
        return raw

    def marker_relative_timestamps(self, marker: str) -> np.ndarray:
        indices = self.marker_indices(marker)
        return np.copy(self.relative_timestamps_markers[indices])

    def marker_indices(self, marker: str) -> np.ndarray:
        markers = np.array(self.markers).flatten()
        indices = np.array(np.where(markers == marker)).flatten()
        return np.copy(indices)

    def crop(self, start: float, duration: float, eeg=True) -> mne.io.RawArray:
        raw = self.original_raw.copy()
        raw: mne.io.RawArray
        if eeg:
            raw.pick_types(eeg=True)
        return raw.crop(tmin=start, tmax=start + duration)

    def get_relative_marker_timestamps(self):
        relative_timestamps_markers = []
        for index, marker in enumerate(self.markers):
            cur_timestamp = self.absolute_timestamps_markers[index]
            for i, t in enumerate(self.absolute_timestamps):
                if t > cur_timestamp:
                    relative_timestamps_markers.append(self.relative_timestamps[i])
                    break
        return np.array(relative_timestamps_markers)


__all__ = ["EEGDataframe"]
