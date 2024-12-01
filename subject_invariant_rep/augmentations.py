from .eeg_datasets.transformations import GaussianNoise, BandStopFilter, TemporalCutout, TemporalDelays, SensorDropout, \
    FakeTransformation
import numpy as np
from braindecode.augmentation import SensorsRotation
from .eeg_datasets.augmentation import EEGAugmentation


def create_augmentations(config: dict | None, sfreq, window_len, sensor_positions, augmentation_count=2,
                         return_original=True):
    known_transformations = {
        "GaussianNoise": GaussianNoise,
        "BandStopFilter": BandStopFilter,
        "TemporalCutout": TemporalCutout,
        "TemporalDelays": TemporalDelays,
        "SensorDropout": SensorDropout,

        "SensorsRotation": SensorsRotation,

    }
    transformations = [
        FakeTransformation()
    ]
    if config is not None:
        for key in config.keys():
            if key not in known_transformations.keys():
                raise Exception(f"Unknown transformation {key}")
            params = config[key]
            if key == "TemporalCutout":
                params["sfreq"] = sfreq
                params["cutout_max_len"] = int((sfreq * window_len) / 2)
            elif key == "TemporalDelays" or key == "BandStopFilter":
                params["sfreq"] = sfreq
            elif key == "SensorsRotation":
                positions = sensor_positions["ch_pos"]
                positions = np.vstack(list(positions.values()))
                positions = positions.T

                params["sensors_positions_matrix"] = positions
            # FIXME : add sensor positions for mmi
            if key != "SensorsRotation":
                transformation = known_transformations[key](**params)
                transformations.append(transformation)
    augmentation = EEGAugmentation(transformations=transformations,
                                   return_original=return_original,
                                   augmentation_count=augmentation_count,
                                   random_order=True)
    return augmentation


__all__ = ["create_augmentations"]
