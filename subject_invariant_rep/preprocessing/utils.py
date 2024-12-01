import numpy as np

from ..min2net.utils import resampling, butter_bandpass_filter


def export_data(save_path, NAME, X_train, y_train, subject_ids_train,
                X_val, y_val, subject_ids_val, X_test, y_test, subject_ids_test):
    np.save(save_path + '/X_train_' + NAME + '.npy', X_train)
    np.save(save_path + '/X_val_' + NAME + '.npy', X_val)
    np.save(save_path + '/X_test_' + NAME + '.npy', X_test)
    np.save(save_path + '/y_train_' + NAME + '.npy', y_train)
    np.save(save_path + '/y_val_' + NAME + '.npy', y_val)
    np.save(save_path + '/y_test_' + NAME + '.npy', y_test)
    np.save(save_path + '/subject_ids_train_' + NAME + '.npy', subject_ids_train)
    np.save(save_path + '/subject_ids_val_' + NAME + '.npy', subject_ids_val)
    np.save(save_path + '/subject_ids_test_' + NAME + '.npy', subject_ids_test)


__all__ = ["resampling", "butter_bandpass_filter", "export_data"]
