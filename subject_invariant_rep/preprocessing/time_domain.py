from sklearn.model_selection import StratifiedKFold

from ..utils.path import Path
from .raw import Raw
import multiprocessing as mp
from functools import partial
from typing import List
import numpy as np
from .utils import butter_bandpass_filter, export_data


def _all_data_have_same_shape(data: List[Raw]):
    first = data[0]
    for sample in data:
        if sample.X.shape != first.X.shape:
            print(f"X shape mismatch: {sample.file_path} {first.file_path}")
            return False
        if sample.y.shape != first.y.shape:
            print(f"y shape mismatch: {sample.file_path} {first.file_path}")
            return False
        if sample.subject_ids.shape != first.subject_ids.shape:
            print(f"subject_ids shape mismatch: {sample.file_path} {first.file_path}")
            return False
    return True


def _load_subject_data(subject, input_path, sfreq, window_len, start, end, training):
    raw = Raw(base_path=input_path,
              subject=subject,
              training=training, sfreq=sfreq,
              window_len=window_len)
    raw.load( start=start, stop=end)
    return raw


def _vstack_data(data: List[Raw], what: str):
    values = [getattr(sample, what) for sample in data]
    return np.stack(values, axis=0)


def _extract_values(data: List[Raw]):
    X = _vstack_data(data=data, what="X")
    y = _vstack_data(data=data, what="y")
    subject_ids = _vstack_data(data=data, what="subject_ids")
    return X, y, subject_ids


def cross_validation_split(subject, training_data, testing_data, sfreq, k_folds, subject_dependent: bool, output_path):
    print("Cross validation split for subject: ", subject + 1,
          " Subject dependent" if subject_dependent else "Subject independent")
    X_train = training_data[0]
    y_train = training_data[1]
    subject_ids_train = training_data[2]

    X_test = testing_data[0]
    y_test = testing_data[1]
    subject_ids_test = testing_data[2]

    if subject_dependent:
        X_train = X_train[subject]
        y_train = y_train[subject]
        subject_ids_train = subject_ids_train[subject]
    else:
        # remove current subject
        X_train = np.delete(X_train, subject, axis=0)
        y_train = np.delete(y_train, subject, axis=0)
        subject_ids_train = np.delete(subject_ids_train, subject, axis=0)
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        subject_ids_train = np.concatenate(subject_ids_train, axis=0)

    # keep only current subject
    X_test = X_test[subject]
    y_test = y_test[subject]
    subject_ids_test = subject_ids_test[subject]

    cv = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True)
    order = 5
    bands = [0.5, 30]
    for fold, (train_index, val_index) in enumerate(cv.split(X_train, y_train)):
        X_train_fold = X_train[train_index]
        y_train_fold = y_train[train_index]
        subject_ids_train_fold = subject_ids_train[train_index]

        X_val_fold = X_train[val_index]
        y_val_fold = y_train[val_index]
        subject_ids_val_fold = subject_ids_train[val_index]

        X_test_fold = X_test
        y_test_fold = y_test
        subject_ids_test_fold = subject_ids_test

        # X_train_fold = butter_bandpass_filter(data=X_train_fold, lowcut=bands[0], highcut=bands[1],
        #                                       fs=sfreq,
        #                                       order=order)
        # X_val_fold = butter_bandpass_filter(data=X_val_fold, lowcut=bands[0], highcut=bands[1],
        #                                     fs=sfreq,
        #                                     order=order)
        # X_test_fold = butter_bandpass_filter(data=X_test_fold, lowcut=bands[0], highcut=bands[1],
        #                                      fs=sfreq,
        #                                      order=order)

        SAVE_NAME = 'S{:03d}_fold{:03d}'.format(subject + 1, fold + 1)
        export_data(save_path=output_path,
                    NAME=SAVE_NAME,
                    X_train=X_train_fold,
                    y_train=y_train_fold,
                    subject_ids_train=subject_ids_train_fold,
                    X_val=X_val_fold,
                    y_val=y_val_fold,
                    subject_ids_val=subject_ids_val_fold,
                    X_test=X_test_fold,
                    y_test=y_test_fold,
                    subject_ids_test=subject_ids_test_fold)


class TimeDomain(object):
    def __init__(self, input_path, output_path, class_combo, sfreq,
                 window_len, n_subjects, k_folds,
                 start=None, end=None, cpu_count=None):
        if cpu_count is None:
            cpu_count = max(int(mp.cpu_count() // 4), 2)

        self.cpu_count = cpu_count
        self.n_subjects = n_subjects
        self.k_folds = k_folds
        self.start = start
        self.end = end
        self.input_path = Path(input_path, class_combo)
        self.output_path = output_path
        self.class_combo = class_combo
        self.sfreq = sfreq
        self.window_len = window_len
        self.subject_dependent_path = Path(output_path, "time_domain", class_combo, "subject_dependent")
        self.subject_independent_path = Path(output_path, "time_domain", class_combo, "subject_independent")
        self.subject_dependent_path.make(override=False, directory=True, ignore_errors=True)
        self.subject_independent_path.make(override=False, directory=True, ignore_errors=True)
        print("Input path: ", self.input_path)
        print("Output path: ", self.output_path)
        print("Subject dependent path: ", self.subject_dependent_path)
        print("Subject independent path: ", self.subject_independent_path)
        print("-" * 100)

    def _load_data(self, training: bool):
        raw_path = self.input_path.path

        pool = mp.Pool(self.cpu_count)
        callback = partial(_load_subject_data, training=training,
                           input_path=raw_path, sfreq=self.sfreq,
                          window_len=self.window_len,
                           start=self.start, end=self.end)
        raw_data = pool.map(callback, range(1, self.n_subjects + 1))
        pool.close()
        pool.join()
        return raw_data

    def _split_data(self, dependent: bool):
        output_path = self.subject_dependent_path if dependent else self.subject_independent_path
        output_path = output_path.path
        training_data = self._load_data(training=True)
        testing_data = self._load_data(training=False)
        assert _all_data_have_same_shape(training_data), "Training data shape mismatch"
        assert _all_data_have_same_shape(testing_data), "Testing data shape mismatch"

        training_X, training_y, training_subject_ids = _extract_values(training_data)
        testing_X, testing_y, testing_subject_ids = _extract_values(testing_data)
        train_data = [training_X, training_y, training_subject_ids]
        test_data = [testing_X, testing_y, testing_subject_ids]
        callback = partial(cross_validation_split, training_data=train_data, testing_data=test_data,
                           sfreq=self.sfreq, k_folds=self.k_folds,
                           subject_dependent=dependent, output_path=output_path)
        pool = mp.Pool(self.cpu_count)
        pool.map(callback, range(0, self.n_subjects))
        pool.close()
        pool.join()

    def subject_dependent(self):
        self._split_data(dependent=True)
        print("-" * 100)
        return self

    def subject_independent(self):
        self._split_data(dependent=False)
        print("-" * 100)
        return self


__all__ = ["TimeDomain"]
