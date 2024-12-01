import argparse
import copy
import json

import numpy as np

from subject_invariant_rep.eeg_datasets.utils import create_class_combinations
from subject_invariant_rep.constants import HEADSET_CHANNELS
from subject_invariant_rep.eeg_datasets.nback import collect_nback_subjects, NBackDataset
from subject_invariant_rep.preprocessing.time_domain import TimeDomain
from subject_invariant_rep.utils.path import Path
import multiprocessing as mp
from functools import partial
from subject_invariant_rep.eeg_datasets.mmi import MMIDataset
from sklearn.utils import shuffle
from sklearn import preprocessing as skpp
from sklearn.model_selection import train_test_split
from subject_invariant_rep.augmentations import create_augmentations


def get_transform(pick_smp_freq, window_len):
    model_config = json.load(open("configs/ssl/exp1.json", "r"))
    sensor_positions = json.load(open(Path("configs", "sensor_positions.json").path, "r"))
    return create_augmentations(model_config["transformations"],
                                sfreq=pick_smp_freq,
                                window_len=window_len,
                                sensor_positions=sensor_positions,
                                return_original=False,
                                augmentation_count=1)


def normalize_all(X, scalings: str, norm: str, factor_value=1000, disable_norm=False, disable_factor=False):
    if scalings == "mean":
        scaler = skpp.StandardScaler()
        print(f"Scaling data with: {scaler}", flush=True)
    else:
        raise ValueError(f"Invalid scaling type {scalings}")
    X = scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)

    if not disable_norm:
        X = skpp.normalize(X.reshape(X.shape[0], -1), norm=norm).reshape(X.shape)
        print(f"Normalizing data with norm: {norm}", flush=True)
    if not disable_factor:
        X = X * factor_value
        print(f"Scaling data with factor: {factor_value}", flush=True)
    return X, scaler


def _do_create_time_domain_data(class_combo, raw_config, time_domain_config, raw_data_path, output_path):
    window_len = raw_config["window_len"]
    sfreq = raw_config["sampling_rate"]
    start = time_domain_config["start"]
    end = time_domain_config["end"]
    k_folds = time_domain_config["k_folds"]
    n_subjects = raw_config["n_subjects"]
    generator = TimeDomain(
        input_path=raw_data_path,
        output_path=output_path,
        class_combo=class_combo,
        sfreq=sfreq,
        window_len=window_len,
        start=start,
        end=end,
        k_folds=k_folds,
        cpu_count=None,
        n_subjects=n_subjects
    )
    generator.subject_dependent().subject_independent()


def get_input_shape(path: Path, class_combo: str, subject_dependent: bool):
    subject_dependent = "subject_dependent" if subject_dependent else "subject_independent"
    fold_name = 'X_train_S{:03d}_fold{:03d}.npy'.format(1, 1)
    path = path.copy().join("time_domain", class_combo, subject_dependent, fold_name)
    data = np.load(path.path)
    C, T = list(data.shape[1:])
    return [1, T, C]


def create_time_domain_data(raw_config_path: Path, time_domain_config_path: Path, output_dir: Path):
    assert raw_config_path.exists(), f"Raw config file {raw_config_path} does not exist"
    assert time_domain_config_path.exists(), f"Time domain config file {time_domain_config_path} does not exist"
    raw_config = json.load(open(raw_config_path.path, "r"))
    time_domain_config = json.load(open(time_domain_config_path.path, "r"))

    tasks = raw_config["tasks"]
    time_domain_config["tasks"] = []
    for task in tasks:
        task_name = task["name"]
        _do_create_time_domain_data(class_combo=task_name,
                                    raw_config=raw_config,
                                    time_domain_config=time_domain_config,
                                    raw_data_path=output_dir.copy().join("raw").path,
                                    output_path=output_dir.copy().path)
        time_domain_config["tasks"].append({
            "name": task_name,
            "input_shape": get_input_shape(output_dir, task_name, subject_dependent=True),
            "num_class": len(task_name.split("_")),
        })

    p = output_dir.copy().join("time_domain.json")
    window_len = raw_config["window_len"]
    start = time_domain_config["start"]
    end = time_domain_config["end"]
    if start is not None and end is not None:
        window_len = end - start
    time_domain_config["window_len"] = window_len
    with open(p.path, "w+") as f:
        json.dump(time_domain_config, f, indent=4, sort_keys=True)


def get_trial_data(subject, dataset: NBackDataset, trial, transform=None):
    all_X = []
    all_y = []
    all_subject_ids = []
    do_augment = False
    for task in dataset.labels:
        if task == "baseline" and trial == 1:
            do_augment = True
            trial = 0
        X, y, subject_ids = dataset.get_subject_task_data(subject_idx=subject, task=task, trials=[trial])
        if do_augment:
            print("Augmenting baseline data as trial 1", flush=True)
            X, y, subject_ids = transform(tuple([X, y, subject_ids]))
            X = X[0]
        all_X.append(X)
        all_y.append(y)
        all_subject_ids.append(subject_ids)
    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    all_subject_ids = np.concatenate(all_subject_ids, axis=0)
    return all_X, all_y, all_subject_ids


def _do_create_raw_data(dataset, dataset_path: Path, dataset_name: str, random_state):
    for subject_id in range(dataset.subjects_count):
        if dataset_name == "nback":
            transform = get_transform(pick_smp_freq=dataset.sampling_rate, window_len=dataset.window_len,
                                      )
            X_train, y_train, subject_ids_train = get_trial_data(subject=subject_id, dataset=dataset, trial=0,
                                                                 transform=transform)
            X_val, y_val, subject_ids_val = get_trial_data(subject=subject_id, dataset=dataset, trial=1,
                                                           transform=transform)

            X_train, y_train, subject_ids_train = shuffle(X_train, y_train, subject_ids_train)
            X_val, y_val, subject_ids_val = shuffle(X_val, y_val, subject_ids_val)
        elif dataset_name == "mmi":
            X, y, subject_ids = dataset.get_subject_data(subject_idx=subject_id)
            X_train, X_val, y_train, y_val, subject_ids_train, subject_ids_val = train_test_split(X, y, subject_ids,
                                                                                                  test_size=0.5,
                                                                                                  random_state=random_state,
                                                                                                  stratify=y)
        else:
            raise Exception("Invalid dataset name")
        train_output_path = dataset_path.copy().join("S{:03d}T".format(subject_id + 1)).path
        test_output_path = dataset_path.copy().join("S{:03d}E".format(subject_id + 1)).path

        np.savez(train_output_path,
                 x=X_train,
                 y=y_train,
                 subject_ids=subject_ids_train)
        np.savez(test_output_path,
                 x=X_val,
                 y=y_val,
                 subject_ids=subject_ids_val)


def create_raw_data_from_class_combo(class_combo, dataset_name, dataset_root, config, subjects, output_dir: Path):
    if dataset_name == "nback":
        dataset = NBackDataset(
            root_dir=None,
            subjects=subjects,
            sampling_rate=config["sampling_rate"],
            window_len=config["window_len"],
            scalings=config["scalings"],
            transform=None,
            selected_classes=class_combo,
            h_freq=config["h_freq"],
            l_freq=config["l_freq"],
            class_encoder_type=config["class_encoder_type"],
            subject_id_encoder_type=config["subject_id_encoder_type"],
            merge_channels=False,
            grid=False,
            subjects_limit=None,
        )
    elif dataset_name == "mmi":
        dataset = MMIDataset(
            root_dir=dataset_root,
            subjects=subjects,
            sampling_rate=config["sampling_rate"],
            window_len=config["window_len"],
            scalings=config["scalings"],
            transform=None,
            selected_classes=class_combo,
            h_freq=config["h_freq"],
            l_freq=config["l_freq"],
            class_encoder_type=config["class_encoder_type"],
            subject_id_encoder_type=config["subject_id_encoder_type"],
            merge_channels=False,
            grid=False,
            subjects_limit=None,
        )
    else:
        raise Exception("Invalid dataset name")
    dataset.X, scaler = normalize_all(dataset.X, scalings=config["scalings"], norm=config["normalization"])
    class_combo_name = "_".join(class_combo)
    dataset_path = output_dir.copy().join("raw", class_combo_name)
    dataset_path.make(directory=True, override=True, ignore_errors=True)
    _do_create_raw_data(dataset=dataset,
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        random_state=config["random_state"])
    return class_combo_name


def create_raw_data(dataset_name: str, dataset_root: str, config_path, class_combinations, subjects, output_dir: Path,
                    cpu_count=None):
    if cpu_count is None:
        cpu_count = max(int(mp.cpu_count() // 4), 2)
    config = json.load(open(config_path.path, "r"))
    callback = partial(create_raw_data_from_class_combo,
                       config=config,
                       subjects=subjects,
                       output_dir=output_dir,
                       dataset_name=dataset_name,
                       dataset_root=dataset_root)

    pool = mp.Pool(cpu_count)
    class_combo_names = pool.map(callback, class_combinations)
    pool.close()
    pool.join()

    config["tasks"] = []
    for class_combo_name in class_combo_names:
        config["tasks"].append({
            "name": class_combo_name,
        })
    config_output_path = output_dir.copy().join("raw.json")
    config["n_subjects"] = len(subjects)
    with open(config_output_path.path, "w+") as f:
        json.dump(config, f, indent=4, sort_keys=True)
    return config


def should_update(config_path: Path, expected_config_path, keys_to_check):
    assert config_path.exists(), f"Raw config file {config_path} does not exist"
    if not expected_config_path.exists():
        return True
    raw_config = json.load(open(config_path.path, "r"))
    expected_raw_config = json.load(open(expected_config_path.path, "r"))

    for key in keys_to_check:
        if raw_config[key] != expected_raw_config[key]:
            return True
    return False


def create_nback_class_combinations(labels=None, only_n_classes=2, extra_classes=None):
    if labels is None:
        labels = ["n0", "n1", "n2", "n3"]
    if extra_classes is None:
        extra_classes = [["n0", "n1", "n2", "n3"], ["n0", "n1", "n2"]]
    class_combinations = create_class_combinations(labels=labels)
    if only_n_classes is not None:
        class_combinations = filter(lambda x: len(x) == only_n_classes, class_combinations)
        class_combinations = list(class_combinations)
    class_combinations += extra_classes
    return class_combinations


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,
                        help="Dataset name. i.e nback, mmi")
    args = parser.parse_args()
    dataset_name = args.dataset
    valid_dataset_names = ["nback", "mmi"]
    assert dataset_name in valid_dataset_names, f"Invalid dataset name {dataset_name}, must be one of {valid_dataset_names}"

    raw_config_path = Path("configs", dataset_name, "raw.json")
    time_domain_config_path = Path("configs", dataset_name, "time_domain.json")

    output_dir = Path( "datasets", dataset_name)

    expected_raw_config_path = output_dir.copy().join("raw.json")
    expected_time_domain_config_path = output_dir.copy().join("time_domain.json")

    # class_combinations = create_class_combinations(labels=["n0", "n1", "n2", "n3"])
    if dataset_name == "nback":
        dataset_root = None
        # n0_n3 n0_n1_n2 n1_n2_n3 n1_n2 n2_n3 n0_n1_n2_n3
        class_combinations = [
            ["n0", "n3"],
            ["n0", "n1", "n2"],
            ["n1", "n2", "n3"],
            ["n1", "n2"],
            ["n2", "n3"],
            ["n0", "n1", "n2", "n3"],
            ["baseline", "n1", "n2"],
            ["baseline", "n3"]
        ]
        with open(raw_config_path.path,"r") as f:
            raw_config = json.load(f)
        subjects = collect_nback_subjects(
            root_dir=raw_config["dataset_dir"],
            exclude_dirs=["incomplete"] + raw_config["exclude_subjects"],
            exclude_files=raw_config["exclude_sessions"]
        )
    elif dataset_name == "mmi":
        dataset_root = "datasets"
        class_combinations = [["open", "closed","t1",  "t3"]]
        subjects = list(range(1, 15))
    else:
        raise Exception("Invalid dataset name")
    x = ["_".join(x) for x in class_combinations]
    print(" ".join(x))

    if should_update(raw_config_path, expected_raw_config_path,
                     keys_to_check=["l_freq", "h_freq", "window_len",
                                    "scalings", "sampling_rate", "class_encoder_type",
                                    "subject_id_encoder_type", "random_state", "normalization"]):
        create_raw_data(dataset_name=dataset_name,
                        dataset_root=dataset_root,
                        config_path=raw_config_path,
                        class_combinations=class_combinations,
                        subjects=subjects,
                        output_dir=output_dir)

        create_time_domain_data(raw_config_path=expected_raw_config_path,
                                time_domain_config_path=time_domain_config_path,
                                output_dir=output_dir)

    else:
        print("Skipping raw data generation")

    if should_update(time_domain_config_path, expected_time_domain_config_path,
                     keys_to_check=["k_folds", "start", "end"]):
        create_time_domain_data(raw_config_path=expected_raw_config_path,
                                time_domain_config_path=time_domain_config_path,
                                output_dir=output_dir)

    else:
        print("Skipping time domain data generation")


if __name__ == "__main__":
    run()
