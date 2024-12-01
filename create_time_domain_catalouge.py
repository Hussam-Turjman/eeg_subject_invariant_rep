import json

import matplotlib.pyplot as plt
import mne
import numpy as np

from dataset_generator import normalize_all
from subject_invariant_rep.eeg_datasets.mmi import MMIDataset
from subject_invariant_rep.eeg_datasets.nback import NBackDataset, collect_nback_subjects
from subject_invariant_rep.utils.path import Path
from subject_invariant_rep.utils.plot import create_pdf_from_figures
from subject_invariant_rep.visualization import plot_scaled_channels


def create_fig(labels, subject_idx, trial_data, trial_num, info):
    fig, axes = plt.subplots(len(labels) // 2, 2)
    axes = axes.flatten()
    j = 0
    for task in labels:
        trial = trial_data[task][subject_idx]
        window_len = trial.shape[1] / info["sfreq"]
        window_len = int(window_len)
        plot_scaled_channels(data=trial,
                             eeg_channels_names=info["ch_names"],
                             sampling_rate=info["sfreq"],
                             ax=axes[j],
                             channels_count=info["nchan"],
                             start=0,
                             duration=5,
                             shift_val=100)

        # trial.plot(axes=axes[j], show=False)
        axes[j].set_title("S{:03d} ".format(subject_idx + 1) + f"{task} trial {trial_num}")
        j += 1
    fig.canvas.manager.set_window_title(f"Subject {subject_idx}")
    plt.tight_layout()
    return fig, axes


def create_mmi_catalogue_open_closed():
    config_path = Path("configs", "mmi", "raw.json")
    config = json.load(open(config_path.path, "r"))

    subjects = list(range(1, 15))
    class_combo = ["open", "closed"]
    dataset = MMIDataset(
        subjects=subjects,
        subjects_limit=None,
        selected_classes=class_combo,
        window_len=config["window_len"],
        sampling_rate=config["sampling_rate"],
        l_freq=config["l_freq"],
        h_freq=config["h_freq"],
        scalings=config["scalings"],
        transform=None,
        merge_channels=False,
        grid=False,
        class_encoder_type=config["class_encoder_type"],
        subject_id_encoder_type=config["subject_id_encoder_type"],
        root_dir="datasets",
        features_order="CT",
    )
    dataset.X, scaler = normalize_all(X=dataset.X, scalings=config["scalings"],
                                      norm=config["normalization"])
    subject_data_trial_0 = {class_label: [] for class_label in dataset.labels}

    for idx in range(dataset.subjects_count):
        for task in dataset.labels:
            X_0, y_0, subject_ids_0 = dataset.get_subject_task_data(subject_idx=idx, task=task, trials=[0])
            X_0 = np.hstack(X_0)
            subject_data_trial_0[task].append(X_0)

    info = mne.create_info(
        dataset.channels_names,
        dataset.sampling_rate,
        ["eeg"] * dataset.channels_count
    )
    info.set_montage('standard_1020')

    for task in dataset.labels:
        for idx in range(dataset.subjects_count):
            subject_data_trial_0[task][idx] = mne.io.RawArray(subject_data_trial_0[task][idx], info,
                                                              verbose='error')

    figs = []
    for idx in range(dataset.subjects_count):
        fig, axes = create_fig(labels=dataset.labels, subject_idx=idx,
                               trial_data=subject_data_trial_0, trial_num=0,
                               info=info)
        figs.append(fig)

    create_pdf_from_figures(output_path="images/mmi_open_closed_time_domain_catalogue.pdf",
                            figures=figs)


def create_mmi_catalogue():
    config_path = Path("configs", "mmi", "raw.json")
    config = json.load(open(config_path.path, "r"))

    subjects = list(range(1, 15))
    class_combo = ["t1", "t2", "t3", "t4"]
    dataset = MMIDataset(
        subjects=subjects,
        subjects_limit=None,
        selected_classes=class_combo,
        window_len=config["window_len"],
        sampling_rate=config["sampling_rate"],
        l_freq=config["l_freq"],
        h_freq=config["h_freq"],
        scalings=config["scalings"],
        transform=None,
        merge_channels=False,
        grid=False,
        class_encoder_type=config["class_encoder_type"],
        subject_id_encoder_type=config["subject_id_encoder_type"],
        root_dir="datasets",
        features_order="CT",
    )
    dataset.X, scaler = normalize_all(X=dataset.X, scalings=config["scalings"],
                                      norm=config["normalization"])
    subject_data_trial_0 = {class_label: [] for class_label in dataset.labels}

    for idx in range(dataset.subjects_count):
        for task in dataset.labels:
            X_0, y_0, subject_ids_0 = dataset.get_subject_task_data(subject_idx=idx, task=task, trials=[0])
            X_0 = np.hstack(X_0)
            subject_data_trial_0[task].append(X_0)

    info = mne.create_info(
        dataset.channels_names,
        dataset.sampling_rate,
        ["eeg"] * dataset.channels_count
    )
    info.set_montage('standard_1020')

    for task in dataset.labels:
        for idx in range(dataset.subjects_count):
            subject_data_trial_0[task][idx] = mne.io.RawArray(subject_data_trial_0[task][idx], info,
                                                              verbose='error')

    figs = []
    for idx in range(dataset.subjects_count):
        fig, axes = create_fig(labels=dataset.labels,
                               subject_idx=idx,
                               trial_data=subject_data_trial_0,
                               trial_num=0,
                               info=info)
        figs.append(fig)

    create_pdf_from_figures(output_path="images/mmi_time_domain_catalogue.pdf",
                            figures=figs)


def get_nback_subjects_trials(factor_value=1000, disable_norm=False, disable_factor=False):
    config_path = Path("configs", "nback", "raw.json")
    config = json.load(open(config_path.path, "r"))

    subjects = collect_nback_subjects(
        root_dir="experiment/eeg_data/eeg",
        exclude_dirs=["incomplete"],
        exclude_files=["sub-P000_ses-S002_task-Default_run-001_eeg.xdf",
                       "sub-P000_ses-S003_task-Default_run-001_eeg.xdf"]
    )
    class_combo = ["n0", "n1", "n2", "n3"]
    dataset = NBackDataset(
        subjects=subjects,
        subjects_limit=None,
        selected_classes=class_combo,
        window_len=config["window_len"],
        sampling_rate=config["sampling_rate"],
        l_freq=config["l_freq"],
        h_freq=config["h_freq"],
        scalings=config["scalings"],
        transform=None,
        merge_channels=False,
        grid=False,
        class_encoder_type=config["class_encoder_type"],
        subject_id_encoder_type=config["subject_id_encoder_type"],
        root_dir=None,
        features_order="CT",
    )
    dataset.X, scaler = normalize_all(X=dataset.X, scalings=config["scalings"],
                                      norm=config["normalization"], factor_value=factor_value,
                                      disable_norm=disable_norm, disable_factor=disable_factor)
    subject_data_trial_0 = {class_label: [] for class_label in dataset.labels}
    subject_data_trial_1 = {class_label: [] for class_label in dataset.labels}
    for idx in range(dataset.subjects_count):
        for task in dataset.labels:
            X_0, y_0, subject_ids_0 = dataset.get_subject_task_data(subject_idx=idx, task=task, trials=[0])
            X_1, y_1, subject_ids_1 = dataset.get_subject_task_data(subject_idx=idx, task=task, trials=[1])
            X_0 = np.hstack(X_0)
            X_1 = np.hstack(X_1)
            subject_data_trial_0[task].append(X_0)
            subject_data_trial_1[task].append(X_1)

    return dataset, config, subject_data_trial_0, subject_data_trial_1


def create_nback_catalogue():
    dataset, config, subject_data_trial_0, subject_data_trial_1 = get_nback_subjects_trials()
    info = mne.create_info(
        dataset.channels_names,
        dataset.sampling_rate,
        ["eeg"] * dataset.channels_count
    )
    info.set_montage('standard_1020')

    # for task in dataset.labels:
    #     for idx in range(dataset.subjects_count):
    #         subject_data_trial_0[task][idx] = mne.io.RawArray(subject_data_trial_0[task][idx], info,
    #                                                           verbose='error')
    #         subject_data_trial_1[task][idx] = mne.io.RawArray(subject_data_trial_1[task][idx], info,
    #                                                           verbose='error')

    figs = []
    for idx in range(dataset.subjects_count):
        fig, axes = create_fig(labels=dataset.labels, subject_idx=idx, trial_data=subject_data_trial_0, trial_num=0,
                               info=info)
        figs.append(fig)

    for idx in range(dataset.subjects_count):
        fig, axes = create_fig(labels=dataset.labels, subject_idx=idx, trial_data=subject_data_trial_1, trial_num=1,
                               info=info)
        figs.append(fig)

    create_pdf_from_figures(output_path="images/nback_time_domain_catalogue.pdf",
                            figures=figs)


def run():
    # create_mmi_catalogue_open_closed()
    # create_mmi_catalogue()
    create_nback_catalogue()


if __name__ == "__main__":
    run()
