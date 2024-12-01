import torch

from subject_invariant_rep.eeg_datasets.nback import NBackDataset
import matplotlib.pyplot as plt
import numpy as np
from mne.time_frequency.multitaper import psd_array_multitaper
#from subject_invariant_rep.visualization import plot_scaled_channels
from subject_invariant_rep.eeg_datasets.transformations.temporal_delays import TemporalDelays
# from framework.eeg_datasets.transformations.gaussian_noise import GaussianNoise
# from framework.eeg_datasets.transformations.bandstop_filter import BandStopFilter
# from framework.eeg_datasets.transformations.sensor_dropout import SensorDropout
# from framework.eeg_datasets.transformations.temporal_cutout import TemporalCutout
# from braindecode.augmentation import TimeReverse, SmoothTimeMask

FONTSIZE = 8
# A4 width: 6.3 inches 2*1.25 margins --> 5.8 figures
W = 12.5


def plot_compare_n_signals(*signals, sampling_rate, duration, xlabel, ylabel, title="title", colors=None):
    samples_limit = int(duration * sampling_rate)
    time = np.arange(0, duration * sampling_rate) / sampling_rate
    n = len(signals)
    if colors is not None:
        assert len(colors) == n
    else:
        colors = [None] * n
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(W, W / 3))
    axes = axes.flatten()
    axes[-1].set_xlabel(xlabel)
    for i, signal in enumerate(signals):
        if isinstance(signal, torch.Tensor):
            signal = signal.detach().cpu().numpy()

        signal = signal[:samples_limit]
        axes[i].plot(time, signal, c=colors[i])
        axes[i].set_ylabel(ylabel)
        axes[i].margins(x=0)
    fig.canvas.manager.set_window_title(title)
    return fig, axes, time


def plot_signal(window, ax, t_start=800, t_stop=1700, sfreq=100, **kwargs):
    time = np.arange(0, len(window)) / sfreq
    ax.plot(time[t_start:t_stop], window[t_start:t_stop], **kwargs)
    ax.set_ylabel('Voltage ($\\mu V$)', fontsize=FONTSIZE)
    ax.margins(x=0)
    return ax


def plot_psd(windows, ax, fmin=4, fmax=20, **kwargs):
    psds, freqs = psd_array_multitaper(
        windows, fmin=fmin, fmax=fmax, sfreq=100)
    psds = 10 * np.log10(psds)  # convert to dB
    psds_mean = psds.mean(0).mean(0)
    ax.plot(freqs, psds_mean, **kwargs)
    ax.set(xlabel='Frequency (Hz)', ylabel='PSD (dB)')
    return ax


def setup_style(grid=False, column_fig=False):
    plt.style.use('seaborn-v0_8-paper')

    if column_fig:
        plt.rcParams["figure.figsize"] = (W, W / 1.7)
    else:
        plt.rcParams["figure.figsize"] = (W, W / 3)
    plt.rcParams["axes.grid"] = grid
    fontsize = FONTSIZE * 2 if column_fig else FONTSIZE
    lw = 1.0 if column_fig else 0.5
    plt.rcParams.update({
        'font.size': fontsize,
        'legend.fontsize': 'x-small',
        'axes.labelsize': 'small',
        'xtick.labelsize': 'small',
        'ytick.labelsize': 'small',
        'axes.titlesize': 'medium',
        'lines.linewidth': lw,
    })


def load_data(sampling_rate=250.0, window_len=10.0,
              root_dir="experiment/eeg_data/eeg/P09",
              task="n3"):
    dataset = NBackDataset(root_dir=root_dir,
                           subjects=None,
                           subjects_limit=None,
                           sampling_rate=sampling_rate,
                           window_len=window_len,
                           selected_classes=None,
                           subject_id_encoder_type="onehot",
                           class_encoder_type="label",
                           scalings="mean",
                           transform=None,
                           l_freq=4.0,
                           h_freq=40.0,
                           grid=False,
                           merge_channels=False,
                           )
    X, y, subject_ids = dataset.get_subject_task_data(0, task)
    total_duration = X.shape[0] * X.shape[2] / sampling_rate
    return X, y, subject_ids, total_duration


def plot_temporal_delays(sampling_rate, original, fixed_duration=2.0, total_duration=5.0):
    td = TemporalDelays(sfreq=sampling_rate,
                        range_duration=2.0,
                        step=0.5,
                        fixed_duration=fixed_duration,
                        )
    td_x = td(original)
    x_aug = td_x.detach().numpy()
    X = np.hstack(original.cpu().numpy())
    x_aug = np.hstack(x_aug)
    fig, axes, time = plot_compare_n_signals(X[-1], x_aug[-1],
                                             sampling_rate=sampling_rate, duration=total_duration,
                                             xlabel="Time (s)",
                                             ylabel="Voltage ($\\mu V$)",
                                             title="Temporal Delay",
                                             colors=['gray', 'blue'],
                                             )

    axes[0].axvspan(0, fixed_duration, alpha=0.7, color='lightgray')
    axes[1].axvspan(fixed_duration, fixed_duration + fixed_duration, alpha=0.7, color='lightgray')
    fig.savefig("images/temporal_delays.png", dpi=300, bbox_inches='tight')


def run():
    # setup_style(column_fig=True)
    sampling_rate = 250.0
    X, y, subject_ids, total_duration = load_data(sampling_rate=sampling_rate)
    X = torch.tensor(X, dtype=torch.float32)
    plot_temporal_delays(sampling_rate=sampling_rate,
                         original=X.clone(),
                         fixed_duration=2.0,
                         total_duration=5.0)

    plt.show()


if __name__ == "__main__":
    run()
