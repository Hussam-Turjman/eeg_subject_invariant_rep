from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE


def plot_scaled_channels(data, sampling_rate,
                         channels_count=8,
                         start=0.0,
                         duration=1.0,
                         shift_val=3,
                         title=None,
                         show_average=True,
                         eeg_channels_names=None,
                         ax=None,
                         xlabel=None,
                         ylabel=None,
                         average_label_name="Average"):
    if eeg_channels_names is None:
        eeg_channels_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
    if show_average:
        eeg_channels_names += [average_label_name]

    shift = shift_val
    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    yticks = np.arange(1, len(eeg_channels_names) + 1) * shift_val
    ax.set_yticklabels(eeg_channels_names)
    ax.set_yticks(yticks)

    assert len(yticks) == len(eeg_channels_names)

    time_vector = np.arange(0, duration * sampling_rate) / sampling_rate
    time_vector += start

    start_idx = int(start * sampling_rate)
    end_idx = start_idx + int(sampling_rate * duration)
    print(start_idx, end_idx)
    for idx in range(channels_count):
        ax.plot(time_vector, data[idx, start_idx:end_idx] + shift)
        shift += shift_val
    averaged = None
    if show_average:
        averaged = data[:, start_idx:end_idx].mean(axis=0)
        ax.plot(time_vector, averaged + shift, color='black')

    if title is not None:
        if fig is not None:
            fig.canvas.manager.set_window_title(title)
        else:
            ax.set_title(title)
    # plt.show()
    return averaged, time_vector


def plot_spectrogram(data, sampling_rate, title=None):
    frequencies, times, spectrogram_data = spectrogram(
        data, fs=sampling_rate, nperseg=256, noverlap=128, nfft=256
    )
    num_channels = data.shape[0]
    for channel in range(num_channels):
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data[channel]), shading='auto')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (seconds)')
        # plt.title(f'Spectrogram - Channel {channel + 1}')
        # plt.colorbar(label='Power Spectral Density (dB)')
        # plt.axis('off')
        plt.ylim(0.5, 30)
        plt.tight_layout()
        plt.savefig('foo.png', bbox_inches='tight')

    # fig, axes = plt.subplots(num_channels // 2, 2)
    # axes = axes.flatten()
    #
    # for channel_idx in range(num_channels):
    #     axes[channel_idx].set_xlabel("Time (Seconds)")
    #     axes[channel_idx].set_ylabel("Frequency (Hz)")
    #     axes[channel_idx].set_yscale('symlog')
    #     axes[channel_idx].set_title(f'Channel [{channel_idx}]')
    #     axes[channel_idx].set_ylim(0.5, 30)
    #
    #     cb = axes[channel_idx].pcolormesh(times, frequencies, 10 * np.log10(spectrogram_data[channel_idx]), shading='auto')
    #     # fig.colorbar(cb,label='Power Spectral Density (dB)')

    if title is not None:
        fig.canvas.manager.set_window_title(title)


def plot_embeddings(method: str, data: np.ndarray, labels: np.ndarray, title=None,
                    save_path=None, perplexity=60,
                    grid=False, random_state=0,ax=None,marker_size=200):
    if method == "tsne":
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    elif method == "pca":
        tsne = PCA(n_components=2)
    elif method == "svd":
        tsne = TruncatedSVD(n_components=2)
    else:
        raise ValueError(f"Unsupported method {method}")

    embedded_data = tsne.fit_transform(data)
    unique_labels = np.unique(labels)
    # colors = cm.get_cmap("viridis")
    colors = ['purple', 'green', 'red', 'black', 'blue',
              'orange',
              'cyan', 'pink', 'lime', 'grey',
              'brown', 'magenta', 'darkred', 'teal', 'yellow']
    # colors = []
    if len(colors) < len(unique_labels):
        color_cm = plt.cm.viridis
        colors = []
        for i in range(len(unique_labels)):
            value = (i / len(unique_labels))
            colors.append(color_cm(value))

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    for idx, unique_label in enumerate(unique_labels):
        label_indices = np.where(labels == unique_label)[0]

        centers = np.mean(embedded_data[label_indices], axis=0)
        ax.scatter(centers[0], centers[1], color=colors[idx], marker='x', s=marker_size,
                   # label=unique_label
                   )
        ax.scatter(embedded_data[label_indices, 0], embedded_data[label_indices, 1],
                   # cmap=ListedColormap(colors),
                   color=colors[idx],
                   label=unique_label,
                   s=100,
                   )

    # scatter = ax.scatter(embedded_data[:, 0], embedded_data[:, 1], c=labels.flatten())
    # legend = ax.legend(*scatter.legend_elements(),
    #                    loc="lower left", title="Labels")

    # legend = ax.legend(loc="lower left", title="Labels")
    # ax.add_artist(legend)

    if grid:
        ax.grid(True)

    if title is not None:
        if fig is not None:
            fig.canvas.manager.set_window_title(title)
    else:
        title = "Embeddings"


__all__ = ["plot_embeddings",
           "plot_scaled_channels",
           "plot_spectrogram"]
