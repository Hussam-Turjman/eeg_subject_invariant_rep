from __future__ import annotations

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib.figure import Figure

from typing import List


def create_pdf_from_figures(output_path, figures: List[Figure], plt_close_all=True) -> None:
    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.
    with PdfPages(output_path) as pdf:
        for fig in figures:
            pdf.savefig(fig)
            if plt_close_all:
                plt.close('all')


def plot_conf_mat(mat, class_names, title="Confusion matrix"):
    conf_plot = ConfusionMatrixDisplay(mat, display_labels=class_names)
    conf_plot = conf_plot.plot()
    conf_plot.ax_.set_title(title)
    return conf_plot


def plot_loss(train_loss, val_loss=None, test_loss=None, title="Loss", ylabel="Loss"):
    fig, ax = plt.subplots()
    epochs = np.arange(len(train_loss))
    ax.plot(epochs, train_loss, label="Train")
    if val_loss is not None:
        ax.plot(epochs, val_loss, label="Validation")

    if test_loss is not None:
        ax.plot(epochs, test_loss, label="Test")

    ax.set(xlabel='Epoch', ylabel=ylabel)
    ax.grid()
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if title is not None:
        fig.canvas.manager.set_window_title(title)
    return fig, ax


__all__ = ["plot_conf_mat", "plot_loss", "create_pdf_from_figures"]
