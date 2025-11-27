import librosa
import numpy as np
import matplotlib.pyplot as plt

# region Audio Display
def display_audio(y, sr=44100):
    plt.figure(figsize=(12,4))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def display_spectrogram(y, sr=44100, linear=True):
    """
    Plot either a linear-frequency STFT spectrogram or a mel-scaled spectrogram.

    Args:
        y      : 1D audio time series (numpy array)
        sr     : sample rate
        linear : if True, plot STFT; if False, plot mel spectrogram
    """
    if linear:
        # Compute STFT
        D = librosa.stft(y=y, n_fft=2048, hop_length=512)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Plot
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(
            D_db,
            sr=sr,
            hop_length=512,
            x_axis='time',
            y_axis='hz'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title("Linear-frequency Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.tight_layout()
        plt.show()

    else:
        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128,
            fmax=sr // 2
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(
            S_dB,
            sr=sr,
            hop_length=512,
            x_axis="time",
            y_axis="mel",
            fmax=sr // 2
        )
        plt.colorbar(label="dB")
        plt.title("Mel-frequency Spectrogram")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.tight_layout()
        plt.show()
# endregion

# region Plots
def plot_data(data, xlabel, ylabel, title=None, labels=None, xticks=None, figsize=(8,4), grid=True):
    # region Description
    # Plot one or more time-series.

    # Args:
    #     data: 
    #         - If a dict: keys are series names, values are lists/arrays of values.
    #         - If a list or 1D array: a single series is plotted.
    #         - If a 2D array or list of lists: each sub-list is a series.
    #     xlabel (str): label for the x-axis (e.g., "Epoch").
    #     ylabel (str): label for the y-axis (e.g., "Loss").
    #     title (str, optional): plot title.
    #     labels (list of str, optional): if `data` is list of series, the labels for each series.
    #     xticks (list, optional): custom tick positions for x-axis.
    #     figsize (tuple, optional): figure size, default (8,4).
    #     grid (bool, optional): whether to show grid, default True.

    # Usage examples:
    #     plot_over_time(losses, "Epoch", "Loss", title="Training Loss")
    #     plot_over_time({"train": loss_tr, "val": loss_val}, "Epoch", "Loss")
    #     plot_over_time([acc1, acc2], "Epoch", "Accuracy", labels=["run1","run2"])
    # endregion

    plt.figure(figsize=figsize)

    # Determine how many series and plot them
    if isinstance(data, dict):
        for name, series in data.items():
            plt.plot(series, label=name)
    else:
        # data is list/array of series or a single series
        # convert to list of series
        try:
            # see if it's 2D: list of lists or array
            iter(data[0])
            series_list = data
        except Exception:
            # 1D series
            series_list = [data]
        for idx, series in enumerate(series_list):
            label = None
            if labels is not None and idx < len(labels):
                label = labels[idx]
            plt.plot(series, label=label)

    # Labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)

    # Legend
    if (isinstance(data, dict) or labels) and plt.gca().get_legend_handles_labels()[1]:
        plt.legend()

    # X-ticks
    if xticks is not None:
        plt.xticks(xticks)

    if grid:
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()    

# endregion
