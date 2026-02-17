# data_analysis.py
import os, time
from pathlib import Path
from pprint import pprint
import sounddevice as sd

from collections import Counter
import librosa
import numpy as np
import pandas as pd
import scipy as sp
import soundfile as sf

from matplotlib import pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path.cwd().parent.parent # prototyping/
SOURCE_ROOT = PROJECT_ROOT / "source"
AUDIO_PROCESSING_ROOT = SOURCE_ROOT / "audio_processing"


import source.config as config
from source.audio.audio_preprocessing import AudioDatasetLoader, get_available_datasets


# - DATASET ANALYSIS
def dataset_analysis():
    # Choose Dataset
    print("\n\t| Choose Dataset |")
    print("Datasets root: ", config.DATASETS_ROOT, "\n")

    dataset_names, dataset_dirs = get_available_datasets(config.DATASETS_ROOT)
    num_datasets = len(dataset_dirs)

    print("Available datasets: ")
    print(dataset_names)

    dataset_index = 5 # int(input(f"Choose dataset (index: 0 — {num_datasets - 1}): "))


    # Load Dataset
    print("\n\t| Load Dataset |")
    start = time.time()
    loader = AudioDatasetLoader([dataset_dirs[dataset_index]], target_sr=config.TARGET_SR, duration=config.CLIP_DURATION)
    wavs, srs, labels, paths = loader.load_audio_dataset()
    print(f"load time: {time.time() - start}")


    # Dataset Info
    print("\n\t| Dataset Info |")

    num_samples = len(labels)
    print(f"num samples: {num_samples}")

    #unique, counts = np.unique(labels, return_counts=True)
    counts = Counter(labels)
    unique = sorted(counts.keys())
    values = [counts[l] for l in unique]
    num_labels = len(unique)
    print(f"labels: \n{unique}\nnum labels: {num_labels}")
    print("counts: ", counts)
    #pprint(counts)

    # plot bar chart for counts
    if False:
        print("\n\t| Label Distribution |")
        input("\nHit enter to continue...")
        plt.figure(figsize=(12,4))
        plt.bar(unique, values)
        plt.xticks(rotation=90)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Label Distribution")
        plt.tight_layout()
        plt.show()

    print("\n\t| Stats |")

    means = []
    stds = []
    variances = []
    mins = []
    maxs = []

    for w in wavs:
        means.append(np.mean(w))
        stds.append(np.std(w))
        variances.append(np.var(w))
        mins.append(np.min(w))
        maxs.append(np.max(w))

    avg_mean = np.mean(means)
    avg_std = np.mean(stds)
    avg_variance = np.mean(variances)
    avg_min = np.mean(mins)
    avg_max = np.mean(maxs)

    print(f"avg_mean: {avg_mean}, avg_std: {avg_std}, avg_variance: {avg_variance}, avg_min: {avg_min}, avg_max: {avg_max}")

    #input("\nHit enter to continue...")




# - SLICE ANALYSIS
def slice_analysis():
    print("\n\t| Inference Slices Analysis |")
    import source.audio.audio_slicer as audio_slicer
    import tempfile
    import tkinter as tk
    from tkinter import filedialog, messagebox


    with tempfile.TemporaryDirectory() as tmpdir:
        # Choose audio file to slice (inference clip)
        print(f"with temp directory: {tmpdir}")
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select guitar audio file",
            filetypes=(("WAV files", "*.wav"), ("All files", "*.*")),
        )
        if not file_path:
            messagebox.showerror("Error", "No file selected.")
            exit()
        audio_path = Path(file_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if audio_path.suffix.lower() != ".wav":
            raise ValueError(f"Input file must be a .wav file: {audio_path}")

        save_dir = Path(tmpdir) / "unknown"

        slicer = audio_slicer.AudioSlicer()

        slicer.sliceNsave(
                audio_path,
                save_dir,
                config.TARGET_SR,
                length_sec=config.CLIP_DURATION,
                hop_len=config.SLICER_CONFIG.HOP_LEN,
                min_sep=config.SLICER_CONFIG.MIN_SEP,
                min_db_threshold=config.SLICER_CONFIG.MIN_IN_DB_THRESHOLD,
                min_slice_rms_db=config.SLICER_CONFIG.MIN_SLICE_RMS_DB,
            )

        loader = AudioDatasetLoader([tmpdir], target_sr=config.TARGET_SR, duration=config.CLIP_DURATION)

        wavs, srs, labels, paths = loader.load_audio_dataset()

        print(paths)

        slice_count = len(wavs)
        slice_sr = srs[0]
        if np.unique(srs).size != 1:
            raise ValueError(f"Slices must have same sampling rate: {slice_sr}")

        print(f"slice_count: {slice_count}, sr: {slice_sr}")

        means = []
        stds = []
        variances = []
        mins = []
        maxs = []

        for w in wavs:
            means.append(np.mean(w))
            stds.append(np.std(w))
            variances.append(np.var(w))
            mins.append(np.min(w))
            maxs.append(np.max(w))

        avg_mean = np.mean(means)
        avg_std = np.mean(stds)
        avg_var = np.mean(variances)
        avg_min = np.min(mins)
        avg_max = np.max(maxs)

        print(f"avg_mean: {avg_mean}, avg_std: {avg_std}, avg_var: {avg_var}, avg_min: {avg_min}, avg_max: {avg_max}")

        print("\n Play each slice ... ")
        for w in wavs:
            input("Hit enter to play next slice ...")
            sd.play(w, samplerate=slice_sr)
            sd.wait()








