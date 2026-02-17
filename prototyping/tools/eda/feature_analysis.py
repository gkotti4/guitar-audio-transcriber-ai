# feature_analysis.py
import os, time
from pathlib import Path
from pprint import pprint

from collections import Counter
import librosa
import numpy as np
import pandas as pd
import scipy as sp

from matplotlib import pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path.cwd().parent.parent # prototyping/
SOURCE_ROOT = PROJECT_ROOT / "source"
AUDIO_PROCESSING_ROOT = SOURCE_ROOT / "audio_processing"


import source.config as config
from source.audio.audio_preprocessing import AudioDatasetLoader, get_available_datasets


# - FEATURE ANALYSIS
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


# Create Features

































