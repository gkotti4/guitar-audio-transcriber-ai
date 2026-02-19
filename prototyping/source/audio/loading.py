# loading.py
import os
import librosa
import librosa.display
import librosa.feature
import numpy as np
from source.config import *

def get_available_datasets(datasets_root):
    datasets_root = Path(datasets_root)
    if not datasets_root.exists():
        print(f"[get_available_datasets] Dataset directory not found: {datasets_root}")
        return [], []

    all_names = []
    all_paths = []

    # iterate over first-level dirs (kaggle/, personal/, etc.)
    for subroot in sorted(datasets_root.iterdir()):
        if not subroot.is_dir() or subroot.name.startswith('.'):
            continue

        # iterate over datasets inside each subroot
        for ds in sorted(subroot.iterdir()):
            if ds.is_dir() and not ds.name.startswith('.'):
                label = f"{subroot.name}/{ds.name}"
                all_names.append(label)
                all_paths.append(ds)

    if not all_names:
        print(f"[get_available_datasets] No datasets found under {datasets_root}")

    return all_names, all_paths


class AudioDatasetLoader:
    def __init__(self,
                 dataset_roots: list[Path] | list[str],
                 target_sr: int = 11025,
                 mono: bool = True,
                 #test_size: float = 0.2,
                 duration: float | None = None
    ):
        self.dataset_roots = dataset_roots
        self.target_sr = target_sr
        self.mono = mono
        #self.test_size = test_size

        if duration is not None:
            self.fixed_len = int(self.target_sr * duration)
        else:
            self.fixed_len = None

    def fix_len(self, y: np.ndarray, fixed_len=None) -> np.ndarray:
        if fixed_len is None:
            return y

        L = len(y)
        N = fixed_len

        if L > N:
            # trim
            return y[:N]
        elif L < N:
            # pad with zeros
            pad_width = N - L
            return np.pad(y, (0, pad_width), mode="constant")
        else:
            # exactly the right length
            return y

    def _iter_audio(self):
        for i in range(len(self.dataset_roots)):
            for folder in os.listdir(self.dataset_roots[i]):
                folder_path = os.path.join(self.dataset_roots[i], folder)
                if not os.path.isdir(folder_path):
                    continue

                label = folder # Note: recent change

                for fname in os.listdir(folder_path):
                    if not fname.endswith(".wav"):
                        continue
                    path = os.path.join(folder_path, fname)
                    x_raw, sr = librosa.load(path, sr=self.target_sr, mono=self.mono)
                    x_raw_fixed = self.fix_len(x_raw, self.fixed_len)
                    yield x_raw_fixed, sr, label, path

    def load_audio_dataset(self, pad_to_max=True):
        wavs, srs, labels, paths = [], [], [], []

        for y_raw, sr, label, path in self._iter_audio():
            wavs.append(y_raw)
            srs.append(sr)
            labels.append(label)
            paths.append(path)

        if len(wavs) == 0:
            raise FileNotFoundError("load_audio_dataset: No audio files found.")
            #print("WARNING: load_audio_dataset: No audio files found.")
            #return [], [], [], []

        if pad_to_max:
            # wavs = [np.ndarray(), np.ndarray(), ...]
            max_len = max(len(w) for w in wavs)
            wavs = [np.pad(w, (0, max_len - len(w)), mode="constant") for w in wavs]

        return wavs, srs, labels, paths
