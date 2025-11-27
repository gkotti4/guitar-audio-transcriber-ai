import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder


AVAILABLE_DATASETS = [
    "Kaggle_Electric_Open_Notes",
    "Kaggle_Electric"
]

DATASETS_NUM_CLASSES = {
    AVAILABLE_DATASETS[0] : 6,
    AVAILABLE_DATASETS[1] : 37
}

def get_dataset_num_classes(dataset):
    return DATASETS_NUM_CLASSES[dataset]


def validate_dataset_name(name):
    if name not in AVAILABLE_DATASETS:
        return None
    return name

class AudioDatasetLoader:
    def __init__(self,
                 dataset_name="Kaggle_Electric_Open_Notes",
                 sr=44100,
                 mono=True,
                 n_mfcc=13,
                 test_size=0.2,
                 duration=None,
                 seed=None):
        self.dataset_name   = dataset_name
        self.sr             = sr
        self.mono           = mono
        self.n_mfcc         = n_mfcc
        self.test_size      = test_size
        self.seed           = seed

        if duration is not None:
            self.fixed_len = int(self.sr * duration)
        else:
            self.fixed_len = None

        self.script_dir   = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(self.script_dir, "Datasets", self.dataset_name)

        np.random.seed(seed)


    def _fix_length(self, y: np.ndarray) -> np.ndarray:
        if self.fixed_len is None:
            return y

        L = len(y)
        N = self.fixed_len

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

    def iter_audio(self):
        for folder in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder)
            if not os.path.isdir(folder_path):
                continue
            try:
                _, label = folder.split("-")
            except ValueError:
                continue
            for fname in os.listdir(folder_path):
                if not fname.endswith(".wav"):
                    continue
                path = os.path.join(folder_path, fname)
                y_raw, sr = librosa.load(path, sr=self.sr, mono=self.mono)
                y_raw_fixed = self._fix_length(y_raw)
                yield y_raw_fixed, sr, label, path


    def load_features(self):
        X, y = [], []
        for y_raw, sr, label, _ in self.iter_audio():
            mfcc = librosa.feature.mfcc(y=y_raw, sr=sr, n_mfcc=self.n_mfcc)
            # Optional: normalize the mfcc features
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
            X.append(mfcc.mean(axis=1))
            y.append(label)

        X = np.vstack(X)                  # shape: (n_samples, n_mfcc)
        y = np.array(y, dtype=str)        # shape: (n_samples,) 

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y) # converts labels (str's) to integer indices

        print(f"Extracted MFCC features for {len(X)} samples.")
        print("Classes:", list(label_encoder.classes_))

        return X, y_encoded, label_encoder


    def load_audio(self):
        audios, srs, labels = [], [], []
        for y_raw, sr, label, _ in self.iter_audio():
            audios.append(y_raw)
            srs.append(sr)
            labels.append(label)
        return audios, srs, labels

    
    def train_test_split(self):
        from sklearn.model_selection import train_test_split
        X, y, label_encoder = self.load_features()
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=y
        )

    def load_files_and_labels(self):
        """
        Walk through the dataset folders and return two lists:
          - file_paths: paths to all .wav files
          - labels:     corresponding integer labels
        """
        paths, labels = [], []
        for folder in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder)
            if not os.path.isdir(folder_path):
                continue
            try:
                _, label_str = folder.split("-")
            except ValueError:
                continue
            for fname in os.listdir(folder_path):
                if not fname.lower().endswith(".wav"):
                    continue
                paths.append(os.path.join(folder_path, fname))
                labels.append(label_str)

        # Encode string labels as integers
        le = LabelEncoder()
        y_labels = le.fit_transform(labels)

        return paths, y_labels, le




def clip_to_mfcc(clip, sr=44100, n_mfcc=13, hop_length=512):
    # y: 1-D audio, fixed length
    mfcc = librosa.feature.mfcc(
        y=clip,
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=hop_length
    )  # shape: (n_mfcc, n_frames)
    # Optionally normalize per‐clip:
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    return mfcc.astype(np.float32)      # return a float32 array









"""

    def load(self):
        X = []
        y = []

        # === Load + Process Audio ===
        print("Loading datasets from: " + self.dataset_path)

        for folder in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder)
            if not os.path.isdir(folder_path):
                continue

            try:
                # Extract note label from folder name
                _, note_label = folder.split("-")
            except ValueError:
                print(f"Skipping folder: {folder}")
                continue

            for filename in os.listdir(folder_path):
                if not filename.endswith(".wav"): # possible remove
                    continue

                file_path = os.path.join(folder_path, filename)

                try:
                    # Load audio (mono, no trimming)
                    y_raw, sr = librosa.load(file_path, sr=self.sr, mono=self.mono)

                    # Feature extraction (MFCC)
                    mfcc = librosa.feature.mfcc(y=y_raw, sr=sr, n_mfcc=self.n_mfcc)
                    mfcc_mean = np.mean(mfcc, axis=1)

                    X.append(mfcc_mean)
                    y.append(note_label)

                except Exception as e:
                    print(f"Error loading {file_path}: {e}")


        # === Encode Note Labels ===
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # === Convert to NP Arrays ===
        X = np.array(X)
        y_encoded = np.array(y_encoded)

        # === Summary ===
        print(f"Loaded {len(X)} samples.")
        print("Note Classes:", list(label_encoder.classes_))

        return X, y_encoded, label_encoder
    
    def train_test_split(self, test_size=0.2, shuffle=True):
        X, y, label_encoder = self.load()
        assert len(X) == len(y), "Features and labels must match in length"

        n_samples = len(X)
        indices = np.arange(n_samples)

        if shuffle:
            np.random.shuffle(indices)

        test_count = int(n_samples * test_size)

        test_idxs = indices[:test_count] # ex. [0,3,11,5,...]
        train_idxs = indices[test_count:] # ex. [6,2,7,10,...]

        return X[train_idxs], X[test_idxs], y[train_idxs], y[test_idxs]

"""