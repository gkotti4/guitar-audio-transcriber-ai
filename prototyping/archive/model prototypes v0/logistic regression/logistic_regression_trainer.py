import os, time
import numpy as np
import librosa
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# CONFIG
TARGET_SR = 22050
DURATION = 0.5  # in seconds
N_MFCCS = 13
HOP_LENGTH = 512






def get_available_datasets(dataset_root="Datasets/"):
    if not os.path.exists(dataset_root):
        return [], []
    names = []
    paths = []
    try:
        for entry in os.listdir(dataset_root):
            path = os.path.join(dataset_root, entry)
            if os.path.isdir(path) and not entry.startswith('.'):
                names.append(entry)
                paths.append(path)
    except Exception as e:
        print(f"Error accessing dataset directory: {e} at {dataset_root}")
        return []
    return names, paths





class MFCCDatasetLoader():
    def __init__(self,
                 dataset_path,
                 target_sr=22050,
                 mono=True,
                 n_mfcc=13,
                 test_size=0.2,
                 duration=None,
                 seed=None):
        self.dataset_path   = dataset_path
        self.target_sr      = target_sr
        self.mono           = mono
        self.n_mfcc         = n_mfcc
        self.test_size      = test_size
        self.seed           = seed

        if duration is not None:
            self.fixed_len = int(self.target_sr * duration)
        else:
            self.fixed_len = None

        np.random.seed(seed)

    def fix_len(self, y: np.ndarray, fixed_len = None) -> np.ndarray:
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
        
    def encode_labels_to_ints(self, labels):                    # OPTIONAL: manual encoding vs sklearn's LabelEncoder
        # -: return list of mapped labels to ints
        classes = sorted(set(labels)) 							# unique label names
        label_to_idx = {c: i for i, c in enumerate(classes)}    # map to int
        
        encoded_labels = [label_to_idx[l] for l in labels]		# mapped list
        
        return encoded_labels, len(classes)						# (list of ints corresponding to label in labels), (num of classes)


    def load_paths_and_labels(self): # optional
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

        le = LabelEncoder()
        y_labels = le.fit_transform(labels)

        return paths, y_labels, le


    def _iter_audio(self):
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
                y_raw, sr = librosa.load(path, sr=self.target_sr, mono=self.mono)
                y_raw_fixed = self.fix_len(y_raw, self.fixed_len)
                yield y_raw_fixed, sr, label, path


    def load_audio_dataset(self):
        wavs, srs, labels = [], [], []
        for y_raw, sr, label, _ in self._iter_audio():
            wavs.append(y_raw)
            srs.append(sr)
            labels.append(label)
        return wavs, srs, labels
    
    def extract_dataset_mfcc_features(self, normalize_features=True):
        X = []
        wavs, srs, labels = self.load_audio_dataset()

        for wave in wavs:
            mfcc = librosa.feature.mfcc(y=wave, sr=self.target_sr, n_mfcc=self.n_mfcc)  # shape: (n_mfcc, n_frames)
            mfcc_vec = mfcc.mean(axis=1)                                                # shape: (n_mfcc,)
            if normalize_features:
                mfcc_vec = (mfcc_vec - mfcc_vec.mean()) / (mfcc_vec.std() + 1e-6)
            X.append(mfcc_vec)

        X = np.vstack(X)                        # shape: (n_samples, n_mfcc)
        y = np.array(labels, dtype=str)         # shape: (n_samples,)       - raw y (strings)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y) # converts labels (str's) to integer indices

        print(f"Extracted MFCC features for {len(X)} samples.")
        print("Classes:", list(label_encoder.classes_))

        return X, y_encoded, label_encoder

    
    def train_test_split(self):
        from sklearn.model_selection import train_test_split
        X, y_encoded, label_encoder = self.extract_dataset_mfcc_features()
        Xtr, Xte, ytr, yte = train_test_split(
            X, y_encoded,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=y_encoded
        )
        return Xtr, Xte, ytr, yte, label_encoder 













class LogisticRegressionTrainer():
    # TODO: Use Pipeline: StandardScaler -> LogisticRegression
    def __init__(self, loader, max_iter=1000, C=1.0, n_jobs=-1):
        self.scaler     = StandardScaler()
        self.model      = LogisticRegression(max_iter=max_iter, C=C, n_jobs=n_jobs)

        self.Loader = loader
        self.X_train, self.X_test, self.y_train, self.y_test, self.label_encoder = loader.train_test_split()

    def train(self):
        # Scale features
        X_train_scaled = self.scaler.fit_transform(self.X_train)

        # Train model
        self.model.fit(X_train_scaled, self.y_train)
        print("[train] Model training completed.")

    def evaluate(self):
        # Scale test features
        X_test_scaled = self.scaler.transform(self.X_test)

        # Accuracy score
        acc_score = self.model.score(X_test_scaled, self.y_test)

        # Evaluate
        print(f"[evaluate] Test Accuracy: {acc_score * 100:.2f}%")
        return acc_score
    
    def predict(self, X, y):
        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        y_pred = self.model.predict(X_scaled)

        # Accuracy score
        acc_score = accuracy_score(y, y_pred)
        print(f"[predict] Prediction Accuracy: {acc_score * 100:.2f}%")

        return acc_score, y_pred







# --- < MAIN > ---
dataset_names, dataset_paths = get_available_datasets()
print(f"Available Datasets: {dataset_names}\nDataset paths: {dataset_paths}", end="\n\n")

dataset_index = int(input("Enter dataset index to use (0, 1, 2, ...): "))
if(dataset_index < 0 or dataset_index >= len(dataset_paths)):
    print("Invalid dataset index. Exiting.")
    exit(1)

Loader = MFCCDatasetLoader(dataset_paths[dataset_index],
                             target_sr=TARGET_SR,
                             duration=DURATION,
                             n_mfcc=N_MFCCS)

Trainer = LogisticRegressionTrainer(Loader)
Trainer.train()
Trainer.evaluate()

















