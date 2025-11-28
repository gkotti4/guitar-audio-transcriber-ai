import os, json
from dataclasses import dataclass
import librosa
import librosa.display
import librosa.feature
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torchaudio as ta
from torch.utils.data import Dataset, TensorDataset, DataLoader
from config import Config, MLPConfig, CNNConfig

_CONFIG = Config()


def get_available_datasets(datasets_root=_CONFIG.DATASETS_ROOT):
    if not os.path.exists(datasets_root):
        print("[get_available_datasets] Dataset directory not found.")
        return [], []
    names = []
    paths = []
    try:
        for entry in os.listdir(datasets_root):
            path = os.path.join(datasets_root, entry)
            if os.path.isdir(path) and not entry.startswith('.'):
                names.append(entry)
                paths.append(path)
    except Exception as e:
        print(f"Error accessing dataset directory: {e} at {datasets_root}")
        return []
    return names, paths

#@dataclass
#class FeatureBundle:
#    pass


class AudioDatasetLoader:
    def __init__(self,
                 dataset_root: str,
                 target_sr: int = 22050,
                 mono: bool = True,
                 test_size: float = 0.2,
                 duration: float | None = None):
        self.dataset_root = dataset_root
        self.target_sr = target_sr
        self.mono = mono
        self.test_size = test_size

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
        for folder in os.listdir(self.dataset_root):
            folder_path = os.path.join(self.dataset_root, folder)
            if not os.path.isdir(folder_path):
                continue

            label = folder # NOTE: recent change

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

        if pad_to_max:
            # wavs = [np.ndarray(), np.ndarray(), ...]
            max_len = max(len(w) for w in wavs)
            wavs = [np.pad(w, (0, max_len - len(w)), mode="constant") for w in wavs]

        return wavs, srs, labels, paths





class MelFeatureBuilder:
    """
    Unified feature builder for MFCC and Mel-spectrogram features.

    The class caches the *last extracted* features in:
        self.X, self.y_encoded, self.num_classes, self.reverse_map
    """
    def __init__(self,
                 seed: int = 42):
        # Later: self.config = config: FeatureConfig ?
        np.random.seed(seed)

    # ---------------------------------------------------------------------
    # Reports
    # ---------------------------------------------------------------------
    def _audio_report(self, audio_loader, sample_paths: bool = False, example_limit_per_class: int = 3):
        report = {}
        # Use pad_to_max=False to inspect real durations
        wavs, srs, labels, paths = audio_loader.load_audio_dataset(pad_to_max=False) # NOTE: hard reload dataset here

        if len(wavs) > 0:
            lengths = [len(w) / sr for w, sr in zip(wavs, srs)]

            report['target_sr']     = self.audio_loader.target_sr
            report['duration_min']  = float(np.min(lengths))
            report['duration_mean'] = float(np.mean(lengths))
            report['duration_max']  = float(np.max(lengths))
            report['unique_srs']    = sorted(list(set(srs)))
        else:
            report['target_sr']     = self.audio_loader.target_sr
            report['duration_min']  = None
            report['duration_mean'] = None
            report['duration_max']  = None
            report['unique_srs']    = []

        if sample_paths and self.y_encoded is not None and self.reverse_map is not None:
            report['example_paths'] = {}
            classes, _ = np.unique(self.y_encoded, return_counts=True)
            for c in classes:
                idxs = np.where(self.y_encoded == c)[0][:example_limit_per_class]
                report['example_paths'][self.reverse_map[int(c)]] = [paths[i] for i in idxs]

        print("--- Audio Data Report ---")
        print(json.dumps(report, indent=4, sort_keys=True))

        return report

    def _mfcc_report(self, X, y_encoded, reverse_map=None, scaler=None, out_root=None, out_filename=None, print_report: bool = True):
        """
        Feature report for the most recently extracted features (MFCC or Mel-spec).

        Name kept for backwards compatibility. It inspects self.X generically, so it
        works whether X is:
            - MFCC: np.ndarray, shape (N, n_mfcc)
            - Mel-spec: torch.Tensor, shape (N, 1, n_mels, T)
        """

        # Convert X to numpy array safely
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)

        report = {}
        n = X_np.shape[0]

        report['n_samples']    = int(n)
        report['feature_shape'] = list(X_np.shape[1:])              # remaining dims
        report['num_features'] = int(np.prod(X_np.shape[1:]))       # flattened size

        classes, counts = np.unique(y_encoded, return_counts=True)
        report['num_classes'] = int(len(classes))

        if reverse_map is not None:
            report['per_class_counts'] = {
                reverse_map[int(c)]: int(cnt) for c, cnt in zip(classes, counts)
            }

        # feature sanity
        report['X_nan_frac'] = float(np.isnan(X_np).mean())
        report['X_inf_frac'] = float(np.isinf(X_np).mean())
        report['X_min']      = float(np.nanmin(X_np))
        report['X_max']      = float(np.nanmax(X_np))
        report['X_mean']     = float(np.nanmean(X_np))
        report['X_std']      = float(np.nanstd(X_np))

        # scaler info if present (MFCC + MLP)
        if scaler is not None:
            report['scaler_mean']  = scaler.mean_.tolist()
            report['scaler_scale'] = scaler.scale_.tolist()

        # write to file
        if out_root is not None and out_filename is not None:
            os.makedirs(out_root, exist_ok=True)
            out_path = os.path.join(out_root, out_filename)
            with open(out_path, 'w') as f:
                json.dump(report, f, indent=2)

        if print_report:
            print("--- Feature Data Report (MFCC or Mel-spec) ---")
            print(json.dumps(report, indent=4, sort_keys=True))

        return report

    # ---------------------------------------------------------------------
    # Shared helpers
    # ---------------------------------------------------------------------
    def _encode_labels_to_ints(self, labels):
        classes = sorted(set(labels))  # unique label names
        label_to_idx = {c: i for i, c in enumerate(classes)}
        idx_to_label = {i: c for i, c in enumerate(classes)}
        encoded_labels = [label_to_idx[l] for l in labels]
        return encoded_labels, len(classes), idx_to_label

    def _create_tensor_dataset(self, X, y):
        # X can be np.ndarray or torch.Tensor
        if isinstance(X, torch.Tensor):
            X_tensor = X
        else:
            X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        ds = TensorDataset(X_tensor, y_tensor)
        return ds

    def _normalize_audio_volume(self, y, eps=1e-9):
        return y / (np.max(np.abs(y)) + eps)

    def extract_inference_features(self,
                                   audio_loader: AudioDatasetLoader,
                                   mlp_config: MLPConfig,
                                   cnn_config: CNNConfig,
                                   scaler: StandardScaler = None
    ):

        mfcc_features,_,_,_ = self.extract_mfcc_features(
            audio_loader,
            mlp_config.N_MFCC,
            mlp_config.NORMALIZE_FEATURES,
            mlp_config.NORMALIZE_AUDIO_VOLUME
        )
        if scaler: # or if mlp_config.STANDARD_SCALER
            mfcc_features = scaler.transform(mfcc_features)

        melspec_features,_,_,_ = self.extract_melspec_features(
            audio_loader,
            cnn_config.N_MELS,
            cnn_config.N_FFT,
            cnn_config.HOP_LENGTH,
            cnn_config.NORMALIZE_AUDIO_VOLUME
        )

        return mfcc_features, melspec_features

    # --- MFCC Functions ---
    def extract_mfcc_features(self,
                              audio_loader,
                              n_mfcc=13,
                              normalize_features: bool = False,
                              normalize_audio_volume: bool = False
    ):
        """
        Convert audio to MFCCs (mean pooled over time).

        Returns:
            X: np.ndarray, shape (N, n_mfcc)
            y_encoded: np.array, shape (N,)
            num_classes: int
            reverse_map: dict[int -> label]
        """
        X = []

        wavs, srs, labels, _ = audio_loader.load_audio_dataset(pad_to_max=True)

        for wave in wavs:
            y = wave
            if normalize_audio_volume:
                y = self._normalize_audio_volume(y)

            mfcc = librosa.feature.mfcc(
                y=y,
                sr=audio_loader.target_sr,
                n_mfcc=n_mfcc
            )  # shape: (n_mfcc, n_frames)

            mfcc_vec = mfcc.mean(axis=1)  # shape: (n_mfcc,)
            if normalize_features:
                mfcc_vec = (mfcc_vec - mfcc_vec.mean()) / (mfcc_vec.std() + 1e-6)
            X.append(mfcc_vec)

        X = np.vstack(X)                      # shape: (n_samples, n_mfcc)
        y = np.array(labels, dtype=str)       # raw label strings

        y_encoded, num_classes, reverse_map = self._encode_labels_to_ints(y)
        y_encoded = np.array(y_encoded, dtype=int)

        print(f"Extracted MFCC features for {len(X)} samples.")

        return X, y_encoded, num_classes, reverse_map

    def build_mfcc_dataloader(self, audio_loader, n_mfcc=13, batch_size: int = 32, shuffle: bool = True):
        """
        MFCC-only DataLoader (for MLP).
        """
        X, y_encoded, num_classes, reverse_map = self.extract_mfcc_features(audio_loader, n_mfcc)
        dataset = self._create_tensor_dataset(X, y_encoded)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader, num_classes, reverse_map

    def build_mfcc_train_val_dataloaders(
            self,
            audio_loader,               # used audio_loader instead of X, y; returns X, y with dataloaders for things like _mfcc_report
            n_mfcc=13,
            batch_size: int = 32,
            val_size: float = 0.2,
            shuffle_train: bool = True,
            shuffle_val: bool = False,
            normalize_audio_volume=False,
            normalize_features=False,
            standard_scaler: bool = True,
            seed: int = 42,
            num_workers: int = 0,
            pin_memory: bool = True,
            drop_last: bool = False
    ):
        """
        MFCC-only train/val DataLoaders (for MLP).
        """
        # 1) extract features
        X, y_encoded, num_classes, reverse_map = self.extract_mfcc_features(audio_loader, n_mfcc, normalize_features, normalize_audio_volume)

        # 2) stratified split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y_encoded,
            test_size=val_size,
            stratify=y_encoded,
            random_state=seed,
        )

        if standard_scaler:
            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_val = scaler.transform(X_val)
            self.scaler = scaler
        else:
            scaler = None

        # 3) datasets
        ds_tr  = self._create_tensor_dataset(X_tr, y_tr)
        ds_val = self._create_tensor_dataset(X_val, y_val)

        # 4) loaders
        dl_tr = DataLoader(
            ds_tr, batch_size=batch_size, shuffle=shuffle_train,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last
        )
        dl_val = DataLoader(
            ds_val, batch_size=batch_size, shuffle=shuffle_val,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False
        )

        return dl_tr, dl_val, X, y_encoded, num_classes, reverse_map, scaler

    # --- Mel Spectrogram Functions ---
    def extract_melspec_features(
        self,
        audio_loader,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 256,
        normalize_audio_volume: bool = False,
        to_db: bool = True,
    ):
        """
        Convert audio to Mel-spectrograms (2D time-frequency map).

        Returns:
            X: torch.Tensor, shape (N, 1, n_mels, T_max)
            y_encoded: np.ndarray, shape (N,)
            num_classes: int
            reverse_map: dict[int -> label]
        """

        target_sr = audio_loader.target_sr

        mel_transform = ta.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        to_db_transform = ta.transforms.AmplitudeToDB(stype="power")

        wavs, srs, labels, _ = audio_loader.load_audio_dataset(pad_to_max=True)

        specs = []
        for wave in wavs:
            y = wave.astype(np.float32)
            if normalize_audio_volume:
                y = self._normalize_audio_volume(y)

            y_t = torch.from_numpy(y).unsqueeze(0)   # (1, T)
            spec = mel_transform(y_t)                # (1, n_mels, T_spec)
            if to_db:
                spec = to_db_transform(spec)         # (1, n_mels, T_spec)

            specs.append(spec)

        # Pad/crop to max T so we can stack
        max_T = max(s.shape[-1] for s in specs)
        padded = []
        for s in specs:
            T = s.shape[-1]
            if T < max_T:
                s = torch.nn.functional.pad(s, (0, max_T - T))
            elif T > max_T:
                s = s[..., :max_T]
            padded.append(s)

        X = torch.stack(padded, dim=0)          # (N, 1, n_mels, T_max)
        y = np.array(labels, dtype=str)

        y_encoded, num_classes, reverse_map = self._encode_labels_to_ints(y)
        y_encoded = np.array(y_encoded, dtype=int)

        print(f"Extracted Mel-spectrogram features for {X.shape[0]} samples. X shape: {tuple(X.shape)}")

        return X, y_encoded, num_classes, reverse_map

    def build_melspec_dataloader(self,
                              audio_loader,
                              n_mels: int = 128,
                              n_fft: int = 1024,
                              hop_length: int = 256,
                              batch_size: int = 32,
                              shuffle: bool = True,
                              normalize_audio_volume: bool = False):
        """
        Mel-spectrogram DataLoader (for CNN).
        """

        X, y_encoded, num_classes, reverse_map = self.extract_melspec_features(
            audio_loader=audio_loader,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            normalize_audio_volume=normalize_audio_volume
        )

        dataset = self._create_tensor_dataset(X, y_encoded)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader, num_classes, reverse_map

    def build_melspec_train_val_dataloaders(
        self,
        audio_loader,               # audio_loader instead of X, y - both have their benefits
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 256,
        batch_size: int = 32,
        val_size: float = 0.2,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        normalize_audio_volume: bool = False,
        seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False
    ):
        """
        Mel-spectrogram train/val DataLoaders (for CNN).

        Note: no StandardScaler is applied here by default, because CNNs usually
        work directly on spectrogram values (optionally normalized/standardized
        inside the model or via custom transforms).
        """

        # 1) features + labels
        X, y_encoded, num_classes, reverse_map = self.extract_melspec_features(
            audio_loader=audio_loader,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            normalize_audio_volume=normalize_audio_volume
        )

        # 2) stratified split
        indices = np.arange(len(y_encoded))
        idx_tr, idx_val, y_tr, y_val = train_test_split(
            indices,
            y_encoded,
            test_size=val_size,
            stratify=y_encoded,
            random_state=seed,
        )

        X_tr  = X[idx_tr]
        X_val = X[idx_val]

        # 3) datasets
        ds_tr  = self._create_tensor_dataset(X_tr, y_tr)
        ds_val = self._create_tensor_dataset(X_val, y_val)

        # 4) loaders
        dl_tr = DataLoader(
            ds_tr,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=shuffle_val,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        return dl_tr, dl_val, X, y_encoded, num_classes, reverse_map

    def build_train_val_dataloaders(
            self,
            X, y_encoded,
            batch_size: int = 32,
            val_size: float = 0.2,
            shuffle_train: bool = True,
            shuffle_val: bool = False,
            standard_scaler: bool = True,
            seed: int = 42,
            num_workers: int = 0,
            pin_memory: bool = True,
            drop_last: bool = False
    ):
        """
        Generic train/val DataLoaders.
        """

        X = np.asarray(X)
        y_encoded = np.asarray(y_encoded, dtype=int)

        # 2) stratified split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y_encoded,
            test_size=val_size,
            stratify=y_encoded,
            random_state=seed,
        )

        if standard_scaler and X.ndim == 2:
            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_val = scaler.transform(X_val)
            self.scaler = scaler
        else:
            scaler = None

        # 3) datasets
        ds_tr = self._create_tensor_dataset(X_tr, y_tr)
        ds_val = self._create_tensor_dataset(X_val, y_val)

        # 4) loaders
        dl_tr = DataLoader(
            ds_tr, batch_size=batch_size, shuffle=shuffle_train,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last
        )
        dl_val = DataLoader(
            ds_val, batch_size=batch_size, shuffle=shuffle_val,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False
        )

        return dl_tr, dl_val, scaler
