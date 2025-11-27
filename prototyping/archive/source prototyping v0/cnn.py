import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from dataset_loader import AudioDatasetLoader, clip_to_mfcc, AVAILABLE_DATASETS
from onset_segmenter import OnsetSegmenter
from display_utils import plot_data

import os, pickle
import librosa
import numpy as np


# -----------------------------
# Config
# -----------------------------
SR          = 44100
DURATION    = 1.0           # seconds per training clip
FIXED_LEN   = int(SR * DURATION)
N_MELS      = 128
HOP         = 512
N_FFT       = 2048
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 1e-3
WD          = 1e-3
SEED        = 42
DATASET     = AVAILABLE_DATASETS[0]  # e.g., "Kaggle_Electric_Open_Notes"


# -----------------------------
# Feature extraction
# -----------------------------
def wav_to_fixed(y, fixed_len=FIXED_LEN):
    """Pad/truncate waveform to fixed_len."""
    L = len(y)
    if L < fixed_len:
        y = np.pad(y, (0, fixed_len - L), mode="constant")
    else:
        y = y[:fixed_len]
    return y

def wav_to_logmel(y, sr=SR, n_mels=N_MELS, hop_length=HOP, n_fft=N_FFT):
    """Return log-mel spectrogram (F, T) as float32."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    # per-clip normalization (helps)
    m, s = S_db.mean(), S_db.std()
    S_db = (S_db - m) / (s + 1e-6)
    return S_db  # (F, T)

def build_arrays(paths, labels, sr=SR):
    """
    Load each file → fixed waveform → log-mel (F,T).
    Stack into X: (N,F,T). Return X (np.float32), y (np.int64), LabelEncoder.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels).astype(np.int64)

    feats = []
    for p in paths:
        y_wav, _ = librosa.load(p, sr=sr, mono=True, duration=DURATION)
        y_fix = wav_to_fixed(y_wav, FIXED_LEN)
        mel = wav_to_logmel(y_fix, sr=sr)  # (F, T)
        feats.append(mel)
    X = np.stack(feats, axis=0).astype(np.float32)  # (N, F, T)
    return X, y, le


# -----------------------------
# Simple Conv2D model
# -----------------------------
class TF_CNN(nn.Module):
    """Tiny time–freq CNN that expects (B,1,F,T)."""
    def __init__(self, n_freq, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool  = nn.AdaptiveAvgPool2d((1,1))
        self.fc    = nn.Linear(32, n_classes)

    def forward(self, x):
        # x: (B,1,F,T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)          # (F/2, T/2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)          # (F/4, T/4)
        x = self.pool(x).flatten(1)     # (B, 32)
        return self.fc(x)               # (B, C)


# -----------------------------
# Train / Eval
# -----------------------------
def train_mvp(dataset_name=DATASET, epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1) Get file paths + string labels
    loader = AudioDatasetLoader(dataset_name=dataset_name, sr=SR)
    paths, labels, _ = loader.load_files_and_labels()

    # 2) Split
    Xtr_p, Xte_p, ytr_s, yte_s = train_test_split(
        paths, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    # 3) Precompute features
    Xtr, ytr, le = build_arrays(Xtr_p, ytr_s, sr=SR)  # (Ntr,F,T), (Ntr,)
    Xte, yte, _  = build_arrays(Xte_p, yte_s, sr=SR)  # (Nte,F,T), (Nte,)

    # 4) Wrap into TensorDataset/DataLoader — no custom Dataset class
    # Add channel dim to (N,F,T) -> (N,1,F,T) inside the training loop (cheaper to do once here too):
    Xtr_t = torch.from_numpy(Xtr).unsqueeze(1)  # (N,1,F,T)
    Xte_t = torch.from_numpy(Xte).unsqueeze(1)  # (N,1,F,T)
    ytr_t = torch.from_numpy(ytr)               # (N,)
    yte_t = torch.from_numpy(yte)               # (N,)

    train_dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=batch_size)

    # 5) Model/optim
    n_classes = len(le.classes_)
    n_freq    = Xtr.shape[1]
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TF_CNN(n_freq=n_freq, n_classes=n_classes).to(device)
    opt   = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.CrossEntropyLoss()

    # 6) Train
    for ep in range(1, epochs+1):
        model.train()
        tl, corr = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = lossf(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl   += loss.item() * xb.size(0)
            corr += (logits.argmax(1) == yb).sum().item()
        train_loss = tl / len(train_dl.dataset)
        train_acc  = corr / len(train_dl.dataset)

        # val
        model.eval()
        vl, vc = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                lo = lossf(model(xb), yb)
                vl += lo.item() * xb.size(0)
                vc += (model(xb).argmax(1) == yb).sum().item()
        val_loss = vl / len(val_dl.dataset)
        val_acc  = vc / len(val_dl.dataset)

        if ep == 1 or ep % 2 == 0 or ep == epochs:
            print(f"Epoch {ep}/{epochs}  train_loss={train_loss:.3f} train_acc={train_acc:.2%}  "
                  f"val_loss={val_loss:.3f} val_acc={val_acc:.2%}")

    return model, le, device


# -----------------------------
# Inference on a song (segments → notes)
# -----------------------------
def predict_segments(model, le, device, wav_path, seg_duration=0.5):
    """
    Segment a song, turn each segment into log-mel (F,T) → (B,1,F,T), predict notes.
    """
    segmenter = OnsetSegmenter(duration=seg_duration)
    clips, times = segmenter.load_audio_segments(wav_path)  # List[np.ndarray] mono clips, each ~seg_duration

    # Ensure fixed length and compute log-mel for each clip
    feats = []
    for y in clips:
        y = wav_to_fixed(y, FIXED_LEN)      # we can reuse the same FIXED_LEN (or recompute for seg_duration)
        mel = wav_to_logmel(y, sr=SR)       # (F,T)
        feats.append(mel)
    X = np.stack(feats, axis=0).astype(np.float32)   # (B,F,T)
    xt = torch.from_numpy(X).unsqueeze(1).to(device) # (B,1,F,T)

    model.eval()
    with torch.no_grad():
        logits = model(xt)
        idxs = logits.argmax(1).cpu().numpy()
    notes = le.inverse_transform(idxs)
    return list(zip(times, notes))


if __name__ == "__main__":
    model, le, device = train_mvp(epochs=5)

    test_wav = "Samples/Gb_comp.wav"
    if os.path.exists(test_wav):
        preds = predict_segments(model, le, device, test_wav, seg_duration=0.5)
        # preds is List[ ((t0, t1), note), ... ]
        for (t0, t1), n in preds[:10]:
            mid = 0.5 * (t0 + t1)
            print(f"{mid:6.3f}s -> {n}")