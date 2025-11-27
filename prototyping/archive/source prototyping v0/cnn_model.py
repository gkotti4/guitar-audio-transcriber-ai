import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from dataset_loader import AudioDatasetLoader, clip_to_mfcc, AVAILABLE_DATASETS
from onset_segmenter import OnsetSegmenter
from display_utils import plot_data

import os, pickle
import librosa
import numpy as np





# ────────────────────────────────────────────────────────────────
# 1) Dataset for MFCC inputs
class MFCCNoteDataset(Dataset):
    def __init__(self, file_paths, labels,
                 sr=44100, duration=1.0,
                 n_mfcc=13, hop_length=512):
        self.paths      = file_paths
        self.labels     = labels
        self.sr         = sr
        self.duration   = duration
        self.fixed_len  = int(sr * duration)
        self.n_mfcc     = n_mfcc
        self.hop_length = hop_length

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        y, _ = librosa.load(path, sr=self.sr, mono=True,
                            duration=self.duration)
        # pad/truncate to fixed length
        L = len(y)
        if L < self.fixed_len:
            y = np.pad(y, (0, self.fixed_len - L), mode="constant")
        else:
            y = y[:self.fixed_len]
        # compute MFCC (n_mfcc x n_frames)
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length
        ).astype(np.float32)
        # normalize per-clip (optional)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
        # to tensor with channel dim: (1, n_mfcc, n_frames)
        x = torch.from_numpy(mfcc).unsqueeze(0)
        y_label = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y_label

# ────────────────────────────────────────────────────────────────
# 2) The CNN model
class MFCC_CNN(nn.Module):
    def __init__(self, n_mfcc, n_classes):
        super().__init__()
        # feature extractor
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        # global pooling + classifier
        self.pool  = nn.AdaptiveAvgPool2d((1,1))
        self.fc    = nn.Linear(32, n_classes)

    def forward(self, x):
        # x: (B,1,n_mfcc,n_frames)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)                  # half height & width
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)                  # half again
        x = self.pool(x).view(x.size(0), -1)    # (B,32)
        return self.fc(x)                       # (B,n_classes)

# ────────────────────────────────────────────────────────────────
# 3) Trainer class
class CNNTrainer:
    def __init__(self,
                 dataset_name=AVAILABLE_DATASETS[0],
                 dataset_loader_cls=AudioDatasetLoader, 
                 batch_size=32,
                 test_size=0.2,
                 seed=42,
                 sr=44100,
                 duration=1.0,
                 n_mfcc=13,
                 hop_length=512,
                 lr=1e-3,
                 weight_decay=1e-3):
        self.dataset_name  = dataset_name
        self.LoaderClass   = dataset_loader_cls
        self.batch_size    = batch_size
        self.test_size     = test_size
        self.seed          = seed
        self.sr            = sr
        self.duration      = duration
        self.n_mfcc        = n_mfcc
        self.hop_length    = hop_length
        self.lr            = lr
        self.weight_decay  = weight_decay

        # Will be set in train():
        self.label_encoder = None
        self.model         = None
        self.history       = {"train_loss":[], "train_acc":[],
                              "val_loss":[],   "val_acc":[]}

    def save_model(self, model_path: str, encoder_path: str=None):
        """
        Saves the trained model’s state_dict and the label encoder.
        If encoder_path is None, it’ll default to model_path + '.le.pkl'.
        """
        # 1) Ensure train() has been called
        if self.model is None or self.label_encoder is None:
            raise RuntimeError("Train the model before saving!")

        # 2) Save weights
        torch.save(self.model.state_dict(), model_path)
        print(f"Model weights saved to {model_path}")

        # 3) Save encoder
        enc_path = encoder_path or f"{os.path.splitext(model_path)[0]}.le.pkl"
        with open(enc_path, "wb") as f:
            pickle.dump(self.label_encoder, f)
        print(f"Label encoder saved to {enc_path}")

    @classmethod
    def load_model(cls,
                   model_path: str,
                   encoder_path: str=None,
                   **trainer_kwargs):
        """
        Reconstructs a CNNTrainer, reloads weights & encoder, and returns the model.
        trainer_kwargs are the same args you’d pass into the constructor 
        (dataset_name, sr, duration, n_mfcc, etc.).
        """
        # 1) Instantiate a trainer (no training run)
        trainer = cls(**trainer_kwargs)

        # 2) Load encoder
        enc_path = encoder_path or f"{os.path.splitext(model_path)[0]}.le.pkl"
        with open(enc_path, "rb") as f:
            trainer.label_encoder = pickle.load(f)
        print(f"Label encoder loaded from {enc_path}")

        # 3) Build model architecture
        n_classes = len(trainer.label_encoder.classes_)
        model = MFCC_CNN(n_mfcc=trainer.n_mfcc, n_classes=n_classes)
        model.to(trainer.device if hasattr(trainer, "device") else "cpu")

        # 4) Load weights
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        print(f"Model weights loaded from {model_path}")

        # 5) Attach to trainer for convenience
        trainer.model = model
        return trainer

    def train(self, epochs=50, device=None, log_every=5, plot=False):
        # 1) Load file paths + labels
        loader = self.LoaderClass(dataset_name=self.dataset_name,
                                  sr=self.sr)
        paths, labels, le = loader.load_files_and_labels()
        self.label_encoder = le
        n_classes = len(le.classes_)

        # 2) Split
        Xtr, Xte, ytr, yte = train_test_split(
            paths, labels,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=labels
        )

        # 3) Build DataLoaders
        train_ds = MFCCNoteDataset(Xtr, ytr,
                                   sr=self.sr,
                                   duration=self.duration,
                                   n_mfcc=self.n_mfcc,
                                   hop_length=self.hop_length)
        val_ds   = MFCCNoteDataset(Xte, yte,
                                   sr=self.sr,
                                   duration=self.duration,
                                   n_mfcc=self.n_mfcc,
                                   hop_length=self.hop_length)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size,
                              shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=self.batch_size)

        # 4) Instantiate model
        self.model = MFCC_CNN(n_mfcc=self.n_mfcc,
                              n_classes=n_classes)
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # 5) Optimizer & loss
        opt = optim.AdamW(self.model.parameters(),
                          lr=self.lr,
                          weight_decay=self.weight_decay)
        lossf = nn.CrossEntropyLoss()

        # 6) Training loop
        for ep in range(1, epochs+1):
            # — Train —
            self.model.train()
            total_loss, correct = 0.0, 0
            for Xb, yb in train_dl:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = self.model(Xb)
                loss   = lossf(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * Xb.size(0)
                correct    += (logits.argmax(1)==yb).sum().item()

            train_loss = total_loss / len(train_dl.dataset)
            train_acc  = correct    / len(train_dl.dataset)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            # — Validate —
            self.model.eval()
            val_loss, val_corr = 0.0, 0
            with torch.no_grad():
                for Xv, yv in val_dl:
                    Xv, yv = Xv.to(device), yv.to(device)
                    lv = lossf(self.model(Xv), yv)
                    val_loss += lv.item() * Xv.size(0)
                    val_corr += (self.model(Xv).argmax(1)==yv).sum().item()

            val_loss = val_loss / len(val_dl.dataset)
            val_acc  = val_corr / len(val_dl.dataset)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # — Log —
            if ep==1 or ep % log_every==0:
                print(f"Epoch {ep}/{epochs}"
                      f"  train_loss={train_loss:.3f} train_acc={train_acc:.2%}"
                      f"  val_loss={val_loss:.3f} val_acc={val_acc:.2%}"
                )

        # 7) Plot curves using plot_data
        if plot:
            plot_data(
                {
                    "train_loss": self.history["train_loss"],
                    "val_loss":   self.history["val_loss"]
                },
                xlabel="Epoch",
                ylabel="Loss",
                title="CNN Loss Over Time"
            )

            plot_data(
                {
                    "train_acc": self.history["train_acc"],
                    "val_acc":   self.history["val_acc"]
                },
                xlabel="Epoch",
                ylabel="Accuracy",
                title="CNN Accuracy Over Time"
            )        

        return self.model  # trained model ready for inference
    
    def forward_mfccs(self, mfccs, device=None):
        x = mfcc_segs_to_tensor(mfccs)
        # 3) run inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x.to(device))        # calls forward()
            idxs   = logits.argmax(1).cpu().numpy()
        notes = self.model.label_encoder.inverse_transform(idxs)
        return notes

        # mfccs shape (n_frames, n_mfcc) or (85, 13)
        # 1) Stack into shape (B, n_mfcc, n_frames)
        batch = np.stack(mfccs, axis=0)  # (B, n_mfcc, n_frames)

        # 2) To torch tensor and add channel dim → (B, 1, n_mfcc, n_frames)
        device = device or next(self.model.parameters()).device
        x = torch.from_numpy(batch).unsqueeze(1).float().to(device)

        # 3) Forward through your CNN
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)           # expects (B,1,H,W)
        idxs = logits.argmax(dim=1).cpu().numpy()

        # 4) Map back to note names
        return self.label_encoder.inverse_transform(idxs)



### TESTING TRAINER ###
def trainer_test(epochs=1):
    trainer = CNNTrainer(
        dataset_name="Kaggle_Electric_Open_Notes",
        dataset_loader_cls=AudioDatasetLoader,
        batch_size=32,
        test_size=0.1,
        seed=42,
        sr=44100,
        duration=1.0,
        n_mfcc=13,
        hop_length=512,
        lr=1e-3,
        weight_decay=1e-3
    )

    trainer.train(epochs=epochs, log_every=5) # = trained MFCC_CNN model and trained CNNTrainer
    return trainer


def mfcc_segs_to_tensor(mfcc_segments):
    """Stack a list of MFCC arrays into a (B,1,H,W) tensor for CNN inference."""  

    print(f"mfcc_segments shape: {mfcc_segments.shape}")

    mfccs = np.transpose(mfcc_segments)
    print(f"np.transpose(mfcc_segments) - mfccs shape: {mfccs.shape}")

    # 2) stack into a single ndarray: (B, H, W)  
    #arr = np.stack(mfccs, axis=0)
    #     print(f"np.stack(mfccs, axis=0) - arr shape: {arr.shape}")
 
    arr = np.expand_dims(mfccs, 0)
    print(f"np.expand_dims(mfccs, 0) - arr shape: {arr.shape}")

    arr2 = np.expand_dims(arr, 1) 
    print(f"np.expand_dims(arr, 1)  - arr2 shape: {arr2.shape}")

    return arr2 # Tensor (1, 1, H, W)

    # 3) add channel dim → (B, 1, H, W)  
    #tensor = torch.from_numpy(arr).unsqueeze(1)  
    #print(f"torch.from_numpy(arr).unsqueeze(1) - tensor shape: {tensor.shape}") 

    #tensor = torch.from_numpy(mfccs).float()
    #print(f"torch.from_numpy(mfccs).float() - tensor shape: {tensor.shape}") 
    #print(f"tensor shape: {tensor.shape}") 

    #return tensor  # dtype=torch.float32 by default if mfccs are float32


def forward_clips_test():
    model = trainer_test()

    #loader = AudioDatasetLoader(duration=DURATION)
    #_, _, le = loader.load_features() 
    segmenter = OnsetSegmenter(duration=0.5)
    mfccs, times = segmenter.load_mfcc_segments("Samples/Gb_comp.wav")
    
    return model.forward_mfccs(mfccs)


forward_clips_test()




