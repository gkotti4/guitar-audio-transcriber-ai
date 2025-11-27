import os, time, json
from datetime import datetime
#from pathlib import Path
from pprint import pprint

import math
import numpy as np
import pandas as pd

import librosa, librosa.display
import torch
import torchaudio as ta
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
import matplotlib.pyplot as plt




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



class AudioDatasetLoader():
    def __init__(self,
                 dataset_root: str,
                 target_sr: int =22050,
                 mono: bool =True,
                 test_size: float =0.2,
                 duration: float|None =None):
        self.dataset_root   = dataset_root
        self.target_sr      = target_sr
        self.mono           = mono
        self.test_size      = test_size

        if duration is not None:
            self.fixed_len = int(self.target_sr * duration)
        else:
            self.fixed_len = None

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

    def _iter_audio(self):
        for folder in os.listdir(self.dataset_root):
            folder_path = os.path.join(self.dataset_root, folder)
            if not os.path.isdir(folder_path):
                continue
            try:
                #_, label = folder.split("-")
                label = folder                  # TODO: refactor function to only include this line - no .split("-")
            except ValueError:
                continue
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


class MFCCFeatureBuilder():
    def __init__(self,
                 audio_loader: AudioDatasetLoader,
                 n_mfcc =13,
                 test_size =0.2,
                 seed =42):
        self.audio_loader = audio_loader
        self.n_mfcc         = n_mfcc
        self.test_size      = test_size

        self.reverse_map    = None
        self.X              = None
        self.y_encoded      = None
        self.num_classes    = None
        self.scaler         = None

        np.random.seed(seed)


    def _audio_report(self, sample_paths=False, example_limit_per_class=3):
        report = {}
        if self.audio_loader is not None and self.y_encoded is not None:
            classes, counts = np.unique(self.y_encoded, return_counts=True)
            wavs, srs, labels, paths = self.audio_loader.load_audio_dataset()

            lengths = [len(w)/sr for w, sr in zip(wavs, srs)]

            report['target_sr']     = self.audio_loader.target_sr
            report['duration_min']  = float(np.min(lengths))
            report['duration_mean'] = float(np.mean(lengths))
            report['duration_max']  = float(np.max(lengths))
            report['unique_srs']    = sorted(list(set(srs)))

            if sample_paths and self.y_encoded and self.reverse_map:
                # sample few paths per class
                report['example_paths'] = {}
                for c in classes:
                    idxs = np.where(self.y_encoded == c)[0][:example_limit_per_class]
                    report['example_paths'][self.reverse_map[int(c)]] = [paths[i] for i in idxs]
        
        print("--- Audio Data Report ---")
        print(json.dumps(report, indent=4, sort_keys=True))
        return report

    def _mfcc_report(self, out_root=None, out_filename=None, print_report=True): # report of most recently loaded and extracted audio_loader and corresponding MFCC data
        
        if self.X is None or self.y_encoded is None:
            self.extract_mfcc_features()

        report = {}
        n = len(self.X)
        report['n_samples'] = int(n)
        report['num_features'] = int(self.X.shape[1])
        classes, counts = np.unique(self.y_encoded, return_counts=True)
        report['num_classes'] = int(len(classes))
        report['per_class_counts'] = {self.reverse_map[int(c)]: int(int(cnt)) for c, cnt in zip(classes, counts)}

        # feature sanity
        report['X_nan_frac'] = float(np.isnan(self.X).mean())
        report['X_inf_frac'] = float(np.isinf(self.X).mean())
        report['X_min'] = float(np.nanmin(self.X))
        report['X_max'] = float(np.nanmax(self.X))
        report['X_mean'] = float(np.nanmean(self.X))
        report['X_std'] = float(np.nanstd(self.X))

        # scaler info if present
        if hasattr(self, 'scaler') and isinstance(self.scaler, StandardScaler):
            report['scaler_mean'] = self.scaler.mean_.tolist()
            report['scaler_scale'] = self.scaler.scale_.tolist()

        # write to file
        if out_root is not None and out_filename is not None:
            os.makedirs(out_root, exist_ok=True)
            out_path = os.path.join(out_root, out_filename)
            with open(out_path, 'w') as f:
                json.dump(report, f, indent=2)

        if print_report:
            print("--- MFCC Feature Data Report ---")
            print(json.dumps(report, indent=4, sort_keys=True))
        return report


    def _encode_labels_to_ints(self, labels):                    # manual encoding vs sklearn's LabelEncoder (used for PyTorch dataset)
        # -: return list of mapped labels to ints
        classes = sorted(set(labels)) 							# unique label names
        label_to_idx = {c: i for i, c in enumerate(classes)}    # map to int

        idx_to_label = {i: c for i, c in enumerate(classes)}    # reverse map
        
        encoded_labels = [label_to_idx[l] for l in labels]		# mapped list
        
        return encoded_labels, len(classes), idx_to_label 

    def _create_tensor_dataset(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        ds = TensorDataset(X_tensor, y_tensor)
        return ds

    def _normalize_audio_volume(self, y, eps=1e-9):
        return y / (np.max(np.abs(y)) + eps)

    def extract_mfcc_features(self, audio_loader=None, normalize_features=False, normalize_audio_volume=False): # convert audio to MFCCs & most PREPROCESSING
        X = []

        if audio_loader is None and self.audio_loader is not None:
            audio_loader = self.audio_loader

        wavs, srs, labels, _ = audio_loader.load_audio_dataset(pad_to_max=True)

        # AUDIO PREPROCESSING, NOTE: Testing
        if normalize_audio_volume:
            wavs = self._normalize_audio_volume(np.array(wavs))

        for wave in wavs:
            mfcc = librosa.feature.mfcc(y=wave, sr=audio_loader.target_sr, n_mfcc=self.n_mfcc)  # shape: (n_mfcc, n_frames)
            mfcc_vec = mfcc.mean(axis=1)                                            # shape: (n_mfcc,)
            if normalize_features:
                mfcc_vec = (mfcc_vec - mfcc_vec.mean()) / (mfcc_vec.std() + 1e-6)   # z-score normalization
            X.append(mfcc_vec)

        X = np.vstack(X)                                                            # shape: (n_samples, n_mfcc)
        y = np.array(labels, dtype=str)                                             # shape: (n_samples,) - raw y (strings)

        y_encoded, num_classes, reverse_map = self._encode_labels_to_ints(y)         # convert labels to 'encoded' integers

        print(f"Extracted MFCC features for {len(X)} samples.")

        self.X, self.y_encoded, self.num_classes, self.reverse_map  = X, y_encoded, num_classes, reverse_map # note to self: caches from last extracted audio_loader
        
        return X, y_encoded, num_classes, reverse_map

    
    def build_dataloader(self, audio_loader=None, batch_size=32, shuffle=True):
        if audio_loader is None and self.audio_loader is not None:
            audio_loader = self.audio_loader
        X, y, num_classes, reverse_map = self.extract_mfcc_features(audio_loader)
        dataset = self._create_tensor_dataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader, num_classes, reverse_map
    
    def build_train_val_dataloaders(
        self,
        audio_loader=None,
        batch_size: int = 32,
        val_size: float = 0.2,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        normalize_features: bool = False,
        standard_scaler: bool = True,
        normalize_audio_volume: bool = False,
        seed: int = 42,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False
        ):

        if audio_loader is None and self.audio_loader is not None:
            audio_loader = self.audio_loader

        # 1) features + labels
        X, y, num_classes, reverse_map = self.extract_mfcc_features(audio_loader, normalize_features=normalize_features)

        # 2) stratified split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y,
            test_size=val_size,
            stratify=y,                             # keeps class balance
            random_state=seed,
        )

        if standard_scaler:                         # note to self: should we use standard scaler here? good pipeline?
            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_val = scaler.transform(X_val)
            self.scaler = scaler                    # should be saved if we want to reuse on REAL song data

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

        return dl_tr, dl_val, num_classes, reverse_map




class MLP(nn.Module):
    # num_features: number of input features
    # hidden_dim: number of neurons in the first hidden layer

    def __init__(self, num_features, hidden_dim, num_hidden_layers, num_classes, dropout = 0.0):
        super().__init__()

        layers = []

        # build the dims list
        dims: list[int] = [hidden_dim]
        for _ in range(num_hidden_layers - 1):
            next_dim = dims[-1] // 2
            if next_dim < 2:     # stop if it gets too small
                break
            dims.append(next_dim)

        # first hidden layer
        layers.append(nn.Linear(num_features, dims[0]))     # dense layer - fully connected layer (computes WX + b)
        layers.append(nn.LayerNorm(dims[0]))                # normalize each row across its out_dim features independently (row-wise normalization)
        #layers.append(nn.BatchNorm1d(dims[0]))             # normalize each of the out_dim columns across all in_dim rows (column-wise, batch-dependent)
        layers.append(nn.LeakyReLU(0.1))                    # activation: f(x) = x if x>0, else 0.01*x (keeps small gradient for negatives)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))              # randomly sets `dropout%` of activations to 0 during training, scales the rest


        # hidden layers
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.LeakyReLU(0.1))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # output layer
        layers.append(nn.Linear(dims[-1], num_classes)) # raw scores (logits)

        self.net = nn.Sequential(*layers)

        print("MLP model created: \n", self.net, end="\n\n")

    def create_model(self, num_features, hidden_dims: list, num_classes, dropout = 0.0): # since we can't overload __init__
        if not hidden_dims:
            raise ValueError("[__init__] hidden_dims does not contain any elements")

        layers = []

        # first layer
        layers.append(nn.Linear(num_features, hidden_dims[0]))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(dropout))

        # hidden layers
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))

        # output layer
        layers.append(nn.Linear(hidden_dims[-1], num_classes)) # raw scores (logits)

        self.net = nn.Sequential(*layers)

        print("MLP model created: \n", self.net, end="\n\n")
        
    def forward(self, x):
        return self.net(x)



class MLPTrainer():
    def __init__(
            self, 
            model: MLP, 
            train_dl,
            val_dl = None,
            reverse_map = None,
            device = None,
            lr=1e-3,
            weight_decay=1e-4):
        
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.model.apply(self._init_weights_kaiming) # new
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_dl = train_dl
        self.val_dl = val_dl

        self.reverse_map = reverse_map

        self.class_names = [str(reverse_map[k]) for k in sorted(reverse_map)] #[str(reverse_map[i]) for i in range(len(reverse_map))] # note: make sure reverse_map keys are valued less to greater (usually 0 to len-1)

        self._check_dims(self.train_dl) 
        if self.val_dl:
            self._check_dims(self.val_dl) 

        self.train_loss_history = []
        self.train_accuracy_history = []
        self.epoch = 0
        
    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _check_dims(self, dl, model = None):
        # checks if input x feature dimensions of a dataloader fit a models in_features
        if model is None:
            if self.model is None:
                raise ValueError("[_check_dims] Trainer has no model defined.")
            model = self.model

        if dl is None:
            print("[_check_dims] No DataLoader provided for dimension check.")
            return
        elif len(dl) == 0:
            raise ValueError("[_check_dims] Provided DataLoader is empty.")
        
        xb,_ = next(iter(dl))
        xb_num_features = xb.shape[1]
        if xb_num_features != self.model.net[0].in_features:
            raise ValueError(f"[_check_dims] Input feature dimension mismatch: DataLoader provides {xb_num_features}, but model expects {self.model.net[0].in_features}")
        
    def _softmax_accuracy(self, logits):
        # logits must be a torch.Tensor of shape (C,) or (B, C)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"_softmax_accuracy expects a torch.Tensor, got {type(logits)}")

        with torch.no_grad():
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)  # (1, C)

            probs = F.softmax(logits, dim=-1)           # (B, C)
            preds = torch.argmax(probs, dim=-1)         # (B,)
            confs = probs.gather(1, preds.unsqueeze(1)) # (B,1)
            confs = confs.squeeze(1)                    # (B,)

        # Print results for each item in the batch
        for i in range(len(preds)):
            idx = int(preds[i].item())
            conf = float(confs[i].item())

            if getattr(self, "reverse_map", None):
                label = self.reverse_map.get(idx, idx)
                print(f"{i}: Predicted note: {label}, Confidence: {conf:.2f}")
            else:
                print(f"{i}: Predicted class index: {idx}, Confidence: {conf:.2f}")

    def _2d_plot(self, x, y, title='Plot', labels=None, figsize=(8,4), grid=True):
        plt.figure(figsize=figsize)

        if not isinstance(y, (list, tuple)):
            y = [y]
        if not isinstance(x, (list, tuple)):
            x = [x] * len(y) # ?

        assert len(x) == len(y), "[_2d_plot] x and y must have same number of curves"
        
        # --- auto-generate labels if not provided
        if labels is None:                                  
            labels = [f"curve_{i}" for i in range(len(y))]
        if len(labels) != len(y):
            labels = [f"curve_{i}" for i in range(len(y))]
        

        for xi, yi, label in zip(x, y, labels):
            # possibly convert to np array
            plt.plot(xi, yi, label=label)

        plt.title(title)
        plt.legend()
        if grid:
            plt.grid(alpha=0.3)
        plt.show()

    def _confusion_matrix(self, y_true, preds_np, classes=None, normalize=False, figsize=(8,6), plot=False):
        y_true  = np.array(y_true).astype(int).ravel()  # ensure 1D np array
        preds_np = np.array(preds_np).astype(int).ravel()

        cm = metrics.confusion_matrix(y_true, preds_np)

        if normalize:
            with np.errstate(all="ignore"):
                cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
                cm = np.nan_to_num(cm)

        if not plot:
            print(cm)
            return cm  # return it too if you want to use later

        # ---- PLOT ----
        plt.figure(figsize=(6,5))
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()

        if classes is not None:
            plt.xticks(range(len(classes)), classes, rotation=45)
            plt.yticks(range(len(classes)), classes)

        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        # Add numbers to each cell
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = f"{cm[i,j]:.2f}" if normalize else str(int(cm[i,j]))
                plt.text(j, i, val, ha="center", va="center",
                        color="white" if cm[i,j] > thresh else "black")

        plt.tight_layout()
        plt.show()

        return cm

    def _classification_report(self, y_true, preds_np, target_names=None):
        y_true = np.array(y_true).astype(int).ravel()
        preds_np = np.array(preds_np).astype(int).ravel()
        report = metrics.classification_report(y_true, preds_np, target_names=target_names, digits=4)
        print(report)

    def _grad_norm_print(self, norm): # in testing
        if norm > 20:       return "██████  exploding"
        if norm > 1:        return "▅▅▅▅▁  high"
        if norm > 0.1:      return "▃▃▂▁▁  healthy"
        if norm > 0.001:    return "▁▁▁▁▁  low"
        return ".....  vanishing"

    def _log_grad(self, norm, history=None): # in testing
        # scale and clamp to 0–5
        level = int(min(5, max(0, math.log10(norm + 1e-6) + 3)))
        bar = "█" * level + " " * (5 - level)
        if history is not None:
            history.append(bar) # revist - delete if not used
        print(bar)

    def train(self, epochs=20, train_dl=None, es_window_len=4, es_slope_limit=1e-5, max_clip_norm=1.0, metrics=False):

        train_dl = train_dl or self.train_dl
        assert train_dl is not None, "[train] No training DataLoader provided."
        self._check_dims(train_dl)

        self.model.train()
        print("[train] Training start.")

        self.val_accuracy_history, self.val_loss_history = [], []
        for ep in range(1, epochs+1):                               # Epoch:
            print(f"[train] EPOCH {ep}/{epochs}")

            epoch_loss_sum, correct, total = 0.0, 0, 0
            for i, (xb, yb) in enumerate(train_dl):                 # Batch:

                # ---- train ----
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()                                  # reset gradients
                logits = self.model(xb)                                     # forward pass (compute logits)
                loss = self.loss_fn(logits, yb)                             # compute loss 
                loss.backward()                                             # backward pass (compute gradients)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_clip_norm) # clip gradients (protect from spikes and log to detect vanishing)
                self.optimizer.step()                                       # update weights (mini-batch gradient descent)

                # ---- metrics ----
                # weighted loss:
                epoch_loss_sum += loss.item() * yb.size(0)                  # sum of losses weighted by batch size
                
                # accuracy:
                preds = torch.argmax(logits, dim=1)                         # logits to class labels via argmax
                correct += (preds == yb).sum().item()
                total += yb.size(0)

                # gradient norms:
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                                param_norm = p.grad.data.norm(2)        # L2 norm of this tensor's grads
                                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5                          # sqrt of sum of squares
                #print(f"total grad norm: {total_norm:.4f}")
                #self._log_grad(total_norm)

                
            # ---- end of epoch metrics ----
            epoch_loss = epoch_loss_sum / total if total > 0 else 0.0
            epoch_acc  = correct / total        if total > 0 else 0.0
            self.train_loss_history.append(epoch_loss)
            self.train_accuracy_history.append(epoch_acc)
            self.epoch += 1

            # ---- early stop (es) val loss/acc ----
            val_acc, val_loss = self.evaluate()
            if ep > es_window_len + es_window_len/2:
                last_accs = self.val_accuracy_history[-es_window_len:]        if len(self.val_accuracy_history) >= es_window_len     else np.array([])
                last_losses = self.val_loss_history[-es_window_len:]    if len(self.val_loss_history) >= es_window_len   else np.array([])

                x_data = np.arange(len(last_losses))
                y_data = last_losses # es_window

                slope, _ = np.polyfit(x_data, y_data, 1) # -> slope, bias

                print(f"[train] early stop slope value: {slope:.4f}, over last {es_window_len} epochs")

                if slope >= es_slope_limit:
                    print("[train] early stop triggered: loss no longer decreasing")
                    break

                # TODO: save best model based on val loss/acc each epoch

            self.val_accuracy_history.append(val_acc)  
            self.val_loss_history.append(val_loss)  

            print(f"[train] train loss: {epoch_loss:.4f} | train accuracy: {epoch_acc:.4f} | val loss: {val_loss:.4f} | val accuracy: {val_acc:.4f}")
            print("\n...\n")

        # ---- overall training metrics ----
        if metrics:
            self._2d_plot(x=np.arange(len(self.train_accuracy_history)), y=[self.train_accuracy_history, self.train_loss_history], title="Training Curves", labels=["Accuracy", "Loss"])
            self._2d_plot(np.arange(len(self.val_accuracy_history)), [self.val_accuracy_history, self.val_loss_history], "Validation Curves", ["Accuracy", "Loss"])

        print("\n[train] Training complete.\n")

    
    def predict(self, xb):

        self.model.eval()
        with torch.no_grad():
            xb = xb.to(self.device)
            logits = self.model(xb)                      # logits
            preds = torch.argmax(logits, dim=1)          # logits to class labels via argmax
            return preds
        


    def evaluate(self, val_dl=None, cm=False, report=False, metrics=False):

        dl = val_dl or self.val_dl
        assert dl is not None, "[evaluate] No validation DataLoader provided."

        self.model.eval()
        correct, total, total_loss = 0, 0, 0.0
        preds_np = []
        y_true = []

        with torch.no_grad():
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits      = self.model(xb) 
                preds       = logits.argmax(dim=1)
                correct     += (preds == yb).sum().item()
                total       += yb.shape[0]
                total_loss  += self.loss_fn(logits, yb).item()

                preds_np.extend(preds.cpu().numpy())  
                y_true.extend(yb.cpu().numpy())      

                #rand_logit = logits[torch.randint(0, int(logits.shape[0]), (1,))]
                #self._softmax_accuracy(logits) # NOTE: in testing

        acc      = correct / total
        avg_loss = total_loss / total

        if metrics:
            self._confusion_matrix(y_true, preds_np, classes=self.class_names, plot=True, normalize=True)
            self._classification_report(y_true, preds_np, target_names=self.class_names)
        else:
            if cm:
                self._confusion_matrix(y_true, preds_np, classes=self.class_names)
            if report:
                self._classification_report(y_true, preds_np, target_names=self.class_names)

        #print(f"[evaluate] val accuracy: {acc:.4f}, val loss: {avg_loss:.4f}")
        return acc, avg_loss
    

    def save(self, filename="checkpoint.ckpt", root="checkpoints/"):
        
        os.makedirs(root, exist_ok=True)

        save_path = os.path.join(root, filename)

        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),            
            "device": str(self.device),
            "train_loss_history": self.train_loss_history,
            "train_accuracy_history": self.train_accuracy_history,
            "val_loss_history": self.val_loss_history,
            "val_accuracy_history": self.val_accuracy_history,
            "epoch": self.epoch,
        }
        
        torch.save(ckpt, save_path)

        print(f"[save] Checkpoint saved to {save_path}")

    def load(self, filename="checkpoint.ckpt", root="checkpoints/"):
        """NOTE: must be initialized with same train/val dl which was saved on"""
        if not os.path.isdir(root):
            raise FileNotFoundError(f"[load] No directory named: {root}")
        
        load_path = os.path.join(root, filename)

        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"[load] No file named: {load_path}")
        
        ckpt = torch.load(load_path, map_location="cpu")                                # map_location - safely remaps tensors to cpu (if trained on cuda but none available, will crash)

        self.model.load_state_dict(ckpt.get("model", {}))
        self.optimizer.load_state_dict(ckpt.get("optimizer", {}))

        self.device                     = torch.device(ckpt.get("device", str(self.device)))    # or torch.device("cpu")
        self.train_loss_history         = ckpt.get("train_loss_history", [])
        self.train_accuracy_history     = ckpt.get("train_accuracy_history", [])
        self.val_loss_history           = ckpt.get("val_loss_history", [])
        self.val_accuracy_history       = ckpt.get("val_accuracy_history", [])
        self.epoch                      = ckpt.get("epoch", 0)

        print(f"[load] Checkpoint loaded from {load_path}")
    




#                               ----- < MAIN > -----
def main():
    # --- config
    DATASET_ROOT = os.path.join("..", "..", "data", "online", "datasets", "kaggle")

    SAVE_CHECKPOINT = False
    LOAD_CHECKPOINT = False

    TARGET_SR               = 11025

    N_MFCC                  = 20
    BATCH_SIZE              = 32
    NORMALIZE_FEATURES      = False     # remember: don't use together with std scaler
    STANDARD_SCALER         = True      # remember
    NORMALIZE_AUDIO_VOLUME  = True      # new

    HIDDEN_DIM              = 128
    NUM_HIDDEN_LAYERS       = 2
    DROPOUT                 = 0.1

    EPOCHS                  = 50
    ES_WINDOW_LEN           = 5
    ES_SLOPE_LIMIT          = -0.0001


    # --- setup
    start_time = time.time()
    last_time = start_time
    # Get available datasets
    dataset_names, dataset_paths = get_available_datasets(dataset_root=DATASET_ROOT)
    print("Available datasets:", *dataset_names, sep="\n", end="\n\n")

    dataset_index = int(input(f"Enter dataset index (0 to {len(dataset_names)-1}): "))
    dataset_index = 0 # TODO: remove
    print()

    last_time = time.time()

    selected_dataset_path = dataset_paths[dataset_index]

    audio_dataset_loader = AudioDatasetLoader(selected_dataset_path, target_sr=TARGET_SR)


    # --- feature extraction
    mfcc_builder = MFCCFeatureBuilder(audio_dataset_loader, n_mfcc=N_MFCC)
    train_dl, val_dl, num_classes, reverse_map = mfcc_builder.build_train_val_dataloaders(
        batch_size=BATCH_SIZE,
        val_size=0.2,
        shuffle_train=True,
        shuffle_val=False,
        normalize_features=NORMALIZE_FEATURES,
        standard_scaler=STANDARD_SCALER,
        normalize_audio_volume=NORMALIZE_AUDIO_VOLUME,
        seed=42,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    # -- data reports
    #mfcc_builder._mfcc_report()
    #mfcc_builder._audio_report()

    print(f"audio loading & feature extraction time: {time.time() - last_time:.2f}s")
    print()

    last_time = time.time()

    xb,_ = next(iter(train_dl))
    num_features = xb.shape[1]
    print("num_features:", num_features)
    print("num_classes:", num_classes)
    print("class_labels:", [str(lbl) for lbl in reverse_map.values()])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print()

    # --- model and trainer setup
    model = MLP(num_features, hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS, num_classes=num_classes, dropout=DROPOUT)

    trainer = MLPTrainer(model, train_dl, val_dl, reverse_map, device=device)

    print(f"Model and Trainer setup time: {time.time() - last_time:.2f}s")
    print()
    last_time = time.time()


    # --- load
    if LOAD_CHECKPOINT:
        trainer.load()


    # --- training
    trainer.train(epochs=EPOCHS, es_window_len=ES_WINDOW_LEN, es_slope_limit=ES_SLOPE_LIMIT)


    # --- evaluation
    #trainer.evaluate(cm=True, report=True, metrics=True)


    # --- save
    if SAVE_CHECKPOINT:
        trainer.save()





if __name__ == "__main__":
    main()
    print("\n--- Script execution complete ---\n")








### HYPERPARAMETER TUNINGS
# @mfcc feature builder
#   n_mfcc:                 20 gave better results for kaggle_electric
#   normalize_features:     False - better for CNN - don't use with Std Scaler
#   standard_scaler:        True -
#
# @mlp
#   hidden_dim:             128 base results
#   num_hidden_layers:      2 base results
#   dropout:                0.0 base results - .1-.3 helps prevent overfitting, kept good accuracy on val for kaggle_electric
#
#
#
#
#
#
#
###






'''
Notes to self:
    standardize using size() vs shape[]
    preds = tensor (torch/NN calculations), preds_np = np.array, list (metrics, non-torch related)

    

REPLACED CODE:

def preds_to_classes(self, preds):
    if self.reverse_map is None:
        raise ValueError("[_to_classes] No reverse map provided for label decoding.")
    return [self.reverse_map[p.item()] for p in preds]

def evaluate(self, val_dl=None):

        if val_dl is None:
            val_dl = self.val_dl

        all_preds, all_labels = [], []
        for xb, yb in val_dl:
            xb, yb = xb.to(self.device), yb.to(self.device)     # note: send xb to device twice (in predict) - ~OK but could be optimized
            try:
                preds = self.predict(xb)
            except Exception as e:
                print(f"Error during prediction: {e}")
                return None
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc = (all_preds == all_labels).float().mean().item()
        return acc

'''