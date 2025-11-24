import os, time, json
from datetime import datetime
from pathlib import Path

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa, librosa.display
import torch
import torchaudio as ta
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics

from audio_preprocessing import *
from config import *

# Torch CNN expects (batch_size, channels, height, width)
#
# 	log-mel spectrogram tensor shape (1, 128, T) : dtype should be X.float()
# 		1 	- input channel (grayscale)
#		128 - mel-frequency bins -> n_mels -> height
#		T 	- time frames -> width (depends on clip length and hop_length)
#
#	encoded_labels shape (N,) : dtype should be y.long()
#		


from typing import List



def format_time(time):
    return "{:.5f}s".format(time)

def format_float(x, precision=2):
    return "{:.{}f}".format(x, precision)


class Preprocess:
    def encode_labels_to_ints(labels):
        # -: return list of mapped labels to ints
        classes = sorted(set(labels)) 							# unique label names
        label_to_idx = {c: i for i, c in enumerate(classes)}    # map to int
        
        encoded_labels = [label_to_idx[l] for l in labels]		# mapped list
        
        return encoded_labels, len(classes)						# (list of ints corresponding to label in labels), (num of classes)
             
    def load_audio_dataset(root):
        file_paths = []
        labels = []

        for root, dirs, files in os.walk(root):
            for fname in files:
                if fname.lower().endswith(".wav"):
                    file_path = os.path.join(root, fname)
                    
                    label_folder = os.path.basename(root)
                    label = label_folder.split("-")[-1]
                    
                    file_paths.append(file_path)
                    labels.append(label)
        
        return file_paths, labels

    def build_mel_specs(wav_paths, sr=11025):
        specs = []
        mel = ta.transforms.MelSpectrogram(
                    sample_rate=sr, n_fft=1024, hop_length=256, n_mels=128, power=2.0
                )
        to_db = ta.transforms.AmplitudeToDB(stype="power")

        
        for i in range(len(wav_paths)):
            
            y, sr = librosa.load(wav_paths[i], sr=sr, mono=True)	# resamples to target_sr
            y = torch.from_numpy(y).unsqueeze(0)						# (1, Time Frames (T))
            
            spec = mel(y)
            spec_db = to_db(spec)
                
            specs.append(spec_db)
        
        return specs
            
    def pad_or_crop_to_max(specs):
        # Find max time dimension
        max_T = max(s.shape[-1] for s in specs)
        out = []
        
        for s in specs:
            T = s.shape[-1]
            
            if T < max_T:
                s = F.pad(s, (0, max_T - T)) 	# pad last dimension
                
            elif T > max_T:
                s = s[..., :max_T]				# crop
                
            out.append(s)
            
        return out
        
    def build_tensor(X, y):
        return torch.stack(X), torch.tensor(y, dtype=torch.long) #-> (N, 1, n_mels, T) 4D tensor, (encoded labels - 1D tensor (N,))
def build_dataset(root):
    wavs, labels = Preprocess.load_audio_dataset(root)
    encoded_labels, num_classes = Preprocess.encode_labels_to_ints(labels)

    specs = Preprocess.build_mel_specs(wavs)
    specs = Preprocess.pad_or_crop_to_max(specs)

    X, y = Preprocess.build_tensor(specs, encoded_labels)
    ds = TensorDataset(X, y)
    
    return ds, num_classes 
    


class CNN(nn.Module):
    """
    num_classes: number of output classes.
    in_channels: input channels (1 for mono spectrogram, 2+ if stacked features).
    base_channels: number of channels in the first conv layer.
    num_blocks: number of conv blocks (each roughly doubles channels).
    hidden_dim: size of the FC hidden layer before logits; if None, skip this layer.
    dropout: dropout probability in the classifier.
    use_batchnorm: whether to use BatchNorm2d after conv layers.
    kernel_size: kernel size for all convs (int or tuple).
    use_maxpool: whether to downsample with MaxPool2d in each block.
    adaptive_pool: (H, W) of the final AdaptiveAvgPool2d to fix feature map size.
    """

    def __init__(
        self,
        num_classes,
        in_channels = 1,
        base_channels = 32,
        num_blocks = 3,
        hidden_dim = 256,
        dropout = 0.1,
        kernel_size = 3,
        use_batchnorm = True,
        use_maxpool = True,
        adaptive_pool = (4, 4),
    ):
        super().__init__()

        conv_layers = []
        channels = [in_channels]

        # Build channel progression: e.g. 1 → 32 → 64 → 128 for 3 blocks
        for b in range(num_blocks):
            in_ch = channels[-1]
            out_ch = base_channels * (2 ** b)
            channels.append(out_ch)

            conv_layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # "same-ish" padding
                )
            )
            if use_batchnorm:
                conv_layers.append(nn.BatchNorm2d(out_ch))
            conv_layers.append(nn.ReLU(inplace=True))

            if use_maxpool:
                conv_layers.append(nn.MaxPool2d(2))  # halve H,W

        # Optional adaptive pooling to a fixed spatial size (independent of input H,W)
        conv_layers.append(nn.AdaptiveAvgPool2d(adaptive_pool))

        self.features = nn.Sequential(*conv_layers)

        # Compute flattened feature size after conv + pool
        feat_h, feat_w = adaptive_pool
        feat_dim = channels[-1] * feat_h * feat_w

        classifier_layers = list()
        classifier_layers.append(nn.Flatten())

        if hidden_dim is not None and hidden_dim > 0:
            classifier_layers.extend(
                [
                    nn.Linear(feat_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            classifier_layers.append(nn.Linear(hidden_dim, num_classes))
        else:
            # Directly from flattened features to logits
            classifier_layers.append(nn.Linear(feat_dim, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

        # For pretty printing & consistency with your old version
        self.model = nn.Sequential(self.features, self.classifier)

        print("CNN model created: \n", self.model, end="\n\n")

    def forward(self, x):
        x = self.model(x)
        return x

class CNNTrainer():
    def __init__(
            self,
            model: CNN,
            train_dl,
            val_dl=None,
            reverse_map=None,
            device=None,
            lr=1e-3,
            weight_decay=1e-4
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.model.apply(self._init_weights_kaiming)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.amp_scaler = None

        self._check_dims(train_dl)
        if val_dl:
            self._check_dims(val_dl)

        self.train_dl 			= train_dl
        self.val_dl             = val_dl
        self.reverse_map        = reverse_map
        self.class_names        = [str(reverse_map[k]) for k in sorted(reverse_map)] if reverse_map else []

        self.train_loss_history     = []
        self.train_accuracy_history = []
        self.val_accuracy_history   = []
        self.val_loss_history       = []
        self.epoch                  = 0

    def _check_dims(self, dl, model=None):
        # checks if input x feature dimensions of a dataloader fit a models in_features
        model = self.model if not model and self.model else None

        if len(dl) == 0:
            raise ValueError("[_check_dims] Provided DataLoader is empty.")

        xb, _ = next(iter(dl))
        xb_num_features = xb.shape[1:]
        if xb_num_features != model.net[0].in_features:
            raise ValueError(
                f"[_check_dims] Input feature dimension mismatch: DataLoader provides {xb_num_features}, but model expects {self.model.net[0].in_features}")

    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _softmax_accuracy(self, logits):
        # logits must be a torch.Tensor of shape (C,) or (B, C)
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"_softmax_accuracy expects a torch.Tensor, got {type(logits)}")

        with torch.no_grad():
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)  # (1, C)

            probs = F.softmax(logits, dim=-1)  # (B, C)
            preds = torch.argmax(probs, dim=-1)  # (B,)
            confs = probs.gather(1, preds.unsqueeze(1))  # (B,1)
            confs = confs.squeeze(1)  # (B,)

        # Print results for each item in the batch
        for i in range(len(preds)):
            idx = int(preds[i].item())
            conf = float(confs[i].item())

            if getattr(self, "reverse_map", None):
                label = self.reverse_map.get(idx, idx)
                print(f"{i}: Predicted note: {label}, Confidence: {conf:.2f}")
            else:
                print(f"{i}: Predicted class index: {idx}, Confidence: {conf:.2f}")

    def _2d_plot(self, x, y, title='Plot', labels=None, figsize=(8, 4), grid=True):
        plt.figure(figsize=figsize)

        if not isinstance(y, (list, tuple)):
            y = [y]
        if not isinstance(x, (list, tuple)):
            x = [x] * len(y)  # ?

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

    def _confusion_matrix(self, y_true, preds_np, classes=None, normalize=False, figsize=(8, 6), plot=False):
        y_true = np.array(y_true).astype(int).ravel()  # ensure 1D np array
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
        plt.figure(figsize=figsize)
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
                val = f"{cm[i, j]:.2f}" if normalize else str(int(cm[i, j]))
                plt.text(j, i, val, ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.show()

        return cm

    def _classification_report(self, y_true, preds_np, target_names=None):
        y_true = np.array(y_true).astype(int).ravel()
        preds_np = np.array(preds_np).astype(int).ravel()
        report = metrics.classification_report(y_true, preds_np, target_names=target_names, digits=4)
        print(report)

    def _grad_norm_print(self, norm):  # in testing
        if norm > 20:       return "██████  exploding"
        if norm > 1:        return "▅▅▅▅▁  high"
        if norm > 0.1:      return "▃▃▂▁▁  healthy"
        if norm > 0.001:    return "▁▁▁▁▁  low"
        return ".....  vanishing"

    def _log_grad(self, norm, history=None):  # in testing
        # scale and clamp to 0–5
        level = int(min(5, max(0, math.log10(norm + 1e-6) + 3)))
        bar = "█" * level + " " * (5 - level)
        if history is not None:
            history.append(bar)  # revist - delete if not used
        print(bar)

    def train(self, epochs = 20, train_dl=None, use_amp = True, max_clip_norm = 1.0, es_window_len=4, es_slope_limit=1e-5, plot_metrics=False):

        train_dl = train_dl or self.train_dl
        if train_dl is None:
            print("[train] No train dataloader provided. Exiting [train].")
            return
        self._check_dims(train_dl)

        use_amp = use_amp and (self.device.type == "cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # optional: (re)initialize histories
        self.train_loss_history = []
        self.train_accuracy_history = []

        self.model.to(self.device)
        self.model.train()
        print("[train] CNN training start.")

        for ep in range(1, epochs + 1):

            epoch_loss_sum, correct, total = 0.0, 0, 0

            for xb, yb in train_dl:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                # ---- forward (with optional AMP) ----
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = self.model(xb)
                    loss = self.loss_fn(logits, yb)

                # ---- backward + step ----
                if use_amp:
                    scaler.scale(loss).backward()

                    if max_clip_norm is not None:
                        # need to unscale before clipping
                        scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_clip_norm)

                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if max_clip_norm is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_clip_norm)
                    self.optimizer.step()

                # ---- metrics ----
                batch_size = yb.size(0)
                epoch_loss_sum += loss.item() * batch_size

                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
                total += batch_size

            # ---- end-of-epoch metrics ----
            epoch_loss = epoch_loss_sum / total if total > 0 else 0.0
            epoch_acc = correct / total if total > 0 else 0.0
            self.train_loss_history.append(epoch_loss)
            self.train_accuracy_history.append(epoch_acc)
            self.epoch += 1

            # ---- early stop (es) val loss/acc ----
            val_acc, val_loss = self.evaluate()
            if ep > es_window_len + es_window_len/2:
                last_accs = self.val_accuracy_history[-es_window_len:]  if len(self.val_accuracy_history) >= es_window_len      else np.array([])
                last_losses = self.val_loss_history[-es_window_len:]    if len(self.val_loss_history) >= es_window_len          else np.array([])

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

            print(f"[train] EPOCH {ep} / {epochs}")
            print(
                  f"[train] train loss: {epoch_loss:.4f} | "
                  f"train accuracy: {epoch_acc:.4f}"
            )
            print("\n...\n")

        # ---- overall training metrics ----
        if plot_metrics:
            self._2d_plot(x=np.arange(len(self.train_accuracy_history)), y=[self.train_accuracy_history, self.train_loss_history], title="Training Curves", labels=["Accuracy", "Loss"])
            self._2d_plot(np.arange(len(self.val_accuracy_history)), [self.val_accuracy_history, self.val_loss_history], "Validation Curves", ["Accuracy", "Loss"])

        print("[train] CNN training complete.\n")

    def evaluate(self, val_dl=None, use_amp=True, cm=False, report=False, metrics=False):
        dl = val_dl or self.val_dl
        if dl is None:
            print(f"[evaluate] No val dataloader provided.")
            return None, None

        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        preds_np = []
        y_true = []

        print("Evaluate start.")
        eval_start_time = time.time()

        use_amp = use_amp and getattr(self.device, "type", str(self.device)) == "cuda"

        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                # Forward pass (with optional AMP)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = self.model(xb)  # raw, unscaled model output values
                    loss = self.loss_fn(logits, yb)

                # accumulate weighted loss + accuracy
                batch_size = xb.size(0)
                loss_sum += loss.item() * batch_size  # weighted by batch size
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += batch_size

                # store for metrics / confusion matrix
                preds_np.extend(preds.cpu().numpy())
                y_true.extend(yb.cpu().numpy())

        acc = correct / total if total > 0 else 0.0
        avg_loss = loss_sum / total if total > 0 else 0.0

        if metrics:
            # full metrics mode: normalized cm + classification report
            self._confusion_matrix(y_true, preds_np, classes=self.class_names, plot=True, normalize=True)
            self._classification_report(y_true, preds_np, target_names=self.class_names)
        else:
            if cm:
                self._confusion_matrix(y_true, preds_np, classes=self.class_names)
            if report:
                self._classification_report(y_true, preds_np, target_names=self.class_names)

        print("Eval time: ", format_time(time.time() - eval_start_time))
        # print(f"[evaluate] val accuracy: {acc:.4f}, val loss: {avg_loss:.4f}")
        return acc, avg_loss
    
    def save(self, path, epoch=None, other=None):
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)		# create dir on disk if not already
        
        ckpt = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "epoch": int(epoch if epoch is not None else -1),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss_fn_type": type(self.loss_fn).__name__,
            "accs": getattr(self, "accs", []),
            "losses": getattr(self, "losses", []),
            "device_used": str(getattr(self, "device", "cpu")),
            "other": other or {},
        }
        
        # optional:
        if hasattr(self, "amp_scaler") and self.amp_scaler is not None:
            try:
                ckpt["amp_scaler"] = self.amp_scaler.state_dict()
            except Exception:
                pass
                            
        
        torch.save(ckpt, path)
        print(f"[save] Saved checkpoint: {path}.")
        
    def load(self, path, load_optimizer=True, load_amp_scaler=True):
        
        map_location = self.device if hasattr(self, "device") else "cpu"	# prevent crashes if loaded onto different torch device
        ckpt = torch.load(path, map_location=map_location)
        
        self.model.load_state_dict(ckpt["model"], strict=True)
        
        if load_optimizer and "optimizer" in ckpt and hasattr(self, "opt"):
            self.optimizer.load_state_dict(ckpt["optimizer"])
            
        # optional:
        if load_amp_scaler and "amp_scaler" in ckpt and hasattr(self, "amp_scaler"):
            self.amp_scaler.load_state_dict(ckpt["amp_scaler"])
            
        self.accs 		= ckpt.get("accs", [])
        self.losses 	= ckpt.get("losses", [])
        
        self.epoch 	= ckpt.get("epoch", -1) + 1 	# not in use
        other 			= ckpt.get("meta", {})			# ...
        
        print(f"[load] Loaded checkpoint: {path}")
        
        
        


def main():
    #--- < CONFIG > ---
    CONFIG = CNNConfig()


    if not os.path.isdir(CONFIG.DATASETS_ROOT):
        raise FileNotFoundError("DATASETS_ROOT is not a valid directory.")
    dataset_names, dataset_paths = get_available_datasets(CONFIG.DATASETS_ROOT)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = os.path.join(CONFIG.CHECKPOINTS_ROOT, "cnn_ckpt")

    #--- < MAIN > ---
    print("Device: ", device)
    start_time = time.time()

    ds, num_classes = build_dataset(dataset_paths[2])
    print("Data Load Time: " + format_time(time.time() - start_time), end="\n\n")

    # Initialize Trainer
    model = CNN()
    trainer = CNNTrainer(model, num_classes, ds, device)

    if CONFIG.LOAD_CHECKPOINT:
        try:
            trainer.load()
        except Exception as e:
            print("exception: ", e)

    # Train
    trainer.train(use_amp=True)

    # Evaluate
    eval_acc,_ = trainer.evaluate(ds)
    print("eval accuracy: ", eval_acc)

    # Save
    if CONFIG.SAVE_CHECKPOINT:
        trainer.save(ckpt_path)










if __name__ == "__main__":
    main()
print("Done.")











# FASTAI example (wrapper over torch, retains same torch functionality
#	learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
#	learn.fit_one_cycle(5, 1e-3)




