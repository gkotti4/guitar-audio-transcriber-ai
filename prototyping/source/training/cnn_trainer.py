# cnn_trainer.py
import os, time, json
from datetime import datetime
from pathlib import Path
from pprint import pprint
from dataclasses import asdict

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

from audio.audio_preprocessing import *
from config import CNNConfig



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
        in_channels: int = 1,
        base_channels: int = 32,
        num_blocks: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        kernel_size: int = 3,
        use_batchnorm: bool = True,
        use_maxpool: bool = True,
        adaptive_pool: tuple = (4, 4),
    ):
        super().__init__()

        self.init_args = {
            "num_classes": num_classes,
            "in_channels": in_channels,
            "base_channels": base_channels,
            "num_blocks": num_blocks,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "kernel_size": kernel_size,
            "use_maxpool": use_maxpool,
            "adaptive_pool": adaptive_pool,
        }

        # - Convolutions
        conv_layers = []
        channels = [in_channels] # (in: grayscale) -> (hidden: kernels)

        # Build channel progression: 1 → 32 → 64 → 128 for 3 blocks
        for b in range(num_blocks):
            in_ch = channels[-1]
            out_ch = base_channels * (2 ** b)
            channels.append(out_ch)

            conv_layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2, # output h x w stays the same as input (based on kernel size)
                )
            )
            # (normalization)
            if use_batchnorm:
                conv_layers.append(nn.BatchNorm2d(out_ch))

            # activation
            conv_layers.append(nn.LeakyReLU(inplace=True))      # Recent Change: to LeakyReLU from ReLU (12/5)

            # (pooling)
            if use_maxpool:
                conv_layers.append(nn.MaxPool2d(2))  # halve H,W

            # (dropout)
            if dropout > 0.0:
                conv_layers.append(nn.Dropout(dropout))

        # (adaptive pooling) to a fixed spatial size (independent of input H,W)
        conv_layers.append(nn.AdaptiveAvgPool2d(adaptive_pool))

        self.features = nn.Sequential(*conv_layers)

        # Compute flattened feature size after conv + pool
        feat_h, feat_w = adaptive_pool
        feat_dim = channels[-1] * feat_h * feat_w

        # - Classification
        classifier_layers = list()
        classifier_layers.append(nn.Flatten())

        if hidden_dim is not None and hidden_dim > 0:
            classifier_layers.extend(
                [
                    nn.Linear(feat_dim, hidden_dim),
                    nn.LeakyReLU(inplace=True),
                ]
            )
            if dropout > 0.0:
                classifier_layers.append(nn.Dropout(dropout))
            classifier_layers.append(nn.Linear(hidden_dim, num_classes))
        else:
            # Directly from flattened features to logits
            classifier_layers.append(nn.Linear(feat_dim, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

        self.net = nn.Sequential(self.features, self.classifier)

        #print("CNN model created: \n", self.net, end="\n\n")

    def forward(self, x):
        x = self.net(x)
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
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)    # + label_smoothing (12/5)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            threshold=1e-4,
        )   # + (12/5)
        self.amp_scaler = None

        self._check_dims(train_dl)
        if val_dl:
            self._check_dims(val_dl)

        self.train_dl 			= train_dl
        self.val_dl             = val_dl
        self.reverse_map        = reverse_map
        self.class_names        = [str(reverse_map[k]) for k in sorted(reverse_map)] if reverse_map else []
        self.num_classes = len(self.class_names) # !...trusting reverse_map instead of train_dl/model...!

        self.train_loss_history     = []
        self.train_accuracy_history = []
        self.val_accuracy_history   = []
        self.val_loss_history       = []
        self.epoch                  = 0

    def _check_dims(self, dl):
        # Check dimension of batch in dataloader, should be 4
        if dl is None or len(dl) == 0:
            raise ValueError("[_check_dims] dl is empty.")

        xb, _ = next(iter(dl))

        if not xb.ndim == 4:
            raise ValueError(f"[_check_dims] Unexpected tensor rank: xb.ndim = {xb.ndim}. ")


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
        scaler = torch.amp.GradScaler(self.device.type ,enabled=use_amp)

        # optional: (re)initialize histories
        self.train_loss_history = []
        self.train_accuracy_history = []

        self.model.to(self.device)
        self.model.train()
        print("[train] CNN training start.")

        for ep in range(1, epochs + 1):
            print(f"[train] EPOCH {ep} / {epochs}")

            epoch_loss_sum, correct, total = 0.0, 0, 0

            for xb, yb in train_dl:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                # ---- forward (with optional AMP) ----
                with torch.amp.autocast(self.device.type, enabled=use_amp):
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

            # ---- early stop (es); scheduler; val loss/acc ----
            val_acc, val_loss = self.evaluate()
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            if ep > int(es_window_len * 1.5):  # TODO: use with new scheduler?
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

            print(
                  f"[train] train loss: {epoch_loss:.4f} | "
                  f"train accuracy: {epoch_acc:.4f} | "
                  f"val loss: {val_loss:.4f} | "
                  f"val accuracy: {val_acc:.4f}"
            )
            print("\n...\n")

        # ---- overall training metrics ----
        if plot_metrics:
            self._2d_plot(x=np.arange(len(self.train_accuracy_history)), y=[self.train_accuracy_history, self.train_loss_history], title="Training Curves", labels=["Accuracy", "Loss"])
            self._2d_plot(np.arange(len(self.val_accuracy_history)), [self.val_accuracy_history, self.val_loss_history], "Validation Curves", ["Accuracy", "Loss"])

        print("[train] CNN training complete.\n")

    def evaluate(self, val_dl=None, use_amp=True, cm=False, report=False, plot_metrics=False): # TODO: re-implement cm, report, show_metrics
        dl = val_dl or self.val_dl
        if dl is None:
            print(f"[evaluate] No val dataloader provided.")
            return None, None

        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        preds_np = []
        y_true = []

        #print("Evaluate start.")
        eval_start_time = time.time()

        use_amp = use_amp and getattr(self.device, "type", str(self.device)) == "cuda"

        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                # Forward pass (with optional AMP)
                with torch.amp.autocast(self.device.type, enabled=use_amp):
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

        #if show_metrics:
        #    # full metrics mode: normalized cm + classification report
        #    self._confusion_matrix(y_true, preds_np, classes=self.class_names, plot=True, normalize=True)
        #    self._classification_report(y_true, preds_np, target_names=self.class_names)
        #else:
        if cm:
            self._confusion_matrix(y_true, preds_np, classes=self.class_names, plot=plot_metrics)
        if report:
            self._classification_report(y_true, preds_np, target_names=self.class_names)

        #print("Eval time: ", format_time(time.time() - eval_start_time))
        #print(f"[evaluate] val accuracy: {acc:.4f}, val loss: {avg_loss:.4f}")
        return acc, avg_loss
    
    def save(self, filename=CNN_CONFIG.DEFAULT_CKPT_NAME, root=CNN_CONFIG.CHECKPOINTS_DIR):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True) #os.makedirs(root, exist_ok=True)
        save_path = root / filename #os.path.join(root, filename)

        ckpt = {
            "meta": {
                "config_version": CONFIG_VERSION,
                "datetime": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "model_type": "cnn",
            },
            "config": {
                "features": {
                    "type": "melspec",
                    "params": asdict(MELSPEC_CONFIG),
                },
                "model": {
                    "type": "cnn",
                    "params": asdict(CNN_CONFIG),
                },
                "target_sr": TARGET_SR,
                "clip_length": CLIP_DURATION,
            },
            "model": self.model.state_dict(),
            "model_init_args": self.model.init_args,
            "optimizer": self.optimizer.state_dict(),
            "device": str(self.device),
            "train_loss_history": getattr(self, "train_loss_history", []),
            "train_accuracy_history": getattr(self, "train_accuracy_history", []),
            "val_loss_history": getattr(self, "val_loss_history", []),
            "val_accuracy_history": getattr(self, "val_accuracy_history", []),
            "epoch": getattr(self, "epoch", 0),
            "reverse_map": self.reverse_map,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
        }

        # optional AMP scaler state
        if hasattr(self, "amp_scaler") and self.amp_scaler is not None:
            try:
                ckpt["amp_scaler"] = self.amp_scaler.state_dict()
            except Exception:
                pass

        torch.save(ckpt, save_path)
        print(f"[save] Checkpoint saved to {save_path}")

    def load(self, filename = CNN_CONFIG.DEFAULT_CKPT_NAME, root = CNN_CONFIG.CHECKPOINTS_DIR):  # DEPRECIATING
        """NOTE: Trainer should be initialized with a compatible model/optimizer before loading."""
        if not os.path.isdir(root):
            raise FileNotFoundError(f"[load] No directory named: {root}")

        load_path = os.path.join(root, filename)

        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"[load] No file named: {load_path}")

        # map_location='cpu' so loading works even if original was trained on CUDA
        ckpt = torch.load(load_path, map_location="cpu")

        model_meta = ckpt.get("model_meta", None)
        if model_meta:
            saved_args = model_meta.get("init_args", {})
            current_args = getattr(self.model, "init_args", {})

            if saved_args != current_args:
                print("[load] WARNING: Mismatch between saved model init args and current model init args!")
                print("Saved:", saved_args)
                print("Current:", current_args)

        self.model.load_state_dict(ckpt.get("model", {}))
        self.optimizer.load_state_dict(ckpt.get("optimizer", {}))

        # restore device info (but you can still override externally if you want)
        self.device = torch.device(ckpt.get("device", str(self.device)))

        self.train_loss_history = ckpt.get("train_loss_history", [])
        self.train_accuracy_history = ckpt.get("train_accuracy_history", [])
        self.val_loss_history = ckpt.get("val_loss_history", [])
        self.val_accuracy_history = ckpt.get("val_accuracy_history", [])
        self.epoch = ckpt.get("epoch", 0)

        # optional AMP restore
        if "amp_scaler" in ckpt and hasattr(self, "amp_scaler") and self.amp_scaler is not None:
            try:
                self.amp_scaler.load_state_dict(ckpt["amp_scaler"])
            except Exception:
                pass

        print(f"[load] Checkpoint loaded from {load_path}")
        
        
        

'''
def main(): # DEPRECIATING - MOVING TOWARDS TRAINING MANAGER
    start_time = time.time()
    #--- < CONFIG > ---
    cnn_cfg = CNNConfig()
    print("\nConfiguration Values: ")
    for k, v in asdict(cnn_cfg).items():
        print(f" -\t{k}: {v}")


    # Get available datasets
    dataset_names, dataset_paths = get_available_datasets(datasets_root=cnn_cfg.DATASETS_ROOT)
    print("Available datasets:", *dataset_names, sep="\n", end="\n\n")
    dataset_index = int(input(f"Enter dataset index (0 to {len(dataset_names)-1}): "))
    selected_dataset_path = dataset_paths[dataset_index]
    print(f"Selected dataset: {selected_dataset_path}\n")

    last_time = time.time()


    # --- build audio loader
    audio_dataset_loader = AudioDatasetLoader(selected_dataset_path, target_sr=cnn_cfg.TARGET_SR)

    # --- feature extraction
    builder = MelFeatureBuilder()

    train_dl, val_dl, X, y_encoded, num_classes, reverse_map = builder.build_melspec_train_val_dataloaders(
        audio_loader=audio_dataset_loader,
        n_mels=cnn_cfg.N_MELS,
        n_fft=cnn_cfg.N_FFT,
        hop_length=cnn_cfg.HOP_LENGTH,
        batch_size=cnn_cfg.BATCH_SIZE,
        val_size=0.2,
        shuffle_train=True,
        shuffle_val=False,
        normalize_audio_volume=cnn_cfg.NORMALIZE_AUDIO_VOLUME,
        #normalize_features=cnn_cfg.NORMALIZE_FEATURES,
        #standard_scaler=cnn_cfg.STANDARD_SCALER,
        seed=42,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    # -- data reports
    # builder._mfcc_report()
    # builder._audio_report()

    print(f"audio loading & feature extraction time: {time.time() - last_time:.2f}s \n")


    xb,_ = next(iter(train_dl))
    num_features = xb.shape[1]
    print("num_features:", num_features)
    print("num_classes:", num_classes)
    print("class_labels:", [str(lbl) for lbl in reverse_map.values()])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")


    # Model and Trainer setup
    model = CNN(num_classes, base_channels=cnn_cfg.BASE_CHANNELS, num_blocks=cnn_cfg.NUM_BLOCKS, hidden_dim=cnn_cfg.HIDDEN_DIM, dropout=cnn_cfg.DROPOUT, kernel_size=cnn_cfg.KERNEL_SIZE)
    trainer = CNNTrainer(model, train_dl, val_dl, reverse_map=reverse_map, device=device, lr=cnn_cfg.LR)

    print(f"Full setup time: {time.time() - start_time:.2f}s\n")
    last_time = time.time()

    # - Load
    if cnn_cfg.LOAD_CHECKPOINT:
        try:
            trainer.load()
        except Exception as e:
            print("Failed to load checkpoint: ", e)

    # - Train
    trainer.train(cnn_cfg.EPOCHS, es_window_len=cnn_cfg.ES_WINDOW_LEN, es_slope_limit=cnn_cfg.ES_SLOPE_LIMIT, max_clip_norm=cnn_cfg.MAX_CLIP_NORM, use_amp=cnn_cfg.USE_AMP)

    # - Evaluate
    # ...

    # - Save
    if cnn_cfg.SAVE_CHECKPOINT:
        trainer.save(config=cnn_cfg)

    print(f"Training time: {time.time() - last_time:.2f}s\n")

    print("\n--- cnn_trainer execution complete ---\n")
if __name__ == "__main__":
    #main()
    pass
'''





