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

from config import MLPConfig
from audio_preprocessing import *





class MLP(nn.Module):
    # num_features: number of input features
    # hidden_dim: number of neurons in the first hidden layer

    def __init__(self, num_features, hidden_dim, num_hidden_layers, num_classes, dropout = 0.1):
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

        self._check_dims(train_dl)
        if val_dl:
            self._check_dims(val_dl)

        self.train_dl = train_dl
        self.val_dl = val_dl

        self.reverse_map = reverse_map
        self.class_names = [str(reverse_map[k]) for k in sorted(reverse_map)] if reverse_map else []

        self.train_loss_history     = []
        self.train_accuracy_history = []
        self.val_accuracy_history   = []
        self.val_loss_history       = []
        self.epoch = 0
        
    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _check_dims(self, dl, model = None):
        # checks if input x feature dimensions of a dataloader fit a models in_features
        model = self.model if not model and self.model else None

        if len(dl) == 0:
            raise ValueError("[_check_dims] Provided DataLoader is empty.")
        
        xb,_ = next(iter(dl))
        xb_num_features = xb.shape[1]
        if xb_num_features != model.net[0].in_features:
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

    def train(self, epochs=20, train_dl=None, es_window_len=4, es_slope_limit=1e-5, max_clip_norm=1.0, plot_metrics=False):

        train_dl = train_dl or self.train_dl
        if not train_dl:
            print("[train] No train dataloader provided. Exiting [train].")
            return
        self._check_dims(train_dl)

        # optional: (re)initialize histories
        self.train_loss_history = []
        self.train_accuracy_history = []

        self.model.to(self.device)
        self.model.train()
        print("[train] Training start.")

        for ep in range(1, epochs+1):                               # Epoch:
            print(f"[train] EPOCH {ep}/{epochs}")

            epoch_loss_sum, correct, total = 0.0, 0, 0

            for i, (xb, yb) in enumerate(train_dl):                 # Batch:

                # ---- train ----
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)                          # reset gradients
                logits = self.model(xb)                                             # forward pass (compute logits)
                loss = self.loss_fn(logits, yb)                                     # compute loss
                loss.backward()                                                     # backward pass (compute gradients)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_clip_norm)    # clip gradients (protect from spikes and log to detect vanishing)
                self.optimizer.step()                                               # update weights (mini-batch gradient descent)

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

        print("\n[train] Training complete.\n")

    def predict(self, xb):

        self.model.eval()
        with torch.no_grad():
            xb = xb.to(self.device)
            logits = self.model(xb)                      # logits
            preds = torch.argmax(logits, dim=1)          # logits to class labels via argmax
            return preds

    def evaluate(self, val_dl=None, cm=False, report=False, show_metrics=False):

        dl = val_dl or self.val_dl
        if not dl:
            print(f"[evaluate] No val dataloader provided.")
            return None, None

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

        if show_metrics:
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
    # --- setup
    CONFIG = MLPConfig()
    start_time = time.time()
    
    # Get available datasets
    dataset_names, dataset_paths = get_available_datasets(datasets_root=CONFIG.DATASETS_ROOT)
    print("Available datasets:", *dataset_names, sep="\n", end="\n\n")
    #dataset_index = int(input(f"Enter dataset index (0 to {len(dataset_names)-1}): "))
    dataset_index = 0 # TODO: remove
    selected_dataset_path = dataset_paths[dataset_index]
    print(f"Selected dataset: {selected_dataset_path}\n")

    ckpt_time = time.time()

    # build audio loader
    audio_dataset_loader = AudioDatasetLoader(selected_dataset_path, target_sr=CONFIG.TARGET_SR)


    # --- feature extraction
    builder = MelFeatureBuilder()
    
    train_dl, val_dl, X, y_encoded, num_classes, reverse_map, scaler = builder.build_mfcc_train_val_dataloaders(
        audio_loader=audio_dataset_loader,
        n_mfcc=CONFIG.N_MFCC,
        batch_size=CONFIG.BATCH_SIZE,
        val_size=0.2,
        shuffle_train=True,
        shuffle_val=False,
        normalize_audio_volume=CONFIG.NORMALIZE_AUDIO_VOLUME,
        normalize_features=CONFIG.NORMALIZE_FEATURES,
        standard_scaler=CONFIG.STANDARD_SCALER,
        seed=42,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    # -- data reports
    #builder._mfcc_report()
    #builder._audio_report()

    print(f"audio loading & feature extraction time: {time.time() - ckpt_time:.2f}s")
    print()

    ckpt_time = time.time()

    xb,_ = next(iter(train_dl))
    num_features = xb.shape[1]
    print("num_features:", num_features)
    print("num_classes:", num_classes)
    print("class_labels:", [str(lbl) for lbl in reverse_map.values()])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    # --- model and trainer setup
    model = MLP(num_features, hidden_dim=CONFIG.HIDDEN_DIM, num_hidden_layers=CONFIG.NUM_HIDDEN_LAYERS, num_classes=num_classes, dropout=CONFIG.DROPOUT)

    trainer = MLPTrainer(model, train_dl, val_dl, reverse_map, device=device)

    print(f"Model and Trainer setup time: {time.time() - ckpt_time:.2f}s\n")

    # --- load
    if CONFIG.LOAD_CHECKPOINT:
        trainer.load()


    # --- training
    trainer.train(epochs=CONFIG.EPOCHS, es_window_len=CONFIG.ES_WINDOW_LEN, es_slope_limit=CONFIG.ES_SLOPE_LIMIT)


    # --- evaluation
    #trainer.evaluate(cm=True, report=True, show_metrics=True)


    # --- save
    if CONFIG.SAVE_CHECKPOINT:
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

"""    DATASETS_ROOT           = CONFIG.DATASETS_ROOT

    SAVE_CHECKPOINT         = CONFIG.SAVE_CHECKPOINT
    LOAD_CHECKPOINT         = CONFIG.LOAD_CHECKPOINT

    TARGET_SR               = CONFIG.TARGET_SR

    N_MFCC                  = CONFIG.N_MFCC
    BATCH_SIZE              = CONFIG.BATCH_SIZE
    NORMALIZE_FEATURES      = CONFIG.NORMALIZE_FEATURES
    STANDARD_SCALER         = CONFIG.STANDARD_SCALER
    NORMALIZE_AUDIO_VOLUME  = CONFIG.NORMALIZE_AUDIO_VOLUME

    HIDDEN_DIM              = CONFIG.HIDDEN_DIM
    NUM_HIDDEN_LAYERS       = CONFIG.NUM_HIDDEN_LAYERS
    DROPOUT                 = CONFIG.DROPOUT

    EPOCHS                  = CONFIG.EPOCHS
    ES_WINDOW_LEN           = CONFIG.ES_WINDOW_LEN
    ES_SLOPE_LIMIT          = CONFIG.ES_SLOPE_LIMIT"""




