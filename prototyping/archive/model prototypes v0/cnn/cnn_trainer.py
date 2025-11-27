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


# Torch CNN expects (batch_size, channels, height, width)
#
# 	log-mel spectrogram tensor shape (1, 128, T) : dtype should be X.float()
# 		1 	- input channel (grayscale)
#		128 - mel-frequency bins -> n_mels -> height
#		T 	- time frames -> width (depends on clip length and hop_length)
#
#	encoded_labels shape (N,) : dtype should be y.long()
#		
#
#
#






def format_time(time):
    return "{:.5f}s".format(time)

def format_float(x, precision=2):
    return "{:.{}f}".format(x, precision)


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
    def __init__(self, num_classes):
        super().__init__()
        
        self.model = nn.Sequential(
                            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                            nn.MaxPool2d(2),   # (H/2, W/2)

                            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                            nn.MaxPool2d(2),   # (H/4, W/4)

                            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                            nn.AdaptiveAvgPool2d((4,4)),      # fixed feature size

                            nn.Flatten(),
                            nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(256, num_classes)
                        )
    
    def forward(self, x):
        return self.model(x)
    

class CNNTrainer():
    def __init__(self, num_classes, train_ds, device="cpu"):
        self.device 	= device
        self.cnn 		= CNN(num_classes).to(self.device)
        self.loss_fn 	= nn.CrossEntropyLoss()
        self.opt		= torch.optim.AdamW(self.cnn.parameters(), lr=1e-3, weight_decay=1e-4)
        
        self.num_classes 		= num_classes
        self.train_ds 			= train_ds
        self.accs, self.losses 	= [], []
        
        
    def train(self, batch_size=32, epochs=20, use_amp=True): # fit
        self.cnn.train()
        use_amp = (use_amp and self.device=="cuda")
        loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        self.amp_scaler = torch.amp.GradScaler(self.device, enabled=self.device=="cuda")
        
        print("Number of classes: ", self.num_classes)
        print("Dataset X tensor shape: ", self.train_ds.tensors[0].shape)
        print("Training start.")
        train_start_time = time.time()
        
        for i in range(epochs):
            total, correct, loss_sum = 0, 0, 0.0
                                                                                                            # Optional:
                                                                                                            # 	GradScaler enables Automatic Mixed Precision (AMP)
                                                                                                            # 	to speed up training on CUDA by using float16 where safe.
            # self.amp_scaler = torch.amp.GradScaler(self.device, enabled=(use_amp and self.device=="cuda"))
            
            for xb, yb in loader: 																			# (X_batch, y_batch) in DataLoader
                                                                                                            # Move data to the correct device (GPU or CPU)
                                                                                                            # 	non_blocking=True allows asynchronous GPU transfers if pinned memory is used
                xb, yb = xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
                                                                                                            # Forward pass (with AMP)
                                                                                                            
                with torch.amp.autocast(self.device, enabled=use_amp):
                    logits 	= self.cnn(xb)
                    loss 	= self.loss_fn(logits, yb)
                    
                                                                                                            # Backpropagation setup
                self.opt.zero_grad(set_to_none=True)
                                                                                                            # Backpropagation pass + Optimizer step
                                                                                                            # AMP branch:
                                                                                                            #   scales loss to prevent underflow,
                                                                                                            #   then unscales and steps optimizer safely.
                                                                                                            # Non-AMP branch:
                                                                                                            #   standard backward() and optimizer step
                if self.amp_scaler.is_enabled() and use_amp:
                    self.amp_scaler.scale(loss).backward()
                    self.amp_scaler.step(self.opt)
                    self.amp_scaler.update()
                else:
                    loss.backward()
                    self.opt.step()
                    
                loss_sum 	+= loss.item() * xb.size(0)
                correct 	+= (logits.argmax(1) == yb).sum().item()
                total 		+= yb.size(0)
                
            acc 	 = correct/total
            avg_loss = loss_sum/total
            self.accs.append(acc)
            self.losses.append(avg_loss)
            print("Epoch ", i+1)
            print("accuracy: ", format_float(acc, 2), "avg_loss: ", format_float(avg_loss, 5))
            
        print("Train time: ", format_time(time.time()-train_start_time))
        #print("Training end.")
    
        
    def evaluate(self, eval_ds, batch_size=32, use_amp=True): 											# final model output
        self.cnn.eval()
        loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
        
        total, correct, loss_sum = 0, 0, 0.0
        print("Evaluate start.")
        eval_start_time = time.time()
                
        #with torch.no_grad():
        with torch.inference_mode():
            for xb, yb in loader:
                
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                
                with torch.amp.autocast(self.device, enabled=(use_amp and self.device=="cuda")):
                    logits = self.cnn(xb) 																	# raw, unscaled model output values
                    loss = self.loss_fn(logits, yb)
                    
                loss_sum 	+= loss.item() * xb.size(0)														# note: .item() converts tensor to a python float
                correct 	+= (logits.argmax(1) == yb).sum().item()
                total 		+= yb.size(0)
        
        acc 	 = correct/total
        avg_loss = loss_sum/total
        print("Eval time: ", format_time(time.time()-eval_start_time))
        #print("Evaluate end.")
        return acc, avg_loss
    
    def save(self, path, epoch=None, other=None):
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)		# create dir on disk if not already
        
        ckpt = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "epoch": int(epoch if epoch is not None else -1),
            "model": self.cnn.state_dict(),
            "optimizer": self.opt.state_dict(),
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
        
        self.cnn.load_state_dict(ckpt["model"], strict=True)
        
        if load_optimizer and "optimizer" in ckpt and hasattr(self, "opt"):
            self.opt.load_state_dict(ckpt["optimizer"])
            
        # optional:
        if load_amp_scaler and "amp_scaler" in ckpt and hasattr(self, "amp_scaler"):
            self.amp_scaler.load_state_dict(ckpt["amp_scaler"])
            
        self.accs 		= ckpt.get("accs", [])
        self.losses 	= ckpt.get("losses", [])
        
        current_epoch 	= ckpt.get("epoch", -1) + 1 	# not in use
        other 			= ckpt.get("meta", {})			# ...
        
        print(f"[load] Loaded checkpoint: {path}")
        
        
        


def main():
    #--- < CONFIG > ---
    DATASET_ROOT = os.path.join("..", "..", "data", "online", "datasets", "kaggle")

    TARGET_SR = 11025

    N_MFCC = 20
    BATCH_SIZE = 32
    NORMALIZE_FEATURES = False  # remember: don't use together with std scaler
    STANDARD_SCALER = True  # remember
    NORMALIZE_AUDIO_VOLUME = True  # new

    SAVE_CHECKPOINT = False
    LOAD_CHECKPOINT = False


    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_names, dataset_paths = get_available_datasets(DATASET_ROOT)
    ckpt_root = "checkpoints/"
    ckpt_path = f"{ckpt_root}trainer"
    TARGET_SR = 16000




    #--- < MAIN > ---
    print("Device: ", device)
    start_time = time.time()

    ds, num_classes = build_dataset(dataset_paths[2])
    print("Data Load Time: " + format_time(time.time() - start_time), end="\n\n")

    # Initialize Trainer
    trainer = CNNTrainer(num_classes, ds, device)
    if LOAD_CHECKPOINT:
        try:
            trainer.load(ckpt_path)
            print
        except Exception as e:
            print("exception: ", e)

    # Train
    trainer.train(use_amp=True)

    # Evaluate
    eval_acc,_ = trainer.evaluate(ds)
    print("eval accuracy: ", eval_acc)

    # Save
    if SAVE_CHECKPOINT:
        trainer.save(ckpt_path)










if __name__ == "__main__":
    main()
print("Done.")



# FASTAI example (wrapper over torch, retains same torch functionality
#	learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
#	learn.fit_one_cycle(5, 1e-3)

'''
MODEL LAYERS
1. nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

Purpose: learns local patterns — like “edges” in images or “frequency–time blobs” in spectrograms.

Parameters:

in_channels: input channels → 1 (your grayscale spec)

out_channels: number of filters to learn (e.g., 16 or 32)

kernel_size: window size (3x3 is standard)

stride: how far the window moves each step (default 1)

padding: adds border zeros to keep size constant


2. nn.ReLU()

Purpose: introduces non-linearity

Converts negative activations to zero → lets model learn complex relationships

Has no parameters.


🪣 3. nn.MaxPool2d(kernel_size, stride)

NOTE: slides a (X x X) window and keeps only the highest activation in that region

Purpose: reduces spatial size while keeping the most important features.

Example: nn.MaxPool2d(2) halves height and width.

Reduces memory and adds small invariance (like translation tolerance).

👉 (B, 16, 128, T) → (B, 16, 64, T/2)


🧱 4. nn.Flatten()

Turns (B, channels, height, width) → (B, all_features)

Prepares for dense (linear) layer.


🔗 5. nn.Linear(in_features, out_features)

Fully connected (dense) layer.

in_features = number of features coming from Flatten().

out_features = number of classes (e.g., 6 or 37 in your dataset).
'''



