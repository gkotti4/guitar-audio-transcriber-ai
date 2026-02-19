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

from audio.loading import *
from audio.features import *
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






