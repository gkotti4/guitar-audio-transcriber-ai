# mlp_trainer.py
import os, time, json
from datetime import datetime
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass, asdict

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
from audio.loading import *
from audio.features import *





class MLP(nn.Module):
    # num_features: number of input features
    # hidden_dim: number of neurons in the first hidden layer
    def __init__(self, num_features, hidden_dim, num_hidden_layers, num_classes, dropout = 0.1):
        super().__init__()

        self.init_args = {
            "num_features": num_features,
            "hidden_dim": hidden_dim,
            "num_hidden_layers": num_hidden_layers,
            "num_classes": num_classes,
            "dropout": dropout,
        }

        layers = []

        # build the dims list
        dims: list[int] = [hidden_dim]
        for _ in range(num_hidden_layers - 1):
            next_dim = dims[-1] // 2
            if next_dim < 8: #2:     # stop if it gets too small
                break
            dims.append(next_dim)

        # first hidden layer
        layers.append(nn.Linear(num_features, dims[0]))     
        layers.append(nn.LayerNorm(dims[0]))                
        #layers.append(nn.BatchNorm1d(dims[0]))             
        layers.append(nn.LeakyReLU(0.1))                    
        if dropout > 0:
            layers.append(nn.Dropout(dropout))              


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

        #print("MLP model created: \n", self.net, end="\n\n")

    def create_model(self, num_features, hidden_dims: list, num_classes, dropout = 0.0): 
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




