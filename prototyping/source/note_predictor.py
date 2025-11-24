
from config import *
from audio_preprocessing import *
from mlp_trainer import *
from cnn_trainer import *

import numpy as np
import torch






class NotePredictor:
    def __init__(self):

        self.mlp_ckpt = None
        self.cnn_ckpt = None

        self.mlp = None
        self.cnn = None


    # - MODELS
    def load_models(self, mlp_ckpt=None, cnn_ckpt=None):

        if self.mlp_ckpt:
            mlp_ckpt = self.mlp_ckpt

        if self.cnn_ckpt:
            cnn_ckpt = self.cnn_ckpt




    # - PREDICT & SCORE



    # - SLICING & LOADING











