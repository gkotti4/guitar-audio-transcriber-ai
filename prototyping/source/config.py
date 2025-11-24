import os
from dataclasses import dataclass

@dataclass
class Config:
    PROJECT_ROOT        = os.path.join("..")
    DATASETS_ROOT       = os.path.join("..", "data", "online", "datasets", "kaggle")
    CHECKPOINTS_ROOT    = os.path.join(DATASETS_ROOT, "checkpoints")

# Optional: AudioConfig, MFCCConfig base classes

@dataclass
class MLPConfig(Config):
    CHECKPOINTS_ROOT = os.path.join(Config.CHECKPOINTS_ROOT, "mlp")

    TARGET_SR               = 11025

    SAVE_CHECKPOINT         = True
    LOAD_CHECKPOINT         = False

    N_MFCC                  = 20
    BATCH_SIZE              = 32
    NORMALIZE_FEATURES      = False     # note: don't use together with std scaler
    STANDARD_SCALER         = True
    NORMALIZE_AUDIO_VOLUME  = True

    HIDDEN_DIM              = 128
    NUM_HIDDEN_LAYERS       = 2
    DROPOUT                 = 0.1

    LR                      = 1e-3
    DECAY                   = 1e-4

    EPOCHS                  = 50
    ES_WINDOW_LEN           = 5
    ES_SLOPE_LIMIT          = -0.0001



@dataclass
class CNNConfig(Config):
    CHECKPOINTS_ROOT = os.path.join(Config.CHECKPOINTS_ROOT, "cnn")

    TARGET_SR               = 11025

    SAVE_CHECKPOINT         = True
    LOAD_CHECKPOINT         = False

    N_MELS                  = 128
    BATCH_SIZE              = 32
    NORMALIZE_FEATURES      = False     # note: don't use together with std scaler
    STANDARD_SCALER         = True
    NORMALIZE_AUDIO_VOLUME  = True

    DROPOUT                 = 0.1

    LR                      = 1e-3
    DECAY                   = 1e-4

    EPOCHS                  = 50
    ES_WINDOW_LEN           = 5
    ES_SLOPE_LIMIT          = -0.0001