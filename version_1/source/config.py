# config.py
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime


# ---------------------------
# META
# ---------------------------
CONFIG_VERSION = "1.0.0"

# ---------------------------
# ROOT PATHS — STATIC CONSTANTS
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent # Path.cwd().parent
DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets"
PERSONAL_DATASETS_ROOT = DATASETS_ROOT / "personal"
#SOURCE_ROOT = PROJECT_ROOT / "source"
INFERENCE_ROOT = PROJECT_ROOT / "data" / "inference"
INFERENCE_CLIPS_ROOT = INFERENCE_ROOT / "sliced_clips"
INFERENCE_AUDIO_ROOT = INFERENCE_ROOT / "in_audio"
INFERENCE_OUTPUT_ROOT = INFERENCE_ROOT / "output"
CHECKPOINTS_ROOT = PROJECT_ROOT / "data" / "checkpoints"


# ---------------------------
# GLOBAL AUDIO CONSTANTS
# ---------------------------
TARGET_SR = 11025 * (2)
CLIP_DURATION = 0.50


# ---------------------------
# CONFIG GROUPS (STATIC STRUCTS)
# ---------------------------
@dataclass(frozen=True)
class MFCCConfig:
    N_MFCC: int = 32 * 2 # original: 32 (2/2)
    BATCH_SIZE: int = 32
    #NORMALIZE_FEATURES: bool = False     # deprecated - replaced w/ scaler
    STANDARD_SCALER: bool = True
    NORMALIZE_AUDIO_VOLUME: bool = True # check effects (2/2)
    ADD_PITCH_FEATURES: bool = True       # recent addition - make sure implemented correctly in ALL mfcc feature functions


@dataclass(frozen=True)
class MelSpecConfig:
    N_MELS: int = 32 * 2
    N_FFT: int = 512 * 4 # original: 256
    HOP_LENGTH: int = 256
    BATCH_SIZE: int = 32
    NORMALIZE_AUDIO_VOLUME: bool = True # check effects (2/2)
    TO_DB: bool = True # TODO: ! not yet enforced ! (2/1)


@dataclass(frozen=True)
class MLPConfig:
    CHECKPOINTS_DIR: Path = CHECKPOINTS_ROOT / "mlp"
    DEFAULT_CKPT_NAME: str = f"mlp_v{CONFIG_VERSION}.ckpt"

    SAVE_CHECKPOINT: bool = True

    HIDDEN_DIM: int = 128
    NUM_HIDDEN_LAYERS: int = 2
    DROPOUT: float = 0.1

    LR: float = 1e-3
    DECAY: float = 1e-4

    EPOCHS: int = 10
    MAX_CLIP_NORM: float = 1.0
    ES_WINDOW_LEN: int = 4
    ES_SLOPE_LIMIT: float = -0.00015


@dataclass(frozen=True)
class CNNConfig:
    CHECKPOINTS_DIR: Path = CHECKPOINTS_ROOT / "cnn"
    DEFAULT_CKPT_NAME: str = f"cnn_v{CONFIG_VERSION}.ckpt"

    SAVE_CHECKPOINT: bool = True

    BASE_CHANNELS: int = 32
    NUM_BLOCKS: int = 3
    KERNEL_SIZE: int = 3
    HIDDEN_DIM: int = 256
    DROPOUT: float = 0.1

    LR: float = 1e-3
    DECAY: float = 1e-4

    EPOCHS: int = 3
    MAX_CLIP_NORM: float = 1.0
    ES_WINDOW_LEN: int = 4
    ES_SLOPE_LIMIT: float = -0.00015
    USE_AMP: bool = True


@dataclass(frozen=True)
class AudioSlicerConfig:
    MIN_IN_DB_THRESHOLD: float = -32.5  # only accept audio values X db or higher
    MIN_SLICE_RMS_DB: float = -37.0     # is slice loud enough

    HOP_LEN: int = 256 * 2
    MIN_SEP: float = 0.3

    ATTACK_SKIP_SEC = 0.1     # duration to jump ahead when slicing notes to avoid 'attack' portion of note (2/13)



# ---------------------------
# STATIC INSTANCES (THE REAL CONFIG OBJECTS)
# ---------------------------
MFCC_CONFIG = MFCCConfig()
MELSPEC_CONFIG = MelSpecConfig()
MLP_CONFIG = MLPConfig()
CNN_CONFIG = CNNConfig()
SLICER_CONFIG = AudioSlicerConfig()







