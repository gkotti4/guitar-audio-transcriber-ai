import os, time, argparse
from pathlib import Path
from datetime import datetime
from pprint import pprint

import tkinter as tk
from tkinter import filedialog, messagebox

from config import *
from audio_processing.audio_preprocessing import AudioDatasetLoader, MelFeatureBuilder, get_available_datasets
from audio_processing.audio_slicer import AudioSlicer
from dsp_algorithms.yin import YinDsp
from note_predictor import NotePredictor

import torch
import numpy as np



class Transcriber():
    def __init__(self):

        self.device = torch.device("cpu") #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.slicer = AudioSlicer()
        self.feature_builder = MelFeatureBuilder()
        self.predictor = NotePredictor()

        self.mlp_ckpt_data = None
        self.cnn_ckpt_data = None
        self.model_configs = {"mlp": None, "cnn": None}

        self.load_model_data()
        if self.mlp_ckpt_data is None or self.cnn_ckpt_data is None:
            raise ValueError("[Transcriber] No MLP or CNN checkpoint found. load_model_data() unsuccessful.")

        self.model_configs["mlp"] = self.mlp_ckpt_data["config"]
        self.model_configs["cnn"] = self.cnn_ckpt_data["config"]
        if self.model_configs["mlp"] is None or self.model_configs["cnn"] is None:
            raise ValueError("[Transcriber] No MLP or CNN config found.")

        self.predictor.load_models(self.mlp_ckpt_data, self.cnn_ckpt_data)

        # New - define config for slicer data
        self.slicer_cfg = AudioSlicerConfig()


    def load_model_data(
            self,
            mlp_ckpt: Path | str = "mlp_ckpt.ckpt",
            cnn_ckpt: Path | str = "cnn_ckpt.ckpt",
            mlp_root: Path | str = "trainers/checkpoints/mlp/",
            cnn_root: Path | str = "trainers/checkpoints/cnn/"
    ):
        # ---- Load MLP checkpoint ----
        mlp_path = os.path.join(mlp_root, mlp_ckpt)
        if not os.path.isfile(mlp_path):
            raise FileNotFoundError(f"[load_models] No MLP checkpoint found: {mlp_path}")

        self.mlp_ckpt_data = torch.load(mlp_path, map_location="cpu", weights_only=False)
        #self.model_configs["mlp"] = self.mlp_ckpt_data["config"]

        # ---- Load CNN checkpoint ----
        cnn_path = os.path.join(cnn_root, cnn_ckpt)
        if not os.path.isfile(cnn_path):
            raise FileNotFoundError(f"[load_models] No MLP checkpoint found: {mlp_path}")

        self.cnn_ckpt_data = torch.load(cnn_path, map_location="cpu", weights_only=False)
        #self.model_configs["cnn"] = self.cnn_ckpt_data["config"]


    def transcribe(self, audio_path: Path, out_root: Path, audio_name="audio", target_sr=11025, clip_len=0.5):
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        loader_root = out_root / f"{audio_name}_{timestamp}" # needed for audio loader - simulated database
        out_dir = loader_root / f"{audio_name}"
        os.makedirs(out_dir, exist_ok=True)

        # slice audio to clips
        self.slicer.sliceNsave(audio_path, out_dir, target_sr, length_sec=clip_len,
                               hop_len=self.slicer_cfg.HOP_LEN, min_sep=self.slicer_cfg.MIN_SEP, min_db_threshold=self.slicer_cfg.MIN_IN_DB_THRESHOLD, min_slice_rms_db=self.slicer_cfg.MIN_SLICE_RMS_DB)

        # load clips in database format
        audio_loader = AudioDatasetLoader([loader_root], target_sr=target_sr, duration=clip_len)

        # convert to features using the same preprocessing as trained models
        mfcc_features, melspec_features = self.feature_builder.extract_inference_features(audio_loader, self.model_configs["mlp"], self.model_configs["cnn"], self.mlp_ckpt_data["scaler"])

        # predict to note labels
        prediction = self.predictor.predict(mfcc_features, melspec_features)

        # map to TAB
        # - - - - - -

        # DSP TESTING
        prediction["dsp_info"] = []
        wavs, _, _, _ = audio_loader.load_audio_dataset()
        yin = YinDsp()
        for wav in wavs:
            pitch_hz, note_info = yin.estimate_pitch(wav, audio_loader.target_sr)
            prediction["dsp_info"].append((pitch_hz, note_info))

        return prediction



def main():
    base_cfg = BaseConfig()

    # temporary
    audio_name = "E2_Only"
    in_audio_path = base_cfg.INFERENCE_AUDIO_ROOT / f"{audio_name}.wav"
    out_audio_root = base_cfg.INFERENCE_CLIPS_ROOT / "Transcriber"

    # minimal TK root (hidden)
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select guitar audio file",
        filetypes=(("WAV files", "*.wav"), ("All files", "*.*")),
    )
    if not file_path:
        messagebox.showerror("Error", "No file selected.")
        return

    in_audio_path = Path(file_path)
    audio_name = in_audio_path.stem

    transcriber = Transcriber()
    prediction = transcriber.transcribe(in_audio_path, out_audio_root, audio_name, target_sr=base_cfg.TARGET_SR, clip_len=base_cfg.CLIP_LENGTH) # use config here?

    print(" ".join(str(x) for x in prediction["labels"]))
    print(" ".join(f"{x:.2f}" for x in prediction["confidences"]))
    print(" ".join(f"{x[1]["midi"], x[1]["note_name"]}" for x in prediction["dsp_info"]))

    print("\nTranscriber finished.\n")

    
    
    


if __name__ == "__main__":
    main()