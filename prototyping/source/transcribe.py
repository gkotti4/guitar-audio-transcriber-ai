# transcribe.py
import os, time, argparse
from pathlib import Path
from datetime import datetime
from pprint import pprint

import tempfile

import tkinter as tk
from tkinter import filedialog, messagebox

from config import *
from audio_processing.audio_preprocessing import AudioDatasetLoader, MelFeatureBuilder, get_available_datasets
from audio_processing.audio_slicer import AudioSlicer
from dsp_algorithms.yin import YinDsp
from note_predictor import NotePredictor

import torch
import numpy as np



class Transcriber:
    def __init__(
        self,
        mlp_ckpt: Path | str | None = None,
        cnn_ckpt: Path | str | None = None,
        mlp_root: Path | str | None = None,
        cnn_root: Path | str | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        self.slicer = AudioSlicer()
        self.feature_builder = MelFeatureBuilder()
        self.predictor = NotePredictor(device=self.device)

        # ----- Resolve checkpoint paths -----
        mlp_root = Path(mlp_root) if mlp_root else MLP_CONFIG.CHECKPOINTS_DIR
        cnn_root = Path(cnn_root) if cnn_root else CNN_CONFIG.CHECKPOINTS_DIR

        mlp_name = Path(mlp_ckpt) if mlp_ckpt else Path(MLP_CONFIG.DEFAULT_CKPT_NAME)
        cnn_name = Path(cnn_ckpt) if cnn_ckpt else Path(CNN_CONFIG.DEFAULT_CKPT_NAME)

        mlp_path = mlp_root / mlp_name
        cnn_path = cnn_root / cnn_name

        # ----- Load checkpoints -----
        if not mlp_path.is_file():
            raise FileNotFoundError(f"[Transcriber] Missing MLP checkpoint: {mlp_path}")

        if not cnn_path.is_file():
            raise FileNotFoundError(f"[Transcriber] Missing CNN checkpoint: {cnn_path}")

        self.model_ckpts = {
            "mlp": torch.load(mlp_path, map_location=self.device, weights_only=False),
            "cnn": torch.load(cnn_path, map_location=self.device, weights_only=False),
        }

        # ----- Extract model configs from checkpoints -----
        self.model_configs = {
            "mlp": self.model_ckpts["mlp"].get("config"),
            "cnn": self.model_ckpts["cnn"].get("config"),
        }

        if not self.model_configs["mlp"] or not self.model_configs["cnn"]:
            raise ValueError("[Transcriber] Checkpoints missing 'config' field.")

        # ----- Initialize predictor models -----
        self.predictor.load_models(
            self.model_ckpts["mlp"],
            self.model_ckpts["cnn"],
        )

    def transcribe(
        self,
        audio_path: Path | str,
        out_root: Path | str = INFERENCE_OUTPUT_ROOT,
        audio_name: str = "transcribe_audio",
        target_sr: int = TARGET_SR,
        clip_len: float = CLIP_LENGTH,
        #save_clips_to_disk: bool = False,
    ) -> dict:
        """
        Run full transcription on a single audio file.
        Returns:
            prediction dict from NotePredictor, plus optional DSP info.
        """

        audio_path = Path(audio_path)
        out_root = Path(out_root)

        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")

        # CHANGE 5: still simulate a "database" directory, but using Path cleanly
        loader_root = out_root / f"{audio_name}_{timestamp}"  # used by AudioDatasetLoader
        out_dir = loader_root / audio_name
        out_dir.mkdir(exist_ok=True, parents=True)

        # ---- 1. Slice audio into clips ----
        # CHANGE 6: use SLICER_CONFIG pulled from central config
        self.slicer.sliceNsave(
            audio_path,
            out_dir,
            target_sr,
            length_sec=clip_len,
            hop_len=SLICER_CONFIG.HOP_LEN,
            min_sep=SLICER_CONFIG.MIN_SEP,
            min_db_threshold=SLICER_CONFIG.MIN_IN_DB_THRESHOLD,
            min_slice_rms_db=SLICER_CONFIG.MIN_SLICE_RMS_DB,
        )

        # ---- 2. Load clips via AudioDatasetLoader ----
        if self.model_configs["mlp"]["target_sr"] != self.model_configs["cnn"]["target_sr"]:
            raise ValueError("[Transcriber] Target SR mismatch.")

        target_sr = self.model_configs["mlp"]["target_sr"]
        audio_loader = AudioDatasetLoader(
            [loader_root],
            target_sr=target_sr,
            duration=clip_len,
        )

        # ---- 3. Extract features using *checkpoint* configs ----
        # NOTE: self.model_configs["mlp"] / ["cnn"] are whatever trainer saved under `"config"`.
        # If moved to the "features/model/params" schema, adapt this call accordingly.
        mfcc_features, melspec_features = self.feature_builder.extract_inference_features(
            audio_loader,
            self.model_configs["mlp"]["features"]["params"],
            self.model_configs["cnn"]["features"]["params"],
            self.model_ckpts["mlp"].get("scaler"),  # CHANGE 7: read scaler from ckpt dict consistently
        )

        # ---- 4. Predict note labels ----
        prediction = self.predictor.predict(mfcc_features, melspec_features)

        # ---- 5. Optional: map to TAB (future step)
        # TODO: call tab-mapping engine here

        # ---- 6. DSP testing (YIN pitch estimation) ----
        prediction["dsp_info"] = []
        wavs, _, _, _ = audio_loader.load_audio_dataset()
        yin = YinDsp()
        for wav in wavs:
            pitch_hz, note_info = yin.estimate_pitch(wav, audio_loader.target_sr)
            prediction["dsp_info"].append((pitch_hz, note_info))

        return prediction



def main():
    pass
"""    # minimal TK root (hidden)
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
    out_audio_root = INFERENCE_CLIPS_ROOT / "Transcriber"

    transcriber = Transcriber()
    prediction = transcriber.transcribe(in_audio_path, out_audio_root, audio_name, target_sr=TARGET_SR, clip_len=CLIP_LENGTH)

    print(" ".join(str(x) for x in prediction["labels"]))
    print(" ".join(f"{x:.2f}" for x in prediction["confidences"]))
    print(" ".join(f"{x[1]["midi"], x[1]["note_name"]}" for x in prediction["dsp_info"]))

    for (x, m) in zip(prediction["labels"], prediction["dsp_info"]):
        print(x, m[1]["note_name"])

    print("\nTranscriber finished.\n")"""

    
    
    


if __name__ == "__main__":
    main()