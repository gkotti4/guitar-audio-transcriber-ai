import os, time
from pathlib import Path
from pprint import pprint

from config import *
from audio_preprocessing import AudioDatasetLoader, MelFeatureBuilder, get_available_datasets
from audio_slicer import AudioSlicer
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


    def load_model_data(
            self,
            mlp_ckpt: str = "mlp_ckpt.ckpt",
            cnn_ckpt: str = "cnn_ckpt.ckpt",
            mlp_root: str = "checkpoints/mlp/",
            cnn_root: str = "checkpoints/cnn/"
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


    def transcribe(self, audio_dir):
        # load clips in database format
        audio_loader = AudioDatasetLoader(audio_dir)

        # convert to features using the same preprocessing as trained models
        mfcc_features, melspec_features = self.feature_builder.extract_inference_features(audio_loader, self.model_configs["mlp"], self.model_configs["cnn"], self.mlp_ckpt_data["scaler"])

        # predict to note labels
        prediction = self.predictor.predict(mfcc_features, melspec_features)

        # map to TAB
        # - - - - - -

        pprint(" ".join(str(x) for x in prediction["labels"]))

        return prediction

def main():
    cfg = TranscribeConfig()
    audio_name = "E2_Only"
    audio_dir = cfg.INFERENCE_AUDIO_ROOT / audio_name

    transcriber = Transcriber()
    transcriber.transcribe(audio_dir)

    print("SUCCESS!")

    
    
    


if __name__ == "__main__":
    main()