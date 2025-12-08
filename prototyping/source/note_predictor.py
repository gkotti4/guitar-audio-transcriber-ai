import os, time
from pathlib import Path
from config import *
from audio_processing.audio_preprocessing import *
from trainers.mlp_trainer import *
from trainers.cnn_trainer import *
import numpy as np
import torch
import dsp_algorithms.yin



class NotePredictor:
    def __init__(self, device=None):

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.mlp = None
        self.cnn = None

        self.reverse_map = None

        self.configs = {"mlp_config":None, "cnn_config":None}

        self.cnn_weight = 0.75  # try multiple predictions at once and compare {x, x, x}
        self.mlp_weight = (1.0 - self.cnn_weight)
        # yin/dsp weight?


    # - MODELS
    def load_models(
            self,
            mlp_ckpt_data: dict = None,
            cnn_ckpt_data: dict = None,
    ):
        """Loads: mlp, cnn, reverse_map from checkpoints."""

        # ---- Load MLP checkpoint ----
        #mlp_path = os.path.join(mlp_root, mlp_ckpt)
        #if not os.path.isfile(mlp_path):
        #    raise FileNotFoundError(f"[load_models] No MLP checkpoint found: {mlp_path}")
        #mlp_ckpt_data = torch.load(mlp_path, map_location="cpu", weights_only=False)

        if mlp_ckpt_data is not None:
            # initialize mlp model
            mlp_init_args = mlp_ckpt_data["model_init_args"]
            self.mlp = MLP(**mlp_init_args)

            if "model" not in mlp_ckpt_data:
                raise KeyError("[load_models] MLP checkpoint missing 'model' field")

            # load model state
            self.mlp.load_state_dict(mlp_ckpt_data["model"])
            self.mlp.to(self.device)
            self.mlp.eval()
            print(f"[load_models] Loaded MLP model")

            # optionally pick up reverse_map from MLP checkpoint (from cnn ckpt?)
            if self.reverse_map is None:
                rm = mlp_ckpt_data.get("reverse_map", None)
                if rm is not None:
                    self.reverse_map = rm
                    print("[load_models] Loaded reverse_map from MLP checkpoint.")

            # assign config
            self.configs["mlp_config"] = mlp_ckpt_data["config"]
            if self.configs["mlp_config"] is not None:
                print("[load_models] Loaded MLP config")


        # ---- Load CNN checkpoint ----
        #cnn_path = os.path.join(cnn_root, cnn_ckpt)
        #if not os.path.isfile(cnn_path):
        #    raise FileNotFoundError(f"[load_models] No MLP checkpoint found: {mlp_path}")
        #cnn_ckpt_data = torch.load(cnn_path, map_location="cpu", weights_only=False)

        if cnn_ckpt_data is not None:
            # initialize cnn model
            cnn_init_args = cnn_ckpt_data["model_init_args"]
            self.cnn = CNN(**cnn_init_args)

            if "model" not in cnn_ckpt_data:
                raise KeyError("[load_models] CNN checkpoint missing 'model' field")

            # load model state
            self.cnn.load_state_dict(cnn_ckpt_data["model"])
            self.cnn.to(self.device)
            self.cnn.eval()
            print(f"[load_models] Loaded CNN model")

            # assign config
            self.configs["cnn_config"] = cnn_ckpt_data["config"]
            if self.configs["cnn_config"] is not None:
                print("[load_models] Loaded CNN config")


        # ---- Final output ----
        print(f"[load_models] Device: {self.device}")
        if self.reverse_map is None:
            print("[load_models] Warning: reverse_map is not set; predictions will be class indices only.")



    # - PREDICT NOTES (confidence and weighting)
    def predict(self, mfcc_features=None, melspec_features=None):
        """
        Args:
            mfcc_features: tensor of mfcc features extracted from sliced clip notes
            melspec_features: tensor of melspec features extracted from sliced clip notes
        """
        if mfcc_features is None and melspec_features is None:
            raise ValueError("[predict] Must provide either mfcc_features or melspec_features")

        # --- MLP branch (MFCC) ---
        if mfcc_features is not None:
            with torch.inference_mode():
                mfcc_features = np.asarray(mfcc_features, np.float32)
                mfcc_features = torch.from_numpy(mfcc_features).to(self.device)
                logits = self.mlp.forward(mfcc_features)  # (N, D)
                mlp_probs = torch.softmax(logits, dim=-1).cpu().numpy()     # confidence

        # --- CNN branch (MelSpec) ---
        if melspec_features is not None:
            with torch.inference_mode():
                melspec_features = np.asarray(melspec_features, np.float32)
                melspec_features = torch.from_numpy(melspec_features).to(self.device)
                logits = self.cnn.forward(melspec_features)  # (N, C, H, W)
                cnn_probs = torch.softmax(logits, dim=-1).cpu().numpy()     # confidence

        # --- Choose probs / combine / weight ---
        if mlp_probs is not None and cnn_probs is not None:
            # simple weighted 'ensemble'
            probs = self.mlp_weight * mlp_probs + self.cnn_weight * cnn_probs
        elif cnn_probs is not None:
            probs = cnn_probs
        elif mlp_probs is not None:
            probs = mlp_probs
        else:
            raise RuntimeError("[predict] No model produced probs. Check model.forward(xb).")

        # --- Argmax & mapping ---
        pred_indices = np.argmax(probs, axis=1)
        pred_labels = [self.reverse_map[int(i)] for i in pred_indices]
        confidences = probs[np.arange(len(pred_indices)), pred_indices]


        return {
            "indices": pred_indices,
            "labels": pred_labels,
            "confidences": confidences,
            "probs": probs,
            "per_model_probs": {
                "mlp": mlp_probs,
                "cnn": cnn_probs,
            }
        }










def main():
    pass
    #predictor = NotePredictor()
    #predictor.load_models()


if __name__ == "__main__":
    #main()
    pass
