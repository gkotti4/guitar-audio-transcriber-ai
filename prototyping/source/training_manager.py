# training_manager.py
import os, time, json
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

from config import *
from audio.audio_preprocessing import *

from training.mlp_trainer import MLP, MLPTrainer
from training.cnn_trainer import CNN, CNNTrainer




class TrainingManager:
    """

    Rules:
        - Models must be trained on SAME dataset

    """
    def __init__(self, mlp_cfg: MLPConfig | None = None, cnn_cfg: CNNConfig | None = None):
        self.mlp_cfg = mlp_cfg or MLPConfig()
        self.cnn_cfg = cnn_cfg or CNNConfig()

    # ---------- shared helpers ----------

    @staticmethod
    def _print_config(cfg):
        print("\nConfiguration Values: ")
        for k, v in asdict(cfg).items():
            print(f" -\t{k}: {v}")
        print()

    @staticmethod
    def _choose_dataset(datasets_root: Path) -> Path:
        dataset_names, dataset_paths = get_available_datasets(datasets_root=datasets_root)
        print("Available datasets:", *dataset_names, sep="\n", end="\n\n")

        dataset_index = int(input(f"Enter dataset index (0 to {len(dataset_names) - 1}): "))
        selected_dataset_path = dataset_paths[dataset_index]

        print(f"Selected dataset: {selected_dataset_path}\n")
        return selected_dataset_path

    @staticmethod
    def _get_device():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}\n")
        return device


    # ---------- MLP training ----------
    def train_mlp(self):
        start_time = time.time()
        #cfg = self.mlp_cfg if self.mlp_cfg is not None else MLPConfig()

        # config
        self._print_config(MFCCConfig())
        self._print_config(MLPConfig())

        # dataset(s)
        selected_dataset_path = self._choose_dataset(DATASETS_ROOT)
        last_time = time.time()

        # build audio loader
        audio_dataset_loader = AudioDatasetLoader(
            [selected_dataset_path],
            target_sr=TARGET_SR,
        )

        # feature extraction
        builder = MelFeatureBuilder()

        (
            train_dl,
            val_dl,
            X,
            y_encoded,
            num_classes,
            reverse_map,
            scaler
        ) = builder.build_mfcc_train_val_dataloaders(
            audio_loader=audio_dataset_loader,
            n_mfcc=MFCC_CONFIG.N_MFCC,
            batch_size=MFCC_CONFIG.BATCH_SIZE,
            val_size=0.2,
            shuffle_train=True,
            shuffle_val=False,
            normalize_audio_volume=MFCC_CONFIG.NORMALIZE_AUDIO_VOLUME,
            #normalize_features=MFCC_CONFIG.NORMALIZE_FEATURES, # depreciating
            standard_scaler=MFCC_CONFIG.STANDARD_SCALER,
            seed=42,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        print(f"audio loading & feature extraction time: {time.time() - last_time:.2f}s \n")

        xb, _ = next(iter(train_dl))
        num_features = xb.shape[1]
        print("num_features:", num_features)
        print("num_classes:", num_classes)
        print("class_labels:", [str(lbl) for lbl in reverse_map.values()])

        device = self._get_device()

        # model + trainer
        model = MLP(
            num_features,
            hidden_dim=MLP_CONFIG.HIDDEN_DIM,
            num_hidden_layers=MLP_CONFIG.NUM_HIDDEN_LAYERS,
            num_classes=num_classes,
            dropout=MLP_CONFIG.DROPOUT,
        )

        trainer = MLPTrainer(
            model,
            train_dl,
            val_dl,
            reverse_map=reverse_map,
            device=device,
            lr=MLP_CONFIG.LR,
            scaler=scaler,
        )

        print(f"Full setup time: {time.time() - start_time:.2f}s\n")
        last_time = time.time()

        # train
        trainer.train(
            epochs=MLP_CONFIG.EPOCHS,
            es_window_len=MLP_CONFIG.ES_WINDOW_LEN,
            es_slope_limit=MLP_CONFIG.ES_SLOPE_LIMIT,
            max_clip_norm=MLP_CONFIG.MAX_CLIP_NORM,
        )

        # eval (optional)
        # trainer.evaluate(cm=True, report=True, plot_metrics=True)

        # save
        if MLP_CONFIG.SAVE_CHECKPOINT:
            trainer.save(root=MLP_CONFIG.CHECKPOINTS_DIR)

        print(f"Training time: {time.time() - last_time:.2f}s\n")


    # ---------- CNN training ----------
    def train_cnn(self):
        start_time = time.time()
        #cfg = self.cnn_cfg if self.cnn_cfg is not None else CNNConfig()

        # config
        self._print_config(MelSpecConfig())
        self._print_config(CNNConfig())

        # dataset
        selected_dataset_path = self._choose_dataset(DATASETS_ROOT)
        last_time = time.time()

        # build audio loader
        audio_dataset_loader = AudioDatasetLoader(
            [selected_dataset_path],
            target_sr=TARGET_SR,
        )

        # feature extraction
        builder = MelFeatureBuilder()

        (
            train_dl,
            val_dl,
            X,
            y_encoded,
            num_classes,
            reverse_map,
        ) = builder.build_melspec_train_val_dataloaders(
            audio_loader=audio_dataset_loader,
            n_mels=MELSPEC_CONFIG.N_MELS,
            n_fft=MELSPEC_CONFIG.N_FFT,
            hop_length=MELSPEC_CONFIG.HOP_LENGTH,
            batch_size=MELSPEC_CONFIG.BATCH_SIZE,
            val_size=0.2,
            shuffle_train=True,
            shuffle_val=False,
            normalize_audio_volume=MELSPEC_CONFIG.NORMALIZE_AUDIO_VOLUME,
            seed=42,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        print(f"audio loading & feature extraction time: {time.time() - last_time:.2f}s \n")

        xb, _ = next(iter(train_dl))
        num_features = xb.shape[1]
        print("num_features:", num_features)
        print("num_classes:", num_classes)
        print("class_labels:", [str(lbl) for lbl in reverse_map.values()])

        device = self._get_device()

        # model + trainer
        model = CNN(
            num_classes,
            base_channels=CNN_CONFIG.BASE_CHANNELS,
            num_blocks=CNN_CONFIG.NUM_BLOCKS,
            hidden_dim=CNN_CONFIG.HIDDEN_DIM,
            dropout=CNN_CONFIG.DROPOUT,
            kernel_size=CNN_CONFIG.KERNEL_SIZE,
        )

        trainer = CNNTrainer(
            model,
            train_dl,
            val_dl,
            reverse_map=reverse_map,
            device=device,
            lr=CNN_CONFIG.LR,
        )

        print(f"Full setup time: {time.time() - start_time:.2f}s\n")
        last_time = time.time()

        # train
        trainer.train(
            CNN_CONFIG.EPOCHS,
            es_window_len=CNN_CONFIG.ES_WINDOW_LEN,
            es_slope_limit=CNN_CONFIG.ES_SLOPE_LIMIT,
            max_clip_norm=CNN_CONFIG.MAX_CLIP_NORM,
            use_amp=CNN_CONFIG.USE_AMP,
        )

        # eval: optional

        # save
        if CNN_CONFIG.SAVE_CHECKPOINT:
            trainer.save(root=CNN_CONFIG.CHECKPOINTS_DIR)

        print(f"Training time: {time.time() - last_time:.2f}s\n")


    # --- TODO: Implement Generalized train_model(model_type) Method (only load dataset once!)

    def train_all(self):
        stime = time.time()
        print("\t--- MLP ---")
        self.train_mlp()
        if input("Hit enter to continue or hit any key to exit: ") != "":
            return
        print("\t--- CNN ---")
        self.train_cnn()
        print(f"[train_all] Total training time: {time.time() - stime:.2f}s\n")




def main():
    manager = TrainingManager()
    #manager.train_mlp()
    #manager.train_cnn()
    manager.train_all()


if __name__ == "__main__":
    main()
















