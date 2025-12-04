import os, time, json
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

from config import *
from audio_processing.audio_preprocessing import *

from trainers.mlp_trainer import MLP, MLPTrainer
from trainers.cnn_trainer import CNN, CNNTrainer




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
        cfg = self.mlp_cfg

        # config
        self._print_config(cfg)

        # dataset(s)
        selected_dataset_path = self._choose_dataset(cfg.DATASETS_ROOT)
        last_time = time.time()

        # build audio loader
        audio_dataset_loader = AudioDatasetLoader(
            [selected_dataset_path],
            target_sr=cfg.TARGET_SR,
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
            n_mfcc=cfg.N_MFCC,
            batch_size=cfg.BATCH_SIZE,
            val_size=0.2,
            shuffle_train=True,
            shuffle_val=False,
            normalize_audio_volume=cfg.NORMALIZE_AUDIO_VOLUME,
            normalize_features=cfg.NORMALIZE_FEATURES,
            standard_scaler=cfg.STANDARD_SCALER,
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
            hidden_dim=cfg.HIDDEN_DIM,
            num_hidden_layers=cfg.NUM_HIDDEN_LAYERS,
            num_classes=num_classes,
            dropout=cfg.DROPOUT,
        )

        trainer = MLPTrainer(
            model,
            train_dl,
            val_dl,
            reverse_map=reverse_map,
            device=device,
            lr=cfg.LR,
            scaler=scaler,
        )

        print(f"Full setup time: {time.time() - start_time:.2f}s\n")
        last_time = time.time()

        # load
        if cfg.LOAD_CHECKPOINT:
            try:
                trainer.load()
            except Exception as e:
                print("Failed to load checkpoint: ", e)

        # train
        trainer.train(
            epochs=cfg.EPOCHS,
            es_window_len=cfg.ES_WINDOW_LEN,
            es_slope_limit=cfg.ES_SLOPE_LIMIT,
            max_clip_norm=cfg.MAX_CLIP_NORM,
        )

        # eval (optional)
        # trainer.evaluate(cm=True, report=True, plot_metrics=True)

        # save
        if cfg.SAVE_CHECKPOINT:
            trainer.save(config=cfg, root=cfg.CHECKPOINTS_DIR)

        print(f"Training time: {time.time() - last_time:.2f}s\n")


    # ---------- CNN training ----------
    def train_cnn(self):
        start_time = time.time()
        cfg = self.cnn_cfg

        # config
        self._print_config(cfg)

        # dataset
        selected_dataset_path = self._choose_dataset(cfg.DATASETS_ROOT)
        last_time = time.time()

        # build audio loader
        audio_dataset_loader = AudioDatasetLoader(
            [selected_dataset_path],
            target_sr=cfg.TARGET_SR,
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
            n_mels=cfg.N_MELS,
            n_fft=cfg.N_FFT,
            hop_length=cfg.HOP_LENGTH,
            batch_size=cfg.BATCH_SIZE,
            val_size=0.2,
            shuffle_train=True,
            shuffle_val=False,
            normalize_audio_volume=cfg.NORMALIZE_AUDIO_VOLUME,
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
            base_channels=cfg.BASE_CHANNELS,
            num_blocks=cfg.NUM_BLOCKS,
            hidden_dim=cfg.HIDDEN_DIM,
            dropout=cfg.DROPOUT,
            kernel_size=cfg.KERNEL_SIZE,
        )

        trainer = CNNTrainer(
            model,
            train_dl,
            val_dl,
            reverse_map=reverse_map,
            device=device,
            lr=cfg.LR,
        )

        print(f"Full setup time: {time.time() - start_time:.2f}s\n")
        last_time = time.time()

        # train
        trainer.train(
            cfg.EPOCHS,
            es_window_len=cfg.ES_WINDOW_LEN,
            es_slope_limit=cfg.ES_SLOPE_LIMIT,
            max_clip_norm=cfg.MAX_CLIP_NORM,
            use_amp=cfg.USE_AMP,
        )

        # eval: optional

        # save
        if cfg.SAVE_CHECKPOINT:
            trainer.save(config=cfg, root=cfg.CHECKPOINTS_DIR)

        print(f"Training time: {time.time() - last_time:.2f}s\n")


    # --- TODO: Implement Generalized train_model(model_type) Method

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
















