import os, argparse

from dataset_loader import AudioDatasetLoader, AVAILABLE_DATASETS, DATASETS_NUM_CLASSES, get_dataset_num_classes
from onset_segmenter import OnsetSegmenter
from mlp_model import MLPClassifier
from cnn_model import CNNTrainer, MFCC_CNN
from logistic_regression_model import LogisticRegressionClassifier
import numpy as np
from enum import Enum

# Config
SONG_PATH = "Samples/Gb_comp.wav"

SEED = 42  
DURATION = 0.5
DATASET = AVAILABLE_DATASETS[0]
NUM_CLASSES = get_dataset_num_classes(DATASET)


class ModelType(Enum):
    mlp = 0
    cnn = 1
    log_reg = 2



def load_trained_mlp(hidden_dim=64, lr=0.0005, epochs=25000, dataset=DATASET, duration=DURATION):
    model = MLPClassifier(hidden_dim, lr, dataset, duration)
    model.train(epochs)
    return model


def load_trained_cnn(lr=0.001, wdecay=0, batch_size=64, test_size=0.2, epochs=50, seed=SEED, sr=44100, dataset=DATASET, dataset_loader_cls=AudioDatasetLoader, model_path=None, loader_path=None, duration=DURATION):

    if model_path and loader_path:
        trainer = trainer.load_model(
        "cnn_0.pkl", 
        "cnn_0.le.pkl", 
        dataset_name=dataset,
        dataset_loader_cls=dataset_loader_cls,
        batch_size=batch_size,
        test_size=test_size,
        seed=seed,
        sr=sr,
        duration=duration,
        n_mfcc=13,
        hop_length=512,
        lr=lr,
        weight_decay=wdecay
        )

    else:
        trainer = CNNTrainer(
        dataset_name=dataset,
        dataset_loader_cls=dataset_loader_cls,
        batch_size=batch_size,
        test_size=test_size,
        seed=seed,
        sr=sr,
        duration=duration,
        n_mfcc=13,
        hop_length=512,
        lr=lr,
        weight_decay=wdecay
        )

    trainer.train(epochs)
    return trainer


def load_trained_log_reg(dataset=DATASET):
    model = LogisticRegressionClassifier(dataset_name=dataset)
    model.train()
    return model

def main(): # MAKE 'TRANSCRIBE' INTO A CLASS WHICH USES ENSEMBLE AI (class Transcriber - modelA = TrainerCNN().get_the_trained_model, modelB = ..., )
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "mlp"], default="cnn")
    p.add_argument("--song", required=True, help="path to .wav file")
    p.add_argument("--dataset", choices=[0,1], default=0)
    #args = p.parse_args()

    loader = AudioDatasetLoader(duration=DURATION)
    _, _, le = loader.load_features() 
    segmenter = OnsetSegmenter(duration=DURATION)
    mfccs, times = segmenter.load_mfcc_segments(SONG_PATH)
    #audios, audios_times = segmenter.load_audio_segments(SONG_PATH)

    print(mfccs.shape)
    #print(audios.shape)
    
    model_type = ModelType.log_reg

    if model_type == ModelType.mlp:
        model = load_trained_mlp()
        notes = model.forward(mfccs)

    elif model_type == ModelType.cnn:
        model = load_trained_cnn(epochs=2)
        notes = model.forward_mfccs(mfccs)

    elif model_type == ModelType.log_reg:
        model = load_trained_log_reg()
        notes = model.forward(mfccs)
    

    for (t0, _), note in zip(times, notes):
        print(f"{t0:.2f}s → {note}")

    


 





























if __name__ == "__main__":
    main()