import os
from pathlib import Path
import numpy as np
import librosa
import librosa.feature
import soundfile as sf
import pandas as pd
import csv
#from tqdm import tqdm

# PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
# CLIPS_ROOT = os.path.join(PROJECT_ROOT, "data", "personal", "recordings", "neck")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLIPS_ROOT = PROJECT_ROOT / "data" / "personal" / "recordings" / "active" / "neck"

OUT_ROOT = PROJECT_ROOT / "data" / "personal" / "datasets" / "open notes"


TARGET_SR = 44100
DURATION = 1.5
HOP_LENGTH = 512           # librosa hop length for onset calc
MIN_SEP_SEC = 0.18         # minimum separation between onsets (seconds)
NORMALIZE_PEAK = False      # peak-normalize each slice
BACKTRACK = False
MANUAL_BACKTRACK = False
MIN_DB = 30




def load_mono(path, sr=TARGET_SR):
    y, sr_out = librosa.load(str(path), sr=sr, mono=True)
    return y

def peak_normalize(y, eps=1e-9):
    peak = np.max(np.abs(y)) + eps
    return y / peak

def save_wav(path, data, sr=TARGET_SR):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        # average channels instead of flattening (safer in case stereo is ever used)
        data = data.mean(axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure .wav extension
    if Path(path).suffix == "":
        path = Path(path).with_suffix('.wav')
    sf.write(str(path), data, sr, subtype="PCM_16")


def detect_onsets(y, sr=TARGET_SR, hop_length=HOP_LENGTH, min_sep_sec=MIN_SEP_SEC, backtrack=BACKTRACK, manual_backtrack=MANUAL_BACKTRACK):
    ''' return onset indices. applies basic backtracking and enforces min separation. '''
    # energy / onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # detect frame indices
    frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, units='frames', backtrack=backtrack)
    if len(frames) == 0:
        return np.array([], dtype=int)

    samples = librosa.frames_to_samples(frames, hop_length=hop_length)

    if manual_backtrack:
        # backtrack to nearest previous energy peak (simple approach)
        peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.01, wait=0)
        if len(peaks) > 0:
            peak_samples = librosa.frames_to_samples(peaks, hop_length=hop_length)
            bt = []
            pi = 0
            for s in samples:
                while pi < len(peak_samples) and peak_samples[pi] < s:
                    pi += 1
                cand = peak_samples[pi-1] if pi > 0 else s
                bt.append(int(cand))
            samples = np.array(bt, dtype=int)

    # enforce minimum separation
    filtered = []
    min_sep = int(min_sep_sec * sr)
    last = -1_000_000
    for s in samples:
        if s - last >= min_sep:
            filtered.append(s)
            last = s


    return [int(o) for o in filtered] #np.array(filtered, dtype=int)


def slice_at_onset(y: np.ndarray, onset_sample, duration=DURATION, sr=TARGET_SR):
    slice_len = int(duration * sr)
    start = int(onset_sample)
    end = start + slice_len
    if start >= len(y):
        return np.zeros(slice_len, dtype=np.float32)
    if end <= len(y):
        clip = y[start:end]
    else:
        clip = np.pad(y[start:], (0, end - len(y)), 'constant')
    return clip


def process_clip(wav_path, label, out_root=OUT_ROOT, normalize_peak=NORMALIZE_PEAK, sr=TARGET_SR):
    meta = []
    try:
        y = load_mono(wav_path)
    except Exception as e:
        print(f"failed to load {wav_path}: {e}")
        return meta

    onsets = detect_onsets(y)
    if len(onsets) == 0:
        print(f"[process_clip] no onsets produced for {wav_path} from [detect_onsets(y)], cannot process file")
        # fallback: use max-energy frame as single onset - MOVE TO DETECT_ONSETS()
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP_LENGTH)[0]
        if len(rms) == 0:
            onset_sample = 0
        else:
            frame_idx = int(np.argmax(rms))
            onset_sample = int(frame_idx * HOP_LENGTH)
        onsets = np.array([onset_sample], dtype=int)

    label_dir = out_root / label
    label_dir.mkdir(parents=True, exist_ok=True)

    for i, onset in enumerate(onsets):
        out_slice = slice_at_onset(y, onset)
        if normalize_peak:
            out_slice = peak_normalize(out_slice)
        onset_ms = int(1000 * (onset / sr))
        out_name = f"{wav_path.stem}_onset{onset_ms}ms_idx{i}"
        out_path = label_dir / f"{out_name}.wav"
        save_wav(out_path, out_slice)
        meta.append({
            #'audio': <np.ndarray>,
            'out_path': str(out_path),
            'label': label,
            'original_file': str(wav_path),
            'onset_s': float(onset/sr),
            'slice_idx': int(i),
            'duration_s': float(len(out_slice) / sr)
        })

    return meta


def build_dataset(clips_root=CLIPS_ROOT, out_root=OUT_ROOT):
    all_meta = []
    labels = [p for p in clips_root.iterdir() if p.is_dir()]

    # for each folder in root, process file and build meta dataset
    for label_dir in labels:
        label = label_dir.name
        wavs = sorted(label_dir.glob("*.wav"))
        if len(wavs)==0:
            print(f"skipping {label} (no wavs)")
            continue
        print(f"Processing label: {label} ({len(wavs)} files)")
        for wav_path in wavs:
            meta = process_clip(wav_path, label, out_root)
            all_meta.extend(meta)
            pass

    # write metadata.csv

    meta_csv = out_root / "metadata.csv"
    fieldnames = ['out_path','label','original_file','onset_s','slice_idx','duration_s']
    with open(meta_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_meta:
            writer.writerow(r)

    print(f"Done. Total slices: {len(all_meta)}. Metadata saved to {meta_csv}")
    return all_meta



def main():
    print("{audio_slicer} program start.\n")
    print("project root: ", PROJECT_ROOT)
    print("audio clips root: ", CLIPS_ROOT)
    print()

    if not CLIPS_ROOT.is_dir():
        raise FileNotFoundError(f"clips directory not found at: {CLIPS_ROOT}")


    if not OUT_ROOT.is_dir():
        print(f"out directory not found at: {OUT_ROOT}, creating it now ...")
        os.makedirs(OUT_ROOT, exist_ok=True)

    print("audio clip folders: ")
    for folder in CLIPS_ROOT.iterdir():
        print(folder.name)
    print()


    # --- build dataset
    meta = build_dataset(CLIPS_ROOT, OUT_ROOT)
    counts = {}
    for m in meta:
        counts[m['label']] = counts.get(m['label'], 0) + 1
    print("per-label slice counts: ")
    for label, c in counts.items():
        print(f"    {label}: {c}")






if __name__ == "__main__":
    main()
    print("\n{audio_slicer} program end.\n")

