import os, time
from datetime import datetime
from pathlib import Path
import librosa
import librosa.onset as onset
import soundfile as sf
import numpy as np



PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_NAME = "E2_Only"
TIME_STAMP = datetime.now().strftime("%m-%d_%H-%M-%S")

IN_AUDIO_ROOT = PROJECT_ROOT / "data" / "inference" / "guitar_audio"
IN_AUDIO_PATH = IN_AUDIO_ROOT / AUDIO_NAME / f"{AUDIO_NAME}.wav"        # in Track

OUT_CLIPS_ROOT = PROJECT_ROOT / "data" / "inference" / "guitar_note_clips"
OUT_CLIPS_DIR = OUT_CLIPS_ROOT / AUDIO_NAME / TIME_STAMP                # out Clip folder


MIN_IN_DB_THRESHOLD = -45.0
MIN_RMS_DB          = -37.5

TARGET_SR = 11025


HOP_LEN = (256 * 3) # window size for detection

MIN_SEP = 0.25 # secs

CLIP_LENGTH = 0.75 # final clip duration (secs)





def load_wav(path, sr=TARGET_SR):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[load_audio] File not found at: {path}")

    wav, sr_out = librosa.load(path, sr=sr, mono=True)
    return wav, sr_out



# -- audio (pre)processing
def apply_db_threshold(y, min_db=MIN_IN_DB_THRESHOLD): # zero_below_db, apply_noise_gate
    # zero out samples whose amplitude is below `min_db`

    eps = 1e-10
    amp = np.abs(y)
    amp_db = 20 * np.log10(amp + eps)

    mask = amp_db > min_db
    y_gated = y * mask.astype(float)

    return y_gated



# -- per slice preprocessing
def is_slice_loud_enough(clip, min_rms_db=MIN_RMS_DB):
    eps = 1e-10
    rms = np.sqrt(np.mean(clip**2)) # root mean squared - avg. energy or loudness of a signal
    rms_db = 20 * np.log10(rms + eps)
    return rms_db > min_rms_db




# -- onset detection slicing
def detect_onsets(y, sr=TARGET_SR, hop_len=HOP_LEN, min_sep=MIN_SEP):
    onset_env = onset.onset_strength(y=y, sr=sr, hop_length=hop_len)

    frames = onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_len, backtrack=True) # backtrack here? or manual?

    samples = librosa.frames_to_samples(frames, hop_length=hop_len)

    # enforce minimum separation
    min_samples = int(min_sep * sr)
    filtered = []
    last = -999999
    for s in samples:
        if s - last >= min_samples:
            filtered.append(s)
            last=s

    return filtered     # list of indices where onsets (notes) start


def slice_audio(y, onset_sample, next_onset, sr=TARGET_SR, length_sec=CLIP_LENGTH):
    length = int(length_sec * sr)
    start = onset_sample
    end = min(start + length, next_onset)

    print("slice end", end)

    clip = y[start:end]
    if len(clip) < length:
        clip = np.pad(clip, (0, length - len(clip)))

    return clip     # fixed-len audio snippet centered at the detected onset



def save_clip(clip, sr, out_dir, idx, onset_s):
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    fname = f"{AUDIO_NAME}__{onset_s:.3f}s_{idx}.wav"
    sf.write((out_root / fname), clip, sr)


def sliceNsave(audio_path, out_dir=OUT_CLIPS_DIR):
    y, sr = load_wav(audio_path, TARGET_SR)
    y_gated = apply_db_threshold(y, min_db=MIN_IN_DB_THRESHOLD)
    onsets = detect_onsets(y=y_gated, sr=sr)
    print("onsets: ")
    for x in onsets:
        print(x)
    for i, sample in enumerate(onsets):
        next_onset = onsets[i+1] if i + 1 < len(onsets) else onsets[-1]
        clip = slice_audio(y=y, onset_sample=sample, next_onset=next_onset, sr=sr, length_sec=CLIP_LENGTH)
        if not is_slice_loud_enough(clip, min_db=MIN_IN_DB_THRESHOLD):
            continue
        onset_s = sample / sr
        save_clip(clip, sr, out_dir, i, onset_s)

    return len(onsets)



def main():
    sliceNsave(IN_AUDIO_PATH)




if __name__ == "__main__":
    main()