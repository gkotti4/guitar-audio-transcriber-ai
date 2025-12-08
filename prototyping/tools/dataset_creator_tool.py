import os, time
from datetime import datetime
from pathlib import Path
import librosa
import librosa.feature
import soundfile as sf
from scipy.ndimage import median_filter
import numpy as np

import re, shutil


class AudioSlicer:
    def __init__(self):
        pass

    @staticmethod
    def load_wav(path, sr=11025):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[load_audio] File not found at: {path}")

        wav, sr_out = librosa.load(path, sr=sr, mono=True)
        return wav, sr_out

    # -- audio (pre)processing
    @staticmethod
    def apply_db_threshold(y, min_db=-45.0): # zero_below_db, apply_noise_gate
        # zero out samples whose amplitude is below `min_db`

        eps = 1e-10
        amp = np.abs(y)
        amp_db = 20 * np.log10(amp + eps)

        mask = amp_db > min_db
        y_gated = y * mask.astype(float)

        return y_gated


    # - rms threshold
    @staticmethod
    def compute_rms_db(y, frame_len=2048, hop_len=512, smooth = True):
        rms = librosa.feature.rms(
            y=y,
            frame_length=frame_len,
            hop_length=hop_len,

            pad_mode="reflect",
        )[0] # shape (num_frames,)
        eps = 1e-10
        rms_db = 20 * np.log10(rms + eps)
        if smooth:
            rms_db = median_filter(rms_db, size=5)
        return rms_db

    @staticmethod
    def compute_dynamic_thresholds(
            #self,
            rms_db,
            noise_pct=20,       # add to config?
            signal_pct=75,
            gate_offset_db=6.0,
            slice_offset_db=10.0
    ):
        noise_floor = np.percentile(rms_db, noise_pct)
        signal_floor = np.percentile(rms_db, signal_pct)

        gate_db = noise_floor + gate_offset_db
        slice_min_db = noise_floor + slice_offset_db

        # clamp slice_min_db so it doesn't go crazy
        slice_min_db = max(slice_min_db, noise_floor + 5.0)
        slice_min_db = min(slice_min_db, signal_floor - 3.0)

        return gate_db, slice_min_db, (noise_floor, signal_floor)

    def apply_rms_threshold(self, y, hop_len=512):
        rms_db = self.compute_rms_db(y=y, hop_len=hop_len)

        gate_db, slice_min_db, stats = self.compute_dynamic_thresholds(rms_db)

        # True where frame energy is above gate_db
        frame_mask = rms_db > gate_db

        # expand to per-sample mask
        mask = np.repeat(frame_mask, hop_len)
        mask = mask[:len(y)]

        y_gated = y * mask.astype(float)
        return y_gated


    """noise_floor ≈ -65 dB
    
    signal_floor ≈ -25 dB
    
    gate_db ≈ -59 dB
    
    slice_min_db ≈ -55 to -45 dB (after clamping)"""



    # -- per slice preprocessing
    @staticmethod
    def is_slice_loud_enough(clip, min_rms_db=-37.5):
        eps = 1e-10
        rms = np.sqrt(np.mean(clip**2)) # root mean squared - avg. energy or loudness of a signal
        rms_db = 20 * np.log10(rms + eps)
        return rms_db > min_rms_db



    # -- onset detection slicing
    @staticmethod
    def detect_onsets(y, sr=11025, hop_len=512, min_sep=0.25):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_len)

        frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_len, backtrack=True) # backtrack here? or manual?

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

    @staticmethod
    def slice_audio(y, onset, next_onset, sr=11025, length_sec=0.5):
        length = int(length_sec * sr)
        start = onset
        end = min(start + length, next_onset)
        clip = y[start:end]
        if len(clip) < length:
            clip = np.pad(clip, (0, length - len(clip)))

        return clip, (start/sr, end/sr)     # fixed-len audio snippet centered at the detected onset


    @staticmethod
    def save_clip(clip, sr, out_dir, idx, onset_s, audio_name="clip"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{audio_name}_{onset_s:.3f}s_{idx}.wav"
        sf.write((out_dir / fname), clip, sr)


    def sliceNsave(self, audio_path, out_dir, target_sr=11025, hop_len=512, length_sec=0.5, min_sep=0.25, min_db_threshold=-45.0, min_slice_rms_db=-37.5, name_prefix="clip"):
        y, sr = self.load_wav(audio_path, target_sr)
        y_gated = self.apply_db_threshold(y=y, min_db=min_db_threshold)
        y_gated = self.apply_rms_threshold(y_gated, hop_len=hop_len)
        onsets = self.detect_onsets(y=y_gated, sr=sr, min_sep=min_sep)
        total = 0
        for i, onset in enumerate(onsets):
            next_onset = onsets[i+1] if i + 1 < len(onsets) else onsets[-1]
            clip, times = self.slice_audio(y=y, onset=onset, next_onset=next_onset, sr=sr, length_sec=length_sec)
            onset_s = onset / sr
            if not self.is_slice_loud_enough(clip, min_slice_rms_db):
                print(f"[sliceNsave] dropped clip at {onset_s:.2f}s")
                continue
            self.save_clip(clip, sr, out_dir, i, onset_s, name_prefix)
            total += 1
            print(f"[sliceNsave] saved clip from: {times[0]:.3f}s to {times[1]:.3f}s")
        print(f"[sliceNsave] total clips saved: {total}")
        print(f"audio sr: {sr}")
        return len(onsets)



def slice_all_clips(IN_ROOT_DIR=None, OUT_ROOT_DIR=None):

    print(f"\nInput Root Directory: {IN_ROOT_DIR}")
    print(f"Output Root Directory: {OUT_ROOT_DIR}\n")

    #if(input("Hit Enter to continue, or Any character to abort...") != ""):
    #    return

    slicer = AudioSlicer()

    
    for i in range(1, 7):
        string_dir = IN_ROOT_DIR / f"String_{i}"
        if not string_dir.is_dir():
            print(f"Skipping missing directory: {string_dir}")
            continue

        print(f"Processing directory: {string_dir}")

        for fret_dir in sorted(string_dir.iterdir()):
            if not fret_dir.is_dir():
                continue
            if not fret_dir.name.startswith("Fret_"):
                # ignore anything that isn't a fret folder
                continue

            print(f"  Processing fret directory: {fret_dir.name}")

            # parse fret number from "Fret_X"
            try:
                fret_num = int(fret_dir.name.split("_")[1])
            except (IndexError, ValueError):
                print(f"[warn] Could not parse fret from folder: {fret_dir.name}")
                continue

            # find wav files in this Fret_X dir
            wav_files = sorted(fret_dir.glob("*.wav"))
            if not wav_files:
                print(f"[skip] No .wav files in {fret_dir}")
                continue

            for wav_path in wav_files:
                print(f"\t[fret {fret_num}] Slicing: {wav_path.name} ...")

                out_dir = OUT_ROOT_DIR / f"String_{i}" / f"Fret_{fret_num}"
                out_dir.mkdir(parents=True, exist_ok=True)

                clip_name_prefix = f"s{i}_f{fret_num}"

                target_sr = 44100
                clip_len = 1.0

                slicer.sliceNsave(
                    audio_path=wav_path,
                    out_dir=out_dir,
                    target_sr=target_sr,
                    length_sec=clip_len,
                    hop_len=512,
                    min_sep=0.5,
                    min_db_threshold=-37.5,
                    min_slice_rms_db=-32.5,
                    name_prefix=clip_name_prefix
                )
        
    print("\n[done] Finished slicing all available String_X/Fret_Y .wav files.\n")



def count_clips(ROOT_DIR):
    for dir in ROOT_DIR.rglob("*"):
        if dir.is_dir():# and dir.name.startswith("Fret_"):
            rel_path = dir.relative_to(ROOT_DIR) 
            clip_files = list(dir.glob("*.wav"))
            num_clips = len(clip_files)
            print(f"{rel_path}: {num_clips} clips")
    


def create_pitch_dataset(
    in_root: Path,
    dataset_name: str = "guitar_pitches_banshee_bridge_v1",
    dry_run: bool = False,
):
    
    STANDARD_TUNING_MIDI = {
        6: 40,  # E2
        5: 45,  # A2
        4: 50,  # D3
        3: 55,  # G3
        2: 59,  # B3
        1: 64,  # E4
    }

    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
                "F#", "G", "G#", "A", "A#", "B"]


    def midi_to_name(midi: int) -> str:
        name   = NOTE_NAMES[midi % 12]
        octave = midi // 12 - 1
        return f"{name}{octave}"


    def string_fret_to_pitch_name(string_num: int, fret: int) -> str:
        midi = STANDARD_TUNING_MIDI[string_num] + fret
        return midi_to_name(midi)

    in_root = Path(in_root)

    out_root = in_root / dataset_name
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"\n[create_pitch_dataset]")
    print(f"  Input root:   {in_root}")
    print(f"  Output root:  {out_root}\n")

    total_copied = 0

    # String_1 ... String_6
    for string_dir in sorted(in_root.glob("String_*")):
        if not string_dir.is_dir():
            continue

        m_string = re.match(r"String_(\d+)", string_dir.name)
        if not m_string:
            print(f"[skip] Not a String_X dir: {string_dir}")
            continue

        string_num = int(m_string.group(1))
        print(f"[string] {string_dir.name} (string {string_num})")

        # Fret_0 ... Fret_22
        for fret_dir in sorted(string_dir.glob("Fret_*")):
            if not fret_dir.is_dir():
                continue

            m_fret = re.match(r"Fret_(\d+)", fret_dir.name)
            if not m_fret:
                print(f"  [skip] Not a Fret_X dir: {fret_dir}")
                continue

            fret_num = int(m_fret.group(1))

            pitch_name = string_fret_to_pitch_name(string_num, fret_num)
            pitch_dir = out_root / pitch_name
            pitch_dir.mkdir(parents=True, exist_ok=True)

            wav_files = sorted(fret_dir.glob("*.wav"))
            if not wav_files:
                print(f"  [fret {fret_num}] no .wav files, skipping.")
                continue

            print(f"  [fret {fret_num}] {len(wav_files)} clips -> {pitch_name}/")

            for idx, wav_path in enumerate(wav_files):
                # keep original stem in case you want traceability
                stem = wav_path.stem
                new_name = f"{pitch_name}_s{string_num}f{fret_num}_{idx:04d}.wav"
                dst = pitch_dir / new_name

                if dry_run:
                    print(f"    (dry) {wav_path} -> {dst}")
                else:
                    shutil.copy2(wav_path, dst)
                    total_copied += 1

    print(f"\n[done] {'Would copy' if dry_run else 'Copied'} {total_copied} clips total.\n")



def main():
    GUITAR_NAME = "Banshee"
    PICKUP_NAME = "Bridge"

    IN_ROOT_DIR = Path(__file__).parent / GUITAR_NAME / PICKUP_NAME
    OUT_ROOT_DIR = Path(__file__).parent / f"{GUITAR_NAME}_Sliced" / PICKUP_NAME

    #slice_all_clips(IN_ROOT_DIR=IN_ROOT_DIR, OUT_ROOT_DIR=OUT_ROOT_DIR)
    count_clips(ROOT_DIR=OUT_ROOT_DIR)
    #create_pitch_dataset(in_root=OUT_ROOT_DIR, dataset_name="guitar_pitches_banshee_bridge_v1", dry_run=False)
    count_clips(ROOT_DIR=OUT_ROOT_DIR / "guitar_pitches_banshee_bridge_v1")





if __name__ == "__main__":
    main()
    