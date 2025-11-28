import os, time
from datetime import datetime
from pathlib import Path
import librosa
import librosa.feature
import soundfile as sf
import numpy as np



_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_AUDIO_NAME = "E2_Only"
_TIME_STAMP = datetime.now().strftime("%m-%d_%H-%M-%S")

_IN_AUDIO_ROOT = _PROJECT_ROOT / "data" / "inference" / "guitar_audio"
_IN_AUDIO_PATH = _IN_AUDIO_ROOT / _AUDIO_NAME / f"{_AUDIO_NAME}.wav"        # in Track

_OUT_CLIPS_ROOT = _PROJECT_ROOT / "data" / "inference" / "guitar_note_clips"
_OUT_CLIPS_DIR = _OUT_CLIPS_ROOT / _AUDIO_NAME / _TIME_STAMP                # out Clip folder


_MIN_IN_DB_THRESHOLD = -45.0
_MIN_SLICE_RMS_DB    = -37.5

_TARGET_SR = 11025


_HOP_LEN = (256 * 3) # window size for detection - number of samples you “move forward” between frames when computing any time–frequency - relates to sensitivity

_MIN_SEP = 0.25 # secs

_CLIP_LENGTH = 0.75 # final clip duration (secs)




class AudioSlicer:
    def __init__(self):
        pass

    @staticmethod
    def load_wav(path, sr=_TARGET_SR):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[load_audio] File not found at: {path}")

        wav, sr_out = librosa.load(path, sr=sr, mono=True)
        return wav, sr_out

    # -- audio (pre)processing
    @staticmethod
    def apply_db_threshold(y, min_db=_MIN_IN_DB_THRESHOLD): # zero_below_db, apply_noise_gate
        # zero out samples whose amplitude is below `min_db`

        eps = 1e-10
        amp = np.abs(y)
        amp_db = 20 * np.log10(amp + eps)

        mask = amp_db > min_db
        y_gated = y * mask.astype(float)

        return y_gated


    # - rms threshold
    @staticmethod
    def compute_rms_db(y, sr=_TARGET_SR, frame_len=2048, hop_len=_HOP_LEN):
        rms = librosa.feature.rms(
            y=y,
            frame_length=frame_len,
            hop_length=hop_len
        )[0] # shape (num_frames,)
        eps = 1e-10
        rms_db = 20 * np.log10(rms + eps)
        return rms_db   #, hop_len

    @staticmethod
    def compute_dynamic_thresholds(
            #self,
            rms_db,
            noise_pct=20,
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

    def apply_rms_threshold(self, y, hop_len=_HOP_LEN):
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
    def is_slice_loud_enough(clip, min_rms_db=_MIN_SLICE_RMS_DB):
        eps = 1e-10
        rms = np.sqrt(np.mean(clip**2)) # root mean squared - avg. energy or loudness of a signal
        rms_db = 20 * np.log10(rms + eps)
        return rms_db > min_rms_db



    # -- onset detection slicing
    @staticmethod
    def detect_onsets(y, sr=_TARGET_SR, hop_len=_HOP_LEN, min_sep=_MIN_SEP):
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
    def slice_audio(y, onset, next_onset, sr=_TARGET_SR, length_sec=_CLIP_LENGTH):
        length = int(length_sec * sr)
        start = onset
        end = min(start + length, next_onset)
        clip = y[start:end]
        if len(clip) < length:
            clip = np.pad(clip, (0, length - len(clip)))

        return clip, (start/sr, end/sr)     # fixed-len audio snippet centered at the detected onset


    @staticmethod
    def save_clip(clip, sr, out_dir, idx, onset_s):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{_AUDIO_NAME}__{onset_s:.3f}s_{idx}.wav"
        sf.write((out_dir / fname), clip, sr)


    def sliceNsave(self, audio_path, out_dir=_OUT_CLIPS_DIR, target_sr=_TARGET_SR, hop_len=_HOP_LEN, length_sec=_CLIP_LENGTH):
        y, sr = self.load_wav(audio_path, target_sr)
        y_gated = self.apply_db_threshold(y=y, min_db=_MIN_IN_DB_THRESHOLD)
        y_gated = self.apply_rms_threshold(y_gated, hop_len=hop_len)
        onsets = self.detect_onsets(y=y_gated, sr=sr)
        total = 0
        for i, onset in enumerate(onsets):
            next_onset = onsets[i+1] if i + 1 < len(onsets) else onsets[-1]
            clip, times = self.slice_audio(y=y, onset=onset, next_onset=next_onset, sr=sr, length_sec=length_sec)
            onset_s = onset / sr
            if not self.is_slice_loud_enough(clip):
                print(f"[sliceNsave] dropped clip at {onset_s:.2f}s")
                continue
            self.save_clip(clip, sr, out_dir, i, onset_s)
            total += 1
            print(f"[sliceNsave] saved clip from: {times[0]:.3f}s to {times[1]:.3f}s")
        print(f"[sliceNsave] total clips saved: {total}")
        print(f"audio sr: {sr}")
        return len(onsets)





def main():
    #slicer = AudioSlicer()
    #slicer.sliceNsave(_IN_AUDIO_PATH, _OUT_CLIPS_DIR)
    pass




if __name__ == "__main__":
    #main()
    pass