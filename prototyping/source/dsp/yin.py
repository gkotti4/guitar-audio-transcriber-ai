import librosa
import numpy as np
from math import log2
import matplotlib.pyplot as plt






class YinDsp:
    def __init__(self, fmin: float = 50.0, fmax: float = 1000.0):
        """
        Simple YIN-based pitch detector.

        fmin, fmax: expected pitch range in Hz (keep wide-ish for guitar).
        """
        self.fmin = fmin
        self.fmax = fmax

    @staticmethod
    def round_to_nearest_pitch(hz):
        """
        Given a pitch in Hz, return:
          - midi_rounded: int MIDI note number
          - note_name:    e.g. "E2"
          - midi_float:   unrounded MIDI value (for cents diff if you want)
        """
        if hz is None or np.isnan(hz) or hz <= 0:
            return None, None, None

        # convert Hz -> MIDI (float), then round to nearest semitone
        midi_float = librosa.hz_to_midi(hz)
        midi_rounded = int(np.round(midi_float))
        note_name = librosa.midi_to_note(midi_rounded)  # e.g. "E2"

        return midi_rounded, note_name, float(midi_float)

    def estimate_pitch(self, signal, target_sr):
        """
        Run YIN on a mono clip.

        Returns:
          f0_series: np.ndarray of frame-wise Hz (may contain NaNs)
          pitch_hz:  float median Hz over valid frames, or None
          note_info: dict with midi, note_name, midi_float (or all None)
        """
        # frame-wise f0 estimate
        f0_series = librosa.yin(
            signal,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=target_sr,
        )  # shape: (T,)

        # drop NaNs for robust summary
        valid = f0_series[~np.isnan(f0_series)]

        if len(valid) == 0:
            pitch_hz = None
            note_info = {
                "midi": None,
                "note_name": None,
                "midi_float": None,
            }
        else:
            pitch_hz = float(np.median(valid))
            midi_rounded, note_name, midi_float = self.round_to_nearest_pitch(pitch_hz)
            note_info = {
                "midi": midi_rounded,
                "note_name": note_name,
                "midi_float": midi_float,
            }

        return pitch_hz, note_info #, f0_series

