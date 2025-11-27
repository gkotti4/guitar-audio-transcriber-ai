'''
    Functionality to seperate an audio file (.wav) into various 'windows' or segments based
    on the onset of notes - of length Duration
'''

import os, argparse
import numpy as np
import librosa
import soundfile as sf



class OnsetSegmenter():
    def __init__(self, sr=44100, hop_length=512, duration=0.5,
                padding_ms=0, backtrack=True):
        """
        Args:
            sr           : sample rate for loading
            hop_length   : frames→samples factor for onset detection
            duration     : fixed window length (seconds) per segment
            padding_ms   : extra milliseconds before/after onset
            backtrack    : whether to refine onsets to local energy minima
        """
        self.sr = sr
        self.hop = hop_length
        self.duration = duration
        self.pad = int(padding_ms * sr / 1000)   # PADDING OFF - messes with the fixed length clips need to be from different datasets/code (model needs exactly same size data)
        self.backtrack = backtrack

    def load(self, path):
        """
        Returns the full waveform and sample rate.
        """
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        return y

    def detect_onsets(self, y):
        """
        Returns sample indices of detected onsets.
        """
        frames = librosa.onset.onset_detect(
            y=y,
            sr=self.sr,
            hop_length=self.hop,
            backtrack=self.backtrack
        )
        return librosa.frames_to_samples(frames, hop_length=self.hop)

    def segment(self, y):
        """
        Returns a list of (clip, start_sec, end_sec), each of length `duration`.
        Pads or truncates as needed.
        """    
        onset_samples = self.detect_onsets(y)
        seg_len = int(self.duration * self.sr)
        segments = []

        for onset in onset_samples:
            start = max(0, onset - self.pad)
            end = onset + seg_len + self.pad
            clip = y[start:end]
            # pad if too short
            if len(clip) < seg_len + 2*self.pad:
                clip = np.pad(clip, (0, seg_len + 2*self.pad - len(clip)))
            segments.append((clip, start/self.sr, min(end, len(y))/self.sr))

        return segments

    def segment_file(self, path):
        y = self.load(path)
        onset_samples = self.detect_onsets(y)
        seg_len = int(self.duration * self.sr)
        segments = []

        for onset in onset_samples:
            start = max(0, onset - self.pad)
            end = onset + seg_len + self.pad # make sure pad doesn't effect overall duration (check if size doesn't match dataset)
            clip = y[start:end]
            # pad if too short
            if len(clip) < seg_len + 2*self.pad:
                clip = np.pad(clip, (0, seg_len + 2*self.pad - len(clip)))
            segments.append((clip, start/self.sr, min(end, len(y))/self.sr))

        return segments
    

    def load_mfcc_segments(self, path, n_mfcc=13):
        X = []
        times = []
        for segment, t0, t1 in self.segment_file(path):
            mfcc = librosa.feature.mfcc(y=segment, sr=self.sr, n_mfcc=n_mfcc).astype(np.float32)
            mfcc_mean = np.mean(mfcc, axis=1)   # (n_mfcc,)
            X.append(mfcc_mean)
            
            times.append((t0, t1))

        X = np.vstack(X)   # (n_clips, n_mfcc)
        return X, times
    

    def load_audio_segments(self, path):
        audios = []
        times = []
        for segment, t0, t1 in self.segment_file(path):
            audios.append(segment)
            times.append((t0, t1))

        #audios = np.vstack(audios)
        audios = np.array(audios)
        return audios, times





