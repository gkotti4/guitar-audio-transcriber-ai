# transcribe_live.py
from numba.np.math.mathimpl import FLT_MIN

from config import SLICER_CONFIG, TARGET_SR, CLIP_DURATION, INFERENCE_OUTPUT_ROOT
from transcribe import Transcriber
from source.audio import audio_slicer
import os, argparse
import tempfile
from pathlib import Path
from pprint import pprint, pformat
import time
from collections import deque
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from enum import Enum

### ----- VERSION NOTES ----- ###
# Only supports .wav files
### ------------------------- ###

#class InferenceState(Enum):
#    IDLE = 0
#    ACTIVE = 1
#snapshot_q = queue.Queue(maxsize=2) # thread_safe audio q, drop if slow
#note_q = queue.Queue(maxsize=2)
#state = InferenceState.IDLE.value

'''
    Prototype Notes:
        - Make sure to set related values to their respective CONFIG vars (i.e. HOP_LEN, MIN_SEP, etc)
'''


def rms_db(x: np.ndarray, eps=1e-12):
    # x: float32 audio [-1, 1]
    r = np.sqrt(np.mean(x * x) + eps)
    return 20.0 * np.log10(r + eps)





class RingBuffer():
    """ MaxLen -> N * SampleRate    """
    """ DataType -> np.float32      """
    """ Dim -> 1D np.array          """
    def __init__(self, maxlen: int):
        self.ring = deque(maxlen=maxlen)

    def push(self, data: np.ndarray):
        self.ring.extend(data.tolist())

    def pop(self):
        self.ring.pop()

    def get_buffer(self) -> np.ndarray:
        return np.array(list(self.ring), dtype=np.float32)

    def get_slice(self, i, j) -> np.ndarray:
        l = list(self.ring)

        # optional: if i > j, return zeros

        if len(l) < i or len(l) < j:
            return np.zeros((0,), dtype=np.float32)
        return np.array(l[i:j], dtype=np.float32)

    def is_full(self):
        return len(self.ring) == self.ring.maxlen

    def size(self):
        return len(self.ring)

    def clear(self):
        self.ring.clear()

    def clear_from(self, idx):
        for i in range(idx):
            self.ring.pop() # or append(zeros)
        #del self.ring[:idx]

class LiveMic:
    def __init__(self, device=None, buffer_duration=1.5, sample_rate=TARGET_SR, channels=1, blocksize=1024):
        self.device = device
        self.buffer_duration = buffer_duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.buffer_maxlen = int(self.buffer_duration * self.sample_rate)
        self.buffer = RingBuffer(maxlen=self.buffer_maxlen)
        self.note_q = queue.Queue(maxsize=2)

    def detect_onsets(self, y):
        onsets = audio_slicer.AudioSlicer.detect_onsets(y, self.sample_rate, hop_len=(256*4), min_sep=0.3)
        return onsets

    @staticmethod
    def slice_from(y: np.ndarray, i, j) -> np.ndarray:
        if len(y) < i or len(y) < j:
            return np.zeros((0,), dtype=np.float32)
        return np.array(y[i:j], dtype=np.float32)

    def live(self):
        last_t = time.time()

        # Thread 2 - Stores live audio into buffer
        def callback(indata, frame_count, time_info, status):
            nonlocal last_t
            if status:
                print(status)

            in_raw = indata[:,0].astype(np.float32) # mono
            self.buffer.push(in_raw)

            now = time.time()
            # Queue current buffer to snapshot_q
            '''if now - last_t > 0.25 and self.buffer.size() > 0:
                audio = self.buffer.get_buffer()
                # gate BEFORE clearing
                #if rms_db(audio) <= -40.0:
                    #self.buffer.clear()
                    #return

                try:
                    self.snapshot_q.put_nowait(audio)
                except queue.Full:
                    try:
                        self.snapshot_q.get_nowait()  # drop oldest
                    except queue.Empty:
                        pass
                    try:
                        self.snapshot_q.put_nowait(audio)  # keep newest
                    except queue.Full:
                        pass
                last_t = time.time()
                #self.buffer.clear()
                '''

        print("Listening to mic...Press Ctrl+C to stop.")
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.blocksize,
            callback=callback,
            dtype="float32"
        ):
            # Main Thread 1
            min_slice_t = 0.3 # seconds # MIN_SEP in config?
            min_slice_len = (min_slice_t * self.sample_rate) # index range

            try:
                while True:
                    try:
                        # X. Check current buffer for note onsets
                        if self.buffer.is_full():
                            buf = self.buffer.get_buffer() # get copy for thread/sync safety

                            onsets = self.detect_onsets(buf)
                            onsets = [int(o) for o in onsets] # redundant

                            # Slice from onsets to valid note
                            h_idx = 0
                            if len(onsets) == 1:
                                s = self.slice_from(buf, onsets[0], -1)
                                if len(s) > min_slice_len:
                                    self.note_q.put_nowait(s)
                                    h_idx = onsets[0]
                                    del onsets[:]

                            while len(onsets) >= 2:
                                s = self.slice_from(buf, onsets[0], onsets[1])
                                if len(s) > min_slice_len:
                                    self.note_q.put_nowait(s)
                                    h_idx = onsets[1]
                                    del onsets[:2]
                                else:
                                    h_idx = onsets[0]
                                    del onsets[:1]

                            self.buffer.clear_from(h_idx+1)






                            # PROTO send to queue, send entire buffer, let inference onset slicing work
                            #if len(onsets) > 0:
                            #    print(onsets)
                            #    self.note_q.put_nowait(np.array(buf[onsets[0]:], dtype=np.float32, copy=False))
                            #    self.buffer.clear()


                        # X. See if note available to send to inference
                        try:
                            note = self.note_q.get_nowait()
                            if note is not None and len(note) > 0:
                                inference(np.array(note, dtype=np.float32, copy=False), self.sample_rate) # redundant check np.array()
                        except queue.Empty:
                            pass

                        time.sleep(0.1)

                    except queue.Empty:
                        pass
            except KeyboardInterrupt:
                print("Stopping live mic...")


def inference(audio: np.ndarray, sr_in = TARGET_SR):
    print("audio:", audio.size, "min/max:", float(audio.min()), float(audio.max()))
    print("first", audio[0], "last", audio[-1])

    if audio is None or len(audio) == 0:
        print("[inference] No audio provided.")
        return

    # check audio length - must be min duration
    min_samples = int(CLIP_DURATION * sr_in)
    if audio.size < min_samples:
        return

    # optional: skip silence
    if audio_slicer.AudioSlicer.is_slice_loud_enough(audio):
        print("[inference] clip not loud enough.")
        #return


    transcriber = Transcriber()
    result = None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        temp_path = tmpdir / "live_audio.wav"
        sf.write(temp_path, audio, TARGET_SR)
        # temp directory used ONLY for sliced clips
        result = transcriber.transcribe_audio(
            audio,
            clip_duration=CLIP_DURATION,
            sr_in=sr_in,
        )

    # --- RESULTS
    labels = result["labels"]
    confs  = result["confidences"]

    # --- Print to console ---
    for i, (lab, conf) in enumerate(zip(labels, confs)):
        print(f"{i:03d}  {lab:>4}  (conf={conf:.2f})")

if __name__ == "__main__":
    listener = LiveMic()
    listener.live()


