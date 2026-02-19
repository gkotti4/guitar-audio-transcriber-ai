# transcribe_cli.py
from config import TARGET_SR, CLIP_DURATION, INFERENCE_OUTPUT_ROOT
from transcribe import Transcriber
import os, argparse
import tempfile
from pathlib import Path
from pprint import pprint, pformat

import tkinter as tk
from tkinter import filedialog, messagebox

### ----- VERSION NOTES ----- ###
# Only supports .wav files
### ------------------------- ###

def main():
    parser = argparse.ArgumentParser(description="Guitar Audio Transcriber – Prototype V1")
    parser.add_argument("--audio", type=str, required=False, default=None,
                        help="Path to input .wav file")
    parser.add_argument("--out", type=str, required=False, default=None,
                        help="Directory to save output file (default: ./output)")
    parser.add_argument("--save_clips", type=bool, required=False, default=False,
                        help="Whether or not to save any clips to disk")
    parser.add_argument("--save_results", type=bool, required=False, default=False,
                        help="Whether or not to save any output to disk")
    args = parser.parse_args()

    # --- HANDLE CLI ARGS
    # --- Resolve / Choose audio file ---
    audio_path: Path | None = None

    if args.audio is not None:
        audio_file = Path(args.audio)
        if audio_file.is_file() and audio_file.suffix.lower() == ".wav":
            audio_path = audio_file

    # If no valid --audio given, open file dialog
    if audio_path is None:
        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename(
            title="Select guitar audio file",
            filetypes=(("WAV files", "*.wav"), ("All files", "*.*")),
        )
        if not file_path:
            messagebox.showerror("Error", "No file selected.")
            return

        audio_path = Path(file_path)

    # Final audio_file safety check
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() != ".wav":
        raise ValueError(f"Input file must be a .wav file: {audio_path}")

    # --- Resolve output directory (for transcription text) ---
    if args.out is None:
        out_dir = INFERENCE_OUTPUT_ROOT # TODO: Remove for V1
    else:
        out_dir = Path(args.out)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Output file path (e.g. output/MySong_transcription.txt)
    out_file = out_dir / f"{audio_path.stem}_transcription.txt"

    save_results = args.save_results
    save_clips = args.save_clips

    # --- TRANSCRIBE AUDIO
    # --- Run transcriber with a temp folder for clips ---
    transcriber = Transcriber()
    result = None

    if save_clips:
        result = transcriber.transcribe(
            audio_path=audio_path,
            out_root=out_dir,  # where slices+dataset live
            audio_name=audio_path.stem,
            target_sr=TARGET_SR,
            clip_duration=CLIP_DURATION,
        )
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            result = transcriber.transcribe(
                audio_path=audio_path,
                out_root=tmpdir,              # where slices+dataset live
                audio_name=audio_path.stem,
                target_sr=TARGET_SR,
                clip_duration=CLIP_DURATION,
            )

    # --- RESULTS
    labels = result["labels"]
    confs  = result["confidences"]
    yin_info = result["dsp_info"]     # note to self: make shortcut

    # --- Print to console ---
    print("\nTranscription Results:")
    print("Idx |  Label |  Confidence | (YIN Note Estimate)")
    for i, (lab, conf, y_info) in enumerate(zip(labels, confs, yin_info)):
        print(f"{i:03d}  {lab:>4}  (conf={conf:.2f})  {y_info[1]["note_name"]}")
    # --- Save to text file ---
    if save_results:
        with out_file.open("w", encoding="utf-8") as f:
            for i, (lab, conf) in enumerate(zip(labels, confs)):
                f.write(f"{i},{lab},{conf:.4f}\n")
            f.write("\nFull result dict:\n")
            f.write(pformat(result))

    print(f"\nSaved transcription to {out_file}")


if __name__ == "__main__":
    print("\t- TRANSCRIBE CLI - Base Version 1.0 -\n")
    main()
