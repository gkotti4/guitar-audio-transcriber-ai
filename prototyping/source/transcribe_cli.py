from transcribe import Transcriber
import os, argparse
import tempfile
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox



def main():
    parser = argparse.ArgumentParser(description="Guitar Audio Transcriber – Prototype V1")
    parser.add_argument("--audio", type=str, required=False, default=None,
                        help="Path to input .wav file")
    parser.add_argument("--out", type=str, required=False, default=None,
                        help="Directory to save output file (default: ./output)")
    parser.add_argument("--console_only", type=bool, required=False, default=False,
                        help="Whether or not to save any output to disk")
    args = parser.parse_args()

    # --- Resolve / choose audio file ---
    audio_path: Path | None = None

    if args.audio is not None:
        candidate = Path(args.audio)
        if candidate.is_file() and candidate.suffix.lower() == ".wav":
            audio_path = candidate

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

    # Final safety check
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() != ".wav":
        raise ValueError(f"Input file must be a .wav file: {audio_path}")

    # --- Resolve output directory (for transcription text) ---
    if args.out is None:
        out_dir = Path(__file__).parent / "output"
    else:
        out_dir = Path(args.out)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Output file path (e.g. output/MySong_transcription.txt)
    out_file = out_dir / f"{audio_path.stem}_transcription.txt"

    console_only = args.console_only

    # --- Run transcriber with a temp folder for clips ---
    transcriber = Transcriber()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # temp directory used ONLY for sliced clips
        # (nothing permanent is left behind)
        result = transcriber.transcribe(
            audio_path=audio_path,
            out_root=tmpdir,              # where slices+dataset live
            audio_name=audio_path.stem,
            target_sr=11025,
            clip_len=0.5,
        )

    labels = result["labels"]
    confs  = result["confidences"]

    # --- Print to console ---
    for i, (lab, conf) in enumerate(zip(labels, confs)):
        print(f"{i:03d}  {lab:>4}  (conf={conf:.2f})")

    # --- Save to text file ---
    if not console_only:
        with out_file.open("w", encoding="utf-8") as f:
            for i, (lab, conf) in enumerate(zip(labels, confs)):
                f.write(f"{i},{lab},{conf:.4f}\n")
            f.write("\nFull result dict:\n")
            f.write(str(result))

    print(f"\nSaved transcription to {out_file}")


if __name__ == "__main__":
    main()
