# Guitar Audio Transcriber (In Development)

A deep learning + DSP research project focused on converting raw guitar audio into note predictions and, eventually, full tablature.  
This repository contains **early-stage prototypes**, model training pipelines, and dataset experimentation for single-note guitar transcription.

---

## Project Status

**This project is currently in active development and prototyping.**  
It is **not feature-complete, not stable, and not yet intended for public use or reproduction**.  

Expect:
- Breaking changes
- Experimental model architectures
- Frequent dataset restructuring
- Incomplete documentation
- Duplicated code
- Non-final evaluation methods
- Paths, configs, and scripts that may depend on a specific local setup

**This is a *research* codebase, not a packaged tool.**  
**This project does *not* include any of the datasets used in model training.**  

---

## Current Goals

- ✓ Build a reliable pipeline for classifying single guitar notes
- ✓ Train and compare MLP and CNN architectures
- ✓ Produce prediction score from multiple models (ensemble AI)
- ✓ Establish feature processing (MFCC / Mel-Spectrograms)
- ✓ Build dataset loaders, slicing tools, and training loops
- ~ Create dataset via slicing tool on recorded guitar clips
- Add fine-tuning and domain adaptation (acoustic → electric)
- ~ Expand beyond single-note classification
- ~ Begin mapping predictions to playable guitar tablature
- Investigate polyphonic transcription and chord recognition

---

## Technical Overview

| Component | Current State |
|---|---|
| Feature Extraction | MFCC & Mel-Spectrogram pipelines tested |
| Models | MLP baseline, CNN in progress |
| Dataset | Acoustic and electric guitar datasets under evaluation |
| Augmentation | Gain, noise, distortion, time-stretch (in progress) |
| Output | Note classification (e.g., `E2`, `F#3`) |
| End Goal | Full audio → guitar tab transcription |

---

## Development Notes

- Acoustic data currently provides the best training signal
- Electric guitar introduces domain variance that will require adaptation
- Label consistency uses **Scientific Pitch Notation** (`E2`, `F#3`, `C4`)
- Model performance is being validated before expanding the problem scope

---

## Public Usage Disclaimer

This repository is **not yet ready for cloning or external execution**.  
Many components assume local files, hardcoded paths, and in-progress tooling.

If you're interested in the project direction or future collaboration, feel free to follow or reach out.

A cleaned, reproducible, and documented version will be released once the prototype stabilizes.

---

## Planned Public Release Milestones

1. **Stable note classification model**
2. **Clean dataset format + instructions**
3. **Unified training & evaluation pipeline**
4. **Inference API for single notes**
5. **Optional tab prediction module**
6. **Public demo or notebook**

---

## Tech Stack

- Python
- PyTorch
- Librosa
- NumPy
- Scikit-learn
- Signal processing + deep learning hybrids
- ~ Pandas + pyplot

---

## Contact / Follow Progress

If this project interests you, star or follow for updates — or reach out for research discussion.

---

*This project is a work in progress — expect improvement, iteration, and change.*
