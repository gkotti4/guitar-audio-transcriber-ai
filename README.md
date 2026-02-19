# Guitar Audio Transcriber

An in-development audio application and research-driven project focused on monophonic guitar pitch transcription.

This system converts raw guitar audio into structured note predictions (e.g., `E2`, `F#3`) using a hybrid signal processing + deep learning pipeline.

The project sits at the intersection of:

- Practical audio tool development  
- Machine learning experimentation  
- Classical DSP 
- Real-time inference systems  

---


## Project Status

Version 1 establishes a stable foundation for single clip transcription.

The current focus is:

- Reliable inference
- Clean architecture
- Structured evaluation
- Modular expansion for future research and application features

This is an actively evolving system — usable, but still expanding in capability and scope.

---

## Development Status

This repository represents an active research and development codebase.

Version 1 prioritizes architectural clarity and model validation over deployment tooling.

**The project is evolving, and you should expect ongoing refactors, experimental features, and occasional breaking changes as the system expands.**

---

## Version 1 Goals

Version 1 focuses on **robust monophonic pitch classification**.
> Note: This version does not contain many of the features found in **'prototyping'**

### Completed

- [x] Reliable end-to-end transcription pipeline  
- [x] Onset-based automatic note slicing  
- [x] MFCC + Mel-Spectrogram feature pipelines  
- [x] CNN and MLP model architectures  
- [x] Ensemble-style inference support
- [x] YIN baseline  
- [x] Structured inference API for in-memory audio  
- [x] Scientific Pitch Notation output (`E2`, `F#3`, etc.)  
- [x] Modular, cleaned source organization  

### In Progress / Next Steps

- [ ] Domain adaptation across guitar types and effects (acoustic → electric, Clean → High Gain)  
- [ ] Expanded evaluation methodology  
- [ ] Improved segmentation robustness  
- [ ] Structured benchmark comparison against DSP (e.g., YIN)  
- [ ] Packaged application


---


## System Overview

### Processing Pipeline

Audio → Segmentation → Feature Extraction → Model Inference → Pitch Output


### Architecture

| Layer | Description |
|--------|-------------|
| Audio Processing | Onset detection and controlled slicing |
| Feature Representation | MFCC (MLP) + Mel-Spectrogram (CNN) |
| Models | PyTorch CNN (primary) + MLP (baseline / ensemble) |
| Inference | Single-clip and batch prediction |
| Output | Structured pitch predictions |

The CNN currently serves as the dominant model, with the MLP acting as a lightweight baseline and ensemble component.

---

## Upcoming / Experimental Prototypes

- Real-time streaming transcription (live mic / interface input)
- HTML viewer with synchronized audio playback
- MIDI export and DAW integration
- Position-aware modeling (pitch + string prediction)
- Extended dataset augmentation pipelines

---

## Research Direction

While the system functions as an application pipeline, it also serves as a research platform exploring:

- Learned pitch representations vs classical DSP
- Model robustness under tonal variation
- Feature representation tradeoffs
- Real-time inference constraints
- Guitar-specific modeling challenges

Long-term research directions include:

- Domain generalization across instruments and pickups  
- Position-aware fretboard inference  
- Hybrid ML + DSP arbitration strategies  
- Polyphonic transcription  

---

## Repository Scope

This repository contains:

- Clean Version 1 transcription pipeline  
- Modular audio processing components  
- Model loading and inference infrastructure  
- Organized source structure  

It does **not** include:

- Full training datasets  
- Complete reproducible training configs  
- Packaged end-user application builds  

---

## Technical Stack

- Python  
- PyTorch  
- Librosa  
- NumPy  
- Scikit-learn  
- Hybrid DSP + Deep Learning  

---

## Vision

This project is both:

- A functional transcription tool in development  
- A long-term exploration of guitar-focused ML modeling  

The goal is not only pitch detection, but understanding how learned representations interact with physical instrument constraints and real-world signal variability.

---

## Follow Development

This is a passion-driven, research-oriented build evolving toward a more complete application.

If you're interested in audio ML, transcription systems, or hybrid DSP + neural approaches, feel free to follow the project or reach out!
