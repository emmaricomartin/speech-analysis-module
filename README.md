# Speech Analysis Module (SAM)

**Speech Analysis Module** is a modular Python library designed to simplify audio preprocessing, feature extraction, visualization, and recording in experimental and clinical psychology settings.

This package provides ready-to-use functions for:

**Recording audio**  
- Compatible with USB microphones and Arduino triggers  
- Includes classes for simple and Arduino-synced recordings

**Preprocessing audio**  
- Bandpass filtering and noise gating  
- Silence trimming and duration standardization

**Analysis**  
- Onset detection  
- Power spectral density (PSD) via STFT or Welch’s method  
- Envelopes, frequency analysis, and energy-based features

**Visualization**  
- Oscillograms  
- Spectrograms  
- PSD curves in both dB and linear scale

**Features*  
- Integrated analysis with Praat via Parselmouth: Automatic extraction of pitch, jitter, shimmer, harmonicity (HNR), and other acoustic biomarkers from speech recordings
- Acoustic and spectral feature extraction with librosa: Includes MFCCs, spectral centroid, spectral bandwidth, spectral contrast, spectral rolloff, zero-crossing rate, RMS energy, chroma features, and tonal centroid features (tonnetz)

**Structure**
```bash
speech_analysis_module/
│
├── analysis/       # Core signal processing (onsets, PSD, Welch)
├── plot/           # Visualization functions (waveforms, spectrograms)
├── prepro/         # Preprocessing utilities (filters, trimming, noise gate)
├── utils/          # File I/O, transcription (Whisper), helper functions
├── psychopy/       # Recording classes for PsychoPy experiments
└── features/       # (Coming soon) Feature extraction for voice analysis


## Installation

You can install SAM directly from PyPI:

```bash
pip install speech-analysis-module


