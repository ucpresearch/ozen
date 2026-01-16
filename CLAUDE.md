# CLAUDE.md

## Project Overview

**WaveAnnotator** - A Python-based acoustic analysis and annotation tool inspired by Praat, built for rapid waveform annotation with extended acoustic measurements.

### Goals
- Fast, intuitive annotation of speech/audio waveforms
- Display richer acoustic information than traditional tools
- Streamlined workflow for phonetic research and speech analysis

## Tech Stack

- **Python 3.11+**
- **PyQt6** - GUI framework
- **pyqtgraph** - Fast interactive plotting for waveform/spectrogram
- **parselmouth** - Praat's acoustic analysis via Python
- **sounddevice** - Low-latency audio playback
- **numpy / scipy** - Signal processing
- **soundfile** - Audio I/O

## Python Environment

This project uses a dedicated virtual environment:

```bash
# Create and activate
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Run the application:
```bash
source .venv/bin/activate
python -m waveannotator
```

## Directory Structure

```
waveannotator/
├── waveannotator/
│   ├── __init__.py
│   ├── __main__.py          # Entry point
│   ├── app.py               # Main application window
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── player.py        # Audio playback (sounddevice)
│   │   └── loader.py        # Audio file loading
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── acoustic.py      # Parselmouth wrapper for core measurements
│   │   ├── extended.py      # CoG, SNR, nasal ratio calculations
│   │   └── cache.py         # Caching computed analyses
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── waveform.py      # Waveform display widget
│   │   ├── spectrogram.py   # Spectrogram display widget
│   │   ├── tracks.py        # Overlay tracks (pitch, formants, etc.)
│   │   └── timeline.py      # Synchronized time axis
│   ├── annotation/
│   │   ├── __init__.py
│   │   ├── tier.py          # Annotation tier model
│   │   ├── interval.py      # Interval/point annotations
│   │   ├── editor.py        # Annotation editing widget
│   │   └── textgrid.py      # Praat TextGrid import/export
│   └── ui/
│       ├── __init__.py
│       ├── main_window.py   # Main window layout
│       ├── toolbar.py       # Playback and tool controls
│       └── shortcuts.py     # Keyboard shortcuts
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Core Features

### 1. Acoustic Displays (Synchronized)
All displays share a common time axis and zoom/scroll together:

- **Waveform** - Amplitude over time
- **Spectrogram** - Time-frequency representation

### 2. Overlay Tracks
Displayed on top of spectrogram (toggleable):

| Track | Source | Notes |
|-------|--------|-------|
| Pitch (F0) | parselmouth | Blue line |
| Intensity | parselmouth | Yellow line |
| Formants (F1-F4) | parselmouth | Red dots |
| Formant bandwidths | parselmouth | Displayed alongside formants |
| Center of Gravity (CoG) | scipy/custom | Spectral centroid per frame |
| Signal-to-Noise Ratio | custom | Frame-wise SNR estimate |
| Nasal ratio (A1-P0) | custom | Amplitude difference measure |

### 3. Audio Playback
- Click-and-drag to select region, press Space to play selection
- Keyboard shortcuts for play/pause/stop
- Visual cursor tracking during playback

### 4. Annotation System
- Multiple annotation tiers (like Praat TextGrids)
- Interval tiers (segments with start/end) and point tiers
- Click to add boundaries, double-click to edit labels
- Import/export Praat TextGrid format (.TextGrid)
- Export to CSV for analysis pipelines

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play selection / pause |
| Escape | Stop playback |
| Tab | Play visible window |
| Ctrl+Scroll | Zoom in/out |
| Shift+Drag | Pan view |
| Enter | Add interval boundary at cursor |
| Backspace | Delete selected annotation |
| 1-9 | Switch to annotation tier 1-9 |
| Ctrl+S | Save annotations |
| Ctrl+O | Open audio file |

## Analysis Implementation Notes

### Performance Philosophy
Compute all values at full resolution - no downsampling. Accept longer computation times in exchange for accuracy. Values are computed on the fly when needed.

### Center of Gravity (CoG)
Spectral centroid calculated per frame:
```python
cog = np.sum(frequencies * magnitudes) / np.sum(magnitudes)
```

### Signal-to-Noise Ratio
Estimate using signal power vs. noise floor (from silent regions or spectral subtraction).

### Nasal Ratio (A1-P0)
Difference between amplitude of first harmonic (A1) and amplitude of nasal peak (P0, ~250Hz region). Requires pitch tracking to locate harmonics.

### Formant Bandwidths
Available directly from parselmouth's formant analysis - extract alongside formant frequencies.

## Development Guidelines

### Code Style
- Use type hints throughout
- Docstrings for public functions/classes
- Keep GUI logic separate from analysis logic
- Architecture should allow for future extensibility (plugins possible but not implemented)

### Testing
```bash
pytest tests/
```

### Git Workflow
- Main branch should always run
- Feature branches for new functionality
- Commit messages: imperative mood, concise

## MVP Milestone

Minimum viable version should support:
1. [ ] Load and display WAV file (waveform + spectrogram)
2. [ ] Play audio, play selection
3. [ ] Display pitch and formants overlay
4. [ ] Single annotation tier with interval creation
5. [ ] Save/load TextGrid

## Future Enhancements
- Batch processing mode
- Spectrogram style options (narrowband/wideband)
