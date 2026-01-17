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
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── player.py        # Audio playback (sounddevice) [IMPLEMENTED]
│   │   └── loader.py        # Audio file loading [IMPLEMENTED]
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── acoustic.py      # Acoustic feature extraction [IMPLEMENTED]
│   │   ├── extended.py      # Extended measurements [NOT IMPLEMENTED]
│   │   └── cache.py         # Caching computed analyses [NOT IMPLEMENTED]
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── waveform.py      # Waveform display widget [IMPLEMENTED]
│   │   ├── spectrogram.py   # Spectrogram + overlays [IMPLEMENTED]
│   │   ├── tracks.py        # Overlay tracks [NOT IMPLEMENTED - in spectrogram.py]
│   │   └── timeline.py      # Synchronized time axis [NOT IMPLEMENTED]
│   ├── annotation/
│   │   ├── __init__.py
│   │   ├── tier.py          # Annotation tier model [IMPLEMENTED]
│   │   ├── interval.py      # Interval/point annotations [IMPLEMENTED - in tier.py]
│   │   ├── editor.py        # Annotation editing widget [IMPLEMENTED]
│   │   └── textgrid.py      # Praat TextGrid import/export [IMPLEMENTED]
│   └── ui/
│       ├── __init__.py
│       ├── main_window.py   # Main window layout [IMPLEMENTED]
│       ├── toolbar.py       # Playback and tool controls [NOT IMPLEMENTED - in main_window.py]
│       └── shortcuts.py     # Keyboard shortcuts [NOT IMPLEMENTED - in main_window.py]
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Core Features

### 1. Acoustic Displays (Synchronized) [IMPLEMENTED]
All displays share a common time axis and zoom/scroll together:

- **Waveform** - Amplitude over time (white background, black line, Praat-style)
- **Spectrogram** - Time-frequency representation (scipy with Gaussian window)

### 2. Overlay Tracks [IMPLEMENTED]
Displayed on top of spectrogram (toggleable via checkboxes):

| Track | Source | Status | Notes |
|-------|--------|--------|-------|
| Pitch (F0) | parselmouth | ✓ | Blue line, fixed 50-800 Hz range, right Y-axis |
| Intensity | parselmouth | ✓ | Yellow line, scaled to frequency range |
| Formants (F1-F4) | parselmouth | ✓ | Red dots, color fades to pink with higher bandwidth |
| Formant bandwidths | parselmouth | ✓ | Encoded in formant dot color (red=narrow, pink=wide) |
| Center of Gravity (CoG) | parselmouth | ✓ | Green line |
| HNR | parselmouth | ✓ | Bright purple dashed line |
| Spectral tilt | parselmouth | Extracted but not displayed |
| Nasal murmur ratio | parselmouth | Extracted but not displayed |

### 3. Audio Playback [IMPLEMENTED]
- Click-and-drag to select region, press Space to play selection
- Click inside selection to play it
- Click without drag to move cursor
- Scroll wheel to zoom
- Keyboard shortcuts: Space (play/pause), Escape (stop), Tab (play visible)
- Visual cursor tracking during playback

### 4. Control Panel [IMPLEMENTED]
- Spectrogram: Narrowband/Wideband toggle, colormap selection (grayscale/inferno/viridis)
- Formants: Voice preset (Female/Male/Child) - affects formant extraction parameters
- Overlays: Toggle checkboxes for Pitch, Formants, Intensity, CoG, HNR
- Auto-extract features for files under 60 seconds

### 5. Annotation System [IMPLEMENTED]
- Multiple annotation tiers (like Praat TextGrids)
- Interval tiers with boundaries and labels
- Double-click to add boundaries, click to select intervals
- Inline text editor for interval labels
- Boundary snapping to upper tier boundaries (15ms threshold)
- Interval duration display
- Play button on selected intervals
- Import/export Praat TextGrid format (.TextGrid)
- Undo support (Ctrl+Z) for add/delete boundary and text changes
- Command-line options: `--tiers` to create predefined tiers, TextGrid file as argument

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play selection / pause |
| Escape | Stop playback / deselect interval |
| Tab | Play visible window |
| Scroll wheel | Zoom in/out (centered on cursor) |
| Horizontal scroll | Pan view |
| Double-click | Add boundary at position |
| Enter | Add boundary at cursor position |
| Delete | Delete hovered boundary (highlighted in orange) |
| Ctrl+Z | Undo (add/delete boundary, text changes) |
| 1-5 | Switch to annotation tier 1-5 |
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
1. [x] Load and display audio file (waveform + spectrogram)
2. [x] Play audio, play selection
3. [x] Display pitch and formants overlay
4. [x] Single annotation tier with interval creation
5. [x] Save/load TextGrid

## Current Implementation Notes

### Spectrogram
- Uses scipy with Gaussian window (similar to Praat's algorithm)
- High resolution: nfft = nperseg * 4, 95% overlap
- Narrowband (25ms window) is default, wideband (5ms) available
- Pre-emphasis filter applied for speech

### Formant Extraction
- Uses parselmouth (Praat's Burg algorithm)
- Voice presets affect max_formant and pitch range:
  - Female: max 5500 Hz, pitch 100-500 Hz
  - Male: max 5000 Hz, pitch 75-300 Hz
  - Child: max 8000 Hz, pitch 150-600 Hz
- Post-processing: frequency range filtering, bandwidth filtering, median smoothing, gap interpolation

### Pitch Display
- Fixed range 50-800 Hz (Praat-style)
- Displayed scaled on main plot, right Y-axis shows pitch scale in blue

### Known Issues
- Spectrogram quality not quite matching Praat (harmonics less clear)
- Waveform/spectrogram alignment may have minor differences

## Future Enhancements
- Batch processing mode
- Point tiers (in addition to interval tiers)
- Export annotations to CSV
- Spectrogram computed via Praat for exact match (resolution issues to solve)
- Additional acoustic measures display (spectral tilt, nasal ratio)
