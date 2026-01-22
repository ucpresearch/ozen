# Ozen

A Python-based acoustic analysis and annotation tool inspired by Praat, built for rapid waveform annotation with extended acoustic measurements.

## Authors

- **Uriel Cohen Priva** ([@ucpresearch](https://github.com/ucpresearch)) - Design, testing, and vibe-coding
- **Claude** (Anthropic) - Implementation

## Features

- **Waveform and Spectrogram Display** - Synchronized views with zoom/pan
- **Acoustic Overlays** - Pitch, formants, intensity, center of gravity, HNR
- **Audio Playback** - Play selections, visual cursor tracking
- **Annotation System** - Multiple tiers, Praat TextGrid import/export
- **Undo Support** - Ctrl+Z for boundary and label changes

## Installation

### macOS / Linux

```bash
# Clone the repository
git clone https://github.com/ucpresearch/ozen.git
cd ozen

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Windows

```powershell
# Clone the repository
git clone https://github.com/ucpresearch/ozen.git
cd ozen

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note:** On Windows, `sounddevice` includes PortAudio automatically. On macOS/Linux, you may need to install it separately:
- macOS: `brew install portaudio`
- Ubuntu/Debian: `sudo apt install portaudio19-dev`

## Updating

```bash
cd ozen
git pull
pip install -r requirements.txt  # if dependencies changed
```

On Windows, remember to activate the virtual environment first with `.venv\Scripts\activate`.

## Usage

### Basic Usage

```bash
# Open the application
python -m ozen

# Open with an audio file
python -m ozen audio.wav
```

### With TextGrid File

```bash
# Import existing TextGrid annotations
python -m ozen audio.wav annotations.TextGrid
```

### With Predefined Tier Names

```bash
# Create tiers automatically when audio loads
python -m ozen audio.wav -t words,phones
```

### With Custom Config File

```bash
# Use a custom configuration file
python -m ozen audio.wav -c myconfig.yaml
```

Config files can customize colors, formant presets, default tiers, and more. See `ozen/config.py` for available options.

## Keyboard Shortcuts

### Playback

| Key | Action |
|-----|--------|
| Space | Play selection / pause |
| Escape | Stop playback |
| Tab | Play visible window |

### Navigation

| Key | Action |
|-----|--------|
| Scroll wheel | Zoom in/out (centered on cursor) |
| Horizontal scroll | Pan left/right |

### Annotation

| Key | Action |
|-----|--------|
| Double-click | Add boundary at position |
| Enter | Add boundary at cursor position |
| Delete | Delete hovered boundary (highlighted in orange) |
| Ctrl+Z | Undo (add/delete boundary, text changes) |
| Escape | Deselect interval / close text editor |
| 1-5 | Switch to annotation tier 1-5 |

### Annotation Workflow

1. **Select an interval** - Click on a tier to select an interval
2. **Edit text** - Type to add/edit the interval label
3. **Add boundaries** - Double-click or press Enter to split intervals
4. **Delete boundaries** - Hover over a boundary (turns orange) and press Delete
5. **Play interval** - Click the green play button on selected intervals

### File Operations

| Key | Action |
|-----|--------|
| Ctrl+O | Open audio file |
| Ctrl+S | Save TextGrid (to current path, or prompts if none) |
| Ctrl+Shift+S | Save TextGrid as... |

## Save Behavior

- **Ctrl+S** saves to the current TextGrid path if one exists (from opening a file or previous save)
- If no path is set, Ctrl+S prompts for a location (same as Save As)
- **Auto-save**: Every 60 seconds, annotations are saved to a `.autosave` backup file
- **Exit confirmation**: If you have unsaved changes, you'll be prompted to save before closing
- When starting with a non-existing TextGrid path, you'll be asked if you want to create it

## Supported Formats

### Audio
- WAV, FLAC, OGG, MP3

### Annotations
- Praat TextGrid (.TextGrid, .txt)

## Requirements

- Python 3.9+
- PyQt6
- pyqtgraph
- parselmouth
- sounddevice
- numpy, scipy
- soundfile

## Known Issues

### macOS: Audio noise during playback
Setting `waveform_line_width` to greater than 1 in the config causes audio static/noise during playback on macOS. This appears to be a bug in Qt/pyqtgraph's rendering interaction with CoreAudio, not an issue with Ozen itself. The default is 1, which works fine. If you customize colors via a config file, keep this value at 1.

## Acknowledgments

Ozen relies heavily on the following projects for acoustic analysis:

**Praat** - The gold standard for phonetic analysis:
> Boersma, Paul & Weenink, David (2024). Praat: doing phonetics by computer [Computer program]. Retrieved from http://www.praat.org/

**Parselmouth** - Python bindings for Praat:
> Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. *Journal of Phonetics*, 71, 1-15. https://doi.org/10.1016/j.wocn.2018.07.001

## License

GPL v3 - see [LICENSE](LICENSE) for details.

**Note:** This project is GPL-licensed solely because of its dependency on parselmouth, which provides Python bindings to Praat. This dependency is isolated to a single file (`ozen/analysis/acoustic.py`), not spread throughout the codebase. If you create a derivative work that replaces this one acoustic analysis module with a non-GPL library, you may use the MIT license for all other components (see LICENSE for details).
