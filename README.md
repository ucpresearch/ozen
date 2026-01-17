# WaveAnnotator

A Python-based acoustic analysis and annotation tool inspired by Praat, built for rapid waveform annotation with extended acoustic measurements.

## Features

- **Waveform and Spectrogram Display** - Synchronized views with zoom/pan
- **Acoustic Overlays** - Pitch, formants, intensity, center of gravity, HNR
- **Audio Playback** - Play selections, visual cursor tracking
- **Annotation System** - Multiple tiers, Praat TextGrid import/export
- **Undo Support** - Ctrl+Z for boundary and label changes

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd waveannotator

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Open the application
python -m waveannotator

# Open with an audio file
python -m waveannotator audio.wav
```

### With TextGrid File

```bash
# Import existing TextGrid annotations
python -m waveannotator audio.wav annotations.TextGrid
```

### With Predefined Tier Names

```bash
# Create tiers automatically when audio loads
python -m waveannotator audio.wav --tiers words phones syllables

# Short form
python -m waveannotator audio.wav -t words phones
```

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

- Python 3.11+
- PyQt6
- pyqtgraph
- parselmouth
- sounddevice
- numpy, scipy
- soundfile

## License

See LICENSE file for details.
