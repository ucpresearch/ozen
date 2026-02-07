# CLAUDE.md

## Project Overview

**Ozen** - A Python-based acoustic analysis and annotation tool inspired by Praat, built for rapid waveform annotation with extended acoustic measurements.

**Repository:** https://github.com/ucpresearch/ozen

### Goals
- Fast, intuitive annotation of speech/audio waveforms
- Display richer acoustic information than traditional tools
- Streamlined workflow for phonetic research and speech analysis

## Tech Stack

- **Python 3.9+**
- **PyQt6** - GUI framework
- **pyqtgraph** - Fast interactive plotting for waveform/spectrogram
- **praatfan** - Unified acoustic analysis API with pluggable backends
- **sounddevice** - Low-latency audio playback (requires PortAudio system library)
- **matplotlib** - Offline spectrogram rendering (headless, no GUI needed)
- **numpy / scipy** - Signal processing
- **soundfile** - Audio I/O

### Acoustic Analysis Backends (via praatfan)

Ozen uses praatfan for acoustic analysis, which supports multiple backends:

| Backend | License | Source | Notes |
|---------|---------|--------|-------|
| `praatfan` | MIT | praatfan-core-clean | Pure Python (default) |
| `praatfan-rust` | MIT | praatfan-core-clean | Rust with Python bindings |
| `praatfan-core` | GPL | praatfan-core-rs | Rust with Python bindings |
| `praat` | GPL | parselmouth | Original Praat bindings |

Configure via `analysis.acoustic_backend` in config, or switch at runtime via the **Backend** dropdown in the control panel. Use `ozen.praat.yaml` config to default to the Praat backend.

## Python Environment

### macOS / Linux
```bash
# Create and activate
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Note: sounddevice requires PortAudio system library
# Ubuntu/Debian: sudo apt install portaudio19-dev
# macOS: brew install portaudio
```

### Windows
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# PortAudio is bundled with sounddevice on Windows
```

Run the application:
```bash
python -m ozen [audio.wav] [annotations.TextGrid] [-t tier1,tier2] [-c config.yaml]
```

## Directory Structure

```
ozen/
├── ozen/
│   ├── __init__.py
│   ├── __main__.py          # Entry point, CLI argument parsing
│   ├── config.py            # Configuration system (YAML/JSON)
│   ├── render.py            # Offline spectrogram renderer (matplotlib, headless)
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── player.py        # Audio playback (sounddevice)
│   │   └── loader.py        # Audio file loading
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── acoustic.py      # Praatfan wrapper for all measurements (pluggable backends)
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── waveform.py      # Waveform display widget
│   │   ├── spectrogram.py   # Spectrogram display + overlay tracks
│   │   └── data_points.py   # Data collection points model
│   ├── annotation/
│   │   ├── __init__.py
│   │   ├── tier.py          # Annotation tier model (Tier, Interval, AnnotationSet)
│   │   ├── editor.py        # Annotation editing widget
│   │   └── textgrid.py      # Praat TextGrid import/export
│   └── ui/
│       ├── __init__.py
│       └── main_window.py   # Main window layout, menus, coordination
├── tests/
├── requirements.txt
├── pyproject.toml
├── LICENSE                  # GPL v3 (due to parselmouth dependency)
└── README.md
```

## Configuration System

Ozen uses a YAML/JSON config system. Config files are searched in order:
1. `./ozen.yaml` (current directory)
2. `~/.config/ozen/config.yaml`
3. `~/.ozen.yaml`

Or specify explicitly: `python -m ozen -c myconfig.yaml`

Key config sections:
- `colors` - All UI colors as `[R, G, B, A]` arrays
- `formant_presets` - Female/Male/Child voice settings (affects `max_formant` for formant analysis only)
- `formant_filters` - Frequency/bandwidth thresholds per formant
- `pitch` - Display range (`display_floor`, `display_ceiling`) for pitch axis
- `analysis` - Pitch extraction settings (`default_pitch_floor`, `default_pitch_ceiling`) and `acoustic_backend` ('auto', 'praatfan', 'praatfan-rust', 'praatfan-core', 'praat')
- `playback` - Audio playback settings (`silence_padding` to prevent cutoff)
- `annotation.default_tiers` - Tier names to create automatically (e.g., `['words', 'phones']`)

Config can also be loaded at runtime via **File > Load Config...**

## Core Features

### 1. Acoustic Displays (Synchronized)
All displays share a common time axis and zoom/scroll together:
- **Waveform** - Amplitude over time (white background, black line, Praat-style)
- **Spectrogram** - Time-frequency representation (scipy with Gaussian window)

### 2. Overlay Tracks
Displayed on top of spectrogram (toggleable via checkboxes):

| Track | Color | Notes |
|-------|-------|-------|
| Pitch (F0) | Blue | Log-scale display for perceptual uniformity, configurable range (`pitch.display_floor/ceiling`), right Y-axis |
| Intensity | Yellow | Scaled to frequency range |
| Formants (F1-F4) | Red→Pink | Color fades to pink with higher bandwidth |
| Center of Gravity (CoG) | Green | Spectral centroid |
| HNR | Purple | Harmonics-to-noise ratio |
| Spectral tilt | Orange | Low vs high frequency energy |
| A1-P0 nasal ratio | Cyan | Requires voicing |
| Nasal murmur ratio (NMR) | Brown | Low-freq energy ratio |

### 3. Audio Playback
- Click-and-drag to select region
- Click the green play button or press Space to play selection
- Clicking on spectrogram always starts a new selection (no click-to-play inside selection)
- Visual cursor tracking during playback

### 4. Annotation System
- Multiple annotation tiers (like Praat TextGrids)
- Double-click to add boundaries, click to select intervals
- Right-click boundary to remove
- Import/export Praat TextGrid format
- Undo support (Ctrl+Z)
- Auto-save every 60s
- Default tiers from config or CLI `--tiers`

### 5. Data Collection Points
Capture acoustic measurements and annotation context at specific spectrogram positions.

**Interactions:**
- **Add point**: Double-click on spectrogram
- **Remove point**: Right-click → "Remove"
- **Move point**: Click and drag
- **Copy all points**: Ctrl+C / Cmd+C (copies as TSV with visible measurements only)
- **Undo**: Ctrl+Z (unified with annotation undo)

**Visual Design:**
- Orange vertical line spanning frequency range
- Circle marker at the clicked frequency position
- Hover effect (brighter color when mouse is near)

**Quick Copy (Ctrl+C / Cmd+C):**
- Copies all data points to clipboard as tab-separated values
- Includes only measurements from visible overlays (respects checkbox state)
- Format: header row + one row per point, sorted by time
- Ready to paste into spreadsheets

**Export (File > Export Point Information...):**
- TSV format with columns: time, frequency, acoustic measurements, annotation tiers
- Annotations are looked up at export time (reflects current labels)
- Remembers last export directory

**Import (File > Import Point Information...):**
- Imports points from TSV file
- Skips duplicate points (same time and frequency)

### 6. Offline Rendering (`ozen/render.py`)
Headless CLI for publication-quality spectrogram images (no GUI required).

```bash
python -m ozen.render audio.wav -o fig.png --overlays pitch,formants --legend
```

- Uses matplotlib (Agg backend) — works on servers, in scripts, batch pipelines
- All 8 acoustic overlays with same scaling formulas as the GUI
- TextGrid annotation tiers rendered below spectrogram
- Data point sets with configurable colors (`--points red=file.tsv`); colors accept names, hex (`#RRGGBB`, `#RRGGBBAA`), or grayscale floats (`0.6`)
- `--point-markers-only` draws only circle markers (no vertical lines)
- `--point-alpha` controls point opacity (0.0–1.0, default 1.0)
- `--font` sets font family for all text (e.g., `--font "Times New Roman"`)
- Output: PNG, PDF, SVG, EPS
- `--legend` adds overlay names, colors, and value ranges
- `--man` shows full manual page
- Also usable as Python API: `from ozen.render import render_spectrogram`

Overlay scaling (matches `spectrogram.py:_update_overlays()` exactly):
- **Pitch**: `(log(f0) - log(floor)) / (log(ceil) - log(floor)) * freq_range + freq_start`
- **Intensity**: `(val - 30) / 60 * freq_range + freq_start`
- **Formants**: Direct frequency; scatter with bandwidth-based color (red→pink)
- **CoG**: Direct frequency
- **HNR**: `(val + 10) / 50 * freq_range + freq_start` (dashed)
- **Spectral tilt**: `(val + 20) / 60 * freq_range + freq_start`
- **A1-P0**: `(val + 20) / 40 * freq_range + freq_start`
- **Nasal murmur**: `val * freq_range + freq_start` (dashed)

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play selection / pause |
| Escape | Stop playback / deselect interval |
| Tab | Play visible window |
| ↑ (Up arrow) | Zoom in |
| ↓ (Down arrow) | Zoom out |
| ← (Left arrow) | Pan left |
| → (Right arrow) | Pan right |
| Scroll wheel | Zoom in/out (centered on cursor) |
| Horizontal scroll | Pan view |
| Double-click (annotation) | Add boundary at position |
| Double-click (spectrogram) | Add data collection point |
| Enter | Add boundary at cursor position |
| Delete | Delete hovered boundary (highlighted in orange) |
| Right-click | Context menu to remove boundary or point |
| Ctrl+Z / Cmd+Z | Undo (boundaries, text changes, data points) |
| Ctrl+C / Cmd+C | Copy all data points to clipboard (visible measurements only) |
| 1-5 | Switch to annotation tier 1-5 |
| Ctrl+S | Save annotations |
| Ctrl+O | Open audio file |

## CLI Options

### GUI
```bash
python -m ozen [audio_file] [textgrid_file] [options]

Options:
  --tiers, -t    Comma-separated tier names (e.g., -t words,phones)
  --config, -c   Path to custom config file (YAML or JSON)
```

### Offline Renderer
```bash
python -m ozen.render AUDIO_FILE -o OUTPUT [options]
```
Key options: `--overlays`, `--textgrid`, `--tiers`, `--points`, `--point-markers-only`, `--point-alpha`, `--legend`, `--bandwidth`, `--colormap`, `--preset`, `--start/--end`, `--pitch-floor/--pitch-ceiling`, `--width/--height/--dpi`, `--title`, `--font`, `-c/--config`, `--backend`. Run `--man` for full manual.

## Key Files by Functionality

**Annotation System:**
- `ozen/annotation/editor.py` - Main annotation editor widget (TierItem, AnnotationEditorWidget)
- `ozen/annotation/tier.py` - Data models (Tier, Interval, AnnotationSet)
- `ozen/annotation/textgrid.py` - TextGrid import/export (handles escaped quotes)

**Visualization:**
- `ozen/visualization/waveform.py` - Waveform display (WaveformWidget)
- `ozen/visualization/spectrogram.py` - Spectrogram + acoustic overlays + data points (SpectrogramWidget)
- `ozen/visualization/data_points.py` - Data collection points model (DataPoint, DataPointCollection)

**Main Application:**
- `ozen/ui/main_window.py` - Main window, menu, save system, widget coordination
- `ozen/__main__.py` - CLI argument parsing, app entry point
- `ozen/config.py` - Configuration loading/merging, DEFAULTS dict

**Offline Rendering:**
- `ozen/render.py` - Headless spectrogram renderer (matplotlib), CLI + Python API

**Audio/Analysis:**
- `ozen/audio/player.py` - Audio playback with sounddevice (timer-based position polling)
- `ozen/audio/loader.py` - Audio file loading
- `ozen/analysis/acoustic.py` - Feature extraction (pitch, formants, intensity, etc.)

## Architecture Notes

- All three display widgets (waveform, spectrogram, annotation editor) use pg.GraphicsLayoutWidget
  - **Important:** Do NOT use pg.PlotWidget - it causes audio noise on macOS during cursor animation
- They synchronize via signals: `time_range_changed`, `cursor_moved`, `selection_changed`
- The main window connects these signals to keep views in sync
- Undo system: each subsystem (annotation editor, data points) has its own undo stack
  - Global undo stack in `main_window.py` (`_global_undo_stack`) tracks operation order across systems
  - Ctrl+Z undoes operations in reverse chronological order regardless of which system
- Save system is in `main_window.py` (`_textgrid_path`, `_is_dirty`, auto-save timer)
- Config is loaded once at import, `reload_config()` updates dict in-place for all modules

## Common Patterns

- `hideButtons()` - Hide pyqtgraph's autorange button
- `setMouseEnabled(x=False, y=False)` - Disable default mouse handling
- `setMouseTracking(True)` - Enable mouse move events without button press
- `setZValue(1000)` - Ensure cursor line is always on top
- `refresh()` on annotation editor - Call after any tier/boundary changes
- Config colors are `[R, G, B, A]` lists, accessed via `config['colors']['key']`

## License

MIT License - Ozen is fully MIT licensed. The migration to praatfan removed the mandatory GPL dependency. Optional GPL backends (like `praat` via parselmouth) can be installed separately but are not required.

## Recent Changes (January 2026)

- **Migrated to praatfan for acoustic analysis:**
  - Replaced direct parselmouth usage with praatfan's unified API
  - Enables backend switching between praatfan (MIT), praatfan-rust (MIT), and parselmouth (GPL)
  - GPL dependency (parselmouth) is now optional - project is MIT by default
  - Added `analysis.acoustic_backend` config option ('auto', 'praatfan', 'praatfan-rust', 'parselmouth')
  - Added Backend dropdown in control panel for runtime backend switching
  - Auto re-extracts features when backend is changed (for files under 60s)
- Fixed pitch extraction and display:
  - Pitch now extracted directly from Praat's Pitch object using array access (more reliable)
  - Log-scale interpolation for pitch resampling (perceptually accurate)
  - Pitch curve breaks at unvoiced frames (`connect='finite'`) - no more straight lines across gaps
  - Removed pitch_floor/ceiling from formant presets - pitch uses global `analysis.default_pitch_floor/ceiling` (75-600 Hz) like Praat
- Changed license from MIT to GPL v3 due to parselmouth dependency (with MIT fallback option for non-GPL acoustic implementations)
- Added data collection points feature:
  - Double-click spectrogram to add points that capture acoustic measurements
  - Points displayed as orange vertical lines with circle markers
  - Drag to move, right-click to remove
  - Export/import via File menu (TSV format)
  - Annotations looked up at export time for accuracy
- Added global undo system (Ctrl+Z / Cmd+Z) that works across annotations and data points
- Added right-click context menu to remove annotation boundaries
- Text editor now finalizes edit on focus loss (not just Enter)
- Added play button to spectrogram selection (green triangle, lower left)
- Changed selection behavior: clicking always starts new selection, use play button or Space to play
- Added `playback.silence_padding` config (default 300ms) to prevent audio cutoff
- Added **File > Load Config...** menu option to load config at runtime
- Pitch display range now configurable (`pitch.display_floor/ceiling`), default changed to 50-400 Hz
- Added Windows installation instructions
- Renamed project from WaveAnnotator to Ozen
- Added config system with YAML/JSON support and `--config` CLI option
- Added `default_tiers` config option for automatic tier creation
- Fixed config `reload_config()` to update dict in-place (so all modules see changes)
- Fixed median filter to use `np.nanmedian()` instead of replacing NaN with 0
- Fixed TextGrid parser with null checks on regex matches
- Fixed escaped quotes in TextGrid (`""` → `"`)
- Removed thread-unsafe audio position callback, using timer polling instead
- Changed `--tiers` to comma-separated format (`-t words,phones`)
- F4 formant now shows bandwidth-based coloring (was missing)
- All formants use consistent size from config

## Testing Notes

- Requires PortAudio system library for audio playback
- X forwarding (`ssh -X`) does NOT forward audio - need local testing or PulseAudio forwarding
- Auto-extracts features for files under 60 seconds

## macOS Audio Noise Issue

On macOS, audio playback can produce noise/static when pyqtgraph widgets are being animated (cursor moving during playback). This was extensively debugged and traced to two causes:

### Cause 1: PlotWidget vs GraphicsLayoutWidget
- `pg.PlotWidget` causes audio interference during cursor animation
- `pg.GraphicsLayoutWidget` does not
- All display widgets must use `GraphicsLayoutWidget` with `self._plot = self.addPlot(row=0, col=0)`

### Cause 2: Waveform line width
- `waveform_line_width` > 1 causes audio noise during cursor animation
- Width of 1 works fine
- The default is set to 1 in config.py; do not increase it
- Root cause unclear - possibly anti-aliasing code paths or event loop contention with CoreAudio
- This is NOT a GPU performance issue (occurs on powerful M1/M2 Macs)

### If noise returns
1. Check that all display widgets inherit from `pg.GraphicsLayoutWidget`, not `pg.PlotWidget`
2. Check `waveform_line_width` in config - must be 1
3. Use diagnostic scripts in `/tmp/audio_test*.py` pattern from debugging session

## Known Issues

### Play Button Visual Mismatch
The spectrogram and annotation play buttons don't render identically despite using the same colors. The annotation editor uses QPainter.drawPolygon in mixed coordinates (seconds × pixels), while the spectrogram uses a custom QGraphicsItem in scene coordinates. The triangles are slightly different sizes/positions. Low priority to fix.

## Related Projects

### praatfan-core-clean
Unified Python API for acoustic analysis with pluggable backends. Repository: https://github.com/ucpresearch/praatfan-core-clean

Provides:
- `praatfan` backend: Pure Python implementation (MIT)
- `praatfan-rust` backend: Rust-accelerated implementation (MIT)
- `parselmouth` backend: Original Praat bindings (GPL)

Ozen uses praatfan as its primary acoustic analysis library, enabling license flexibility and potential future performance improvements.

# pip

Please us `uv pip` rather than `pip`. 
