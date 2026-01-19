# Development Guide

This document provides technical information for developers who want to understand, modify, or contribute to Ozen.

## Architecture Overview

Ozen is built on PyQt6 with pyqtgraph for high-performance visualization. The application follows a signal-based architecture where three main display widgets stay synchronized through Qt signals.

### Main Components

```
┌─────────────────────────────────────────────────────────┐
│                     MainWindow                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │              WaveformWidget                       │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │            SpectrogramWidget                      │  │
│  │    (with acoustic overlay tracks)                 │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │          AnnotationEditorWidget                   │  │
│  │    (multiple TierItems)                           │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │              Control Panel                        │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Signal Flow

All three display widgets synchronize via signals:
- `time_range_changed` - When zoom/pan changes the visible time range
- `cursor_moved` - When the playback cursor position updates
- `selection_changed` - When the user selects a time region

The MainWindow connects these signals between widgets so changes propagate automatically.

## File Structure

### Entry Point
| File | Purpose |
|------|---------|
| `ozen/__main__.py` | CLI argument parsing, application entry point |

### Core Application
| File | Purpose |
|------|---------|
| `ozen/ui/main_window.py` | Main window, menus, toolbar, widget coordination, save system |
| `ozen/config.py` | Configuration loading/merging from YAML/JSON files |

### Audio
| File | Purpose |
|------|---------|
| `ozen/audio/loader.py` | Audio file loading (WAV, FLAC, MP3, OGG via soundfile) |
| `ozen/audio/player.py` | Audio playback using sounddevice, position tracking |

### Analysis
| File | Purpose |
|------|---------|
| `ozen/analysis/acoustic.py` | Parselmouth wrapper for pitch, formants, intensity, HNR, spectral moments |

### Visualization
| File | Purpose |
|------|---------|
| `ozen/visualization/waveform.py` | Waveform display widget |
| `ozen/visualization/spectrogram.py` | Spectrogram display with toggleable acoustic overlay tracks |

### Annotation
| File | Purpose |
|------|---------|
| `ozen/annotation/tier.py` | Data models: `Tier`, `Interval`, `AnnotationSet` |
| `ozen/annotation/editor.py` | Annotation editor widget with `TierItem` components |
| `ozen/annotation/textgrid.py` | Praat TextGrid import/export |

## Design Considerations

### Why PyQt6?

Cross-platform portability is a primary design goal. PyQt6 provides:
- Native look and feel on Windows, macOS, and Linux
- Single codebase that runs identically across platforms
- Mature, well-documented framework with long-term support
- No platform-specific code required for core functionality

Ozen is developed and tested on Linux and macOS, and runs on Windows without modification.

### Why pyqtgraph?

Ozen needs to display and interact with potentially long audio files with smooth zooming and panning. pyqtgraph provides:
- GPU-accelerated rendering via OpenGL
- Efficient handling of large datasets
- Built-in pan/zoom with mouse interaction
- Easy overlay of multiple plot items
- Cross-platform compatibility (builds on PyQt6)

### Why parselmouth?

Parselmouth provides Python bindings to Praat's acoustic analysis algorithms. These are:
- Battle-tested and trusted by phoneticians
- Well-documented in academic literature
- Consistent with the tool most researchers already use

### Configuration System

The config system allows customization without code changes:
- Colors, line widths, sizes for all UI elements
- Formant presets for different voice types
- Analysis parameters
- Default tier names

**Config file locations** (searched in order, first found wins):

| Location | Purpose |
|----------|---------|
| `./ozen.yaml` or `./ozen.json` | Project-specific config |
| `~/.config/ozen/config.yaml` or `.json` | User config (XDG standard) |
| `~/.ozen.yaml` or `~/.ozen.json` | User config (simple) |

Config files are merged with defaults, so users only need to specify values they want to change. Config can also be loaded at runtime via **File > Load Config...** or specified on the command line with `-c path/to/config.yaml`.

### Undo System

The annotation editor maintains an undo stack (`_undo_stack`) that captures:
- Boundary additions/deletions
- Label text changes

Each undo entry stores enough state to reverse the operation. The stack is managed via `_push_undo()` and `undo()` methods in `editor.py`.

### Auto-save

Annotations auto-save every 60 seconds to a `.autosave` file alongside the main TextGrid. This prevents data loss without overwriting the user's intentional saves.

## Known Issues and Workarounds

### macOS Audio Noise

On macOS, audio playback can produce static/noise when pyqtgraph widgets are animating (e.g., cursor moving during playback).

**Known cause: Waveform line width**

Setting `waveform_line_width` greater than 1 causes audio noise during cursor animation. The default is 1, which works correctly. The root cause is unclear but appears related to anti-aliasing or event loop contention with CoreAudio.

**If noise returns:**
Check `waveform_line_width` in config equals 1.

### Audio Playback Cutoff

PortAudio can cut off the last few milliseconds of audio before the buffer fully plays out. To prevent this, the player pads audio with silence (configurable via `playback.silence_padding`, default 300ms). This ensures the actual audio content plays completely before the stream closes.

### Play Button Visual Inconsistency

The play button triangles in the spectrogram selection and annotation tiers don't match perfectly despite using the same colors and similar drawing code. The annotation editor draws directly via QPainter in mixed coordinates (seconds × pixels), while the spectrogram uses a QGraphicsItem positioned in scene coordinates. Further work needed to unify the rendering approach.

## Common Patterns

### pyqtgraph Idioms

```python
# Hide the auto-range "A" button
self._plot.hideButtons()

# Disable default mouse pan/zoom (we handle it ourselves)
self._plot.setMouseEnabled(x=False, y=False)

# Enable mouse tracking without button press
self.setMouseTracking(True)

# Ensure an item renders on top of others
item.setZValue(1000)
```

### Config Access

Colors and settings are accessed from the global config dict:

```python
from ..config import config

# Colors are [R, G, B, A] lists
color = config['colors']['pitch']
pen = pg.mkPen(color=color[:3], width=config['colors']['pitch_width'])

# Reload config at runtime
from ..config import reload_config
reload_config('/path/to/config.yaml')
```

### Annotation Editor Refresh

After modifying tiers or boundaries programmatically, call `refresh()` to update the display:

```python
self._annotation_editor.refresh()
```

## Adding New Features

### Adding an Overlay Track

1. Add color/width settings to `DEFAULTS` in `config.py`
2. In `spectrogram.py`:
   - Add a plot item in `_setup_overlays()`
   - Add visibility tracking in `_track_visibility`
   - Add checkbox in main_window.py control panel
   - Update the track in `_update_overlays()`
3. If the feature needs new analysis, add it to `acoustic.py` and `AcousticFeatures`

### Adding a Config Option

1. Add the default value to `DEFAULTS` in `config.py`
2. Access it via `config['section']['key']` where needed
3. Document it in the README if user-facing

## Testing

- Audio playback requires PortAudio system library
- X11 forwarding (`ssh -X`) does not forward audio; test locally or use PulseAudio forwarding
- Feature extraction auto-runs for files under 60 seconds (configurable)
