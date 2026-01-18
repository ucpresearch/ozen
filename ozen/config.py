"""
Configuration management for Ozen.

This module provides centralized configuration with sensible defaults
that can be overridden by a user config file. The config file is loaded
from (in order of priority):
    1. ./ozen.yaml (current directory)
    2. ~/.config/ozen/config.yaml
    3. ~/.ozen.yaml

All settings have defaults, so no config file is required.

Usage:
    from ozen.config import config

    # Access settings
    color = config['colors']['cursor']
    preset = config['formant_presets']['female']
"""

import os
from pathlib import Path
from typing import Any
import copy

# Try to import yaml, fall back to json if not available
try:
    import yaml
    HAS_YAML = True

    # Custom representer for compact list output (colors, short lists)
    def _represent_list(dumper, data):
        """Represent short lists of numbers in flow style [1, 2, 3, 4]."""
        # Use flow style for short lists of numbers (like RGBA colors)
        if len(data) <= 5 and all(isinstance(x, (int, float)) for x in data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

    # Register the custom representer
    yaml.add_representer(list, _represent_list)

except ImportError:
    HAS_YAML = False
    import json


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULTS = {
    # -------------------------------------------------------------------------
    # Color settings (RGBA tuples: red, green, blue, alpha 0-255)
    # -------------------------------------------------------------------------
    'colors': {
        # Waveform widget
        'waveform_background': [255, 255, 255, 255],  # White
        'waveform_line': [0, 0, 0, 255],  # Black
        'waveform_line_width': 2,

        # Cursor (playback position indicator)
        'cursor': [200, 0, 0, 255],  # Dark red
        'cursor_width': 2,

        # Selection region
        'selection_fill': [180, 180, 255, 100],  # Light blue, semi-transparent
        'selection_border': [80, 80, 180, 255],  # Darker blue
        'selection_border_width': 2,

        # Spectrogram overlays
        'pitch': [0, 100, 255, 255],  # Blue
        'pitch_width': 2,
        'intensity': [255, 200, 0, 255],  # Yellow
        'intensity_width': 2,
        'formant': [255, 0, 0, 255],  # Red (narrow bandwidth)
        'formant_wide': [255, 150, 150, 255],  # Pink (wide bandwidth)
        'formant_size': 8,
        'cog': [0, 200, 0, 255],  # Green
        'cog_width': 2,
        'hnr': [180, 0, 220, 255],  # Purple
        'hnr_width': 3,
        'spectral_tilt': [255, 140, 0, 255],  # Orange
        'spectral_tilt_width': 3,
        'a1p0': [0, 200, 200, 255],  # Cyan
        'a1p0_width': 3,
        'nasal_murmur': [160, 82, 45, 255],  # Brown/sienna
        'nasal_murmur_width': 3,

        # Annotation editor
        'tier_background': [240, 240, 240, 255],  # Light gray
        'tier_border': [100, 100, 100, 255],  # Gray
        'tier_selected': [200, 220, 255, 255],  # Light blue
        'boundary': [0, 100, 200, 255],  # Blue
        'boundary_hover': [255, 150, 0, 255],  # Orange (when hovering)
        'boundary_width': 2,
        'interval_text': [0, 0, 0, 255],  # Black
        'play_button': [0, 120, 0, 230],  # Dark green
        'play_button_hover': [0, 180, 0, 255],  # Bright green
    },

    # -------------------------------------------------------------------------
    # Formant extraction presets
    # -------------------------------------------------------------------------
    'formant_presets': {
        'female': {
            'max_formant': 5500,  # Maximum formant frequency (Hz)
            'num_formants': 5,    # Number of formants to track
            'pitch_floor': 100,   # Minimum F0 (Hz)
            'pitch_ceiling': 500, # Maximum F0 (Hz)
        },
        'male': {
            'max_formant': 5000,
            'num_formants': 5,
            'pitch_floor': 75,
            'pitch_ceiling': 300,
        },
        'child': {
            'max_formant': 8000,
            'num_formants': 5,
            'pitch_floor': 150,
            'pitch_ceiling': 600,
        },
    },

    # -------------------------------------------------------------------------
    # Formant filtering thresholds
    # -------------------------------------------------------------------------
    'formant_filters': {
        'F1': {'min_freq': 150, 'max_freq': 1200, 'max_bandwidth': 600},
        'F2': {'min_freq': 500, 'max_freq': 3500, 'max_bandwidth': 700},
        'F3': {'min_freq': 1400, 'max_freq': 4500, 'max_bandwidth': 800},
        'F4': {'min_freq': 2500, 'max_freq': 6000, 'max_bandwidth': 900},
    },

    # -------------------------------------------------------------------------
    # Spectrogram settings
    # -------------------------------------------------------------------------
    'spectrogram': {
        'default_colormap': 'grayscale',  # grayscale, viridis, inferno
        'default_bandwidth': 'Narrowband',  # Narrowband or Wideband
        'narrowband_window': 0.025,  # Window length in seconds (25ms)
        'wideband_window': 0.005,    # Window length in seconds (5ms)
        'dynamic_range': 70.0,       # dB range for display
        'default_max_frequency': 5000,  # Default max frequency (Hz)
        'max_frequency_options': [5000, 7500, 10000],  # Available options
    },

    # -------------------------------------------------------------------------
    # Pitch display settings
    # -------------------------------------------------------------------------
    'pitch': {
        'display_floor': 50,    # Minimum displayed pitch (Hz)
        'display_ceiling': 800, # Maximum displayed pitch (Hz)
    },

    # -------------------------------------------------------------------------
    # Analysis settings
    # -------------------------------------------------------------------------
    'analysis': {
        'time_step': 0.01,           # Analysis time step (seconds)
        'default_pitch_floor': 75,   # Default minimum F0 (Hz)
        'default_pitch_ceiling': 600, # Default maximum F0 (Hz)
        'hnr_silence_threshold': 0.1, # HNR silence threshold
        'auto_extract_max_duration': 60.0,  # Auto-extract for files under this duration (seconds)
    },

    # -------------------------------------------------------------------------
    # Nasal ratio settings
    # -------------------------------------------------------------------------
    'nasal_ratio': {
        'a1_bandwidth_factor': 0.1,  # A1 band is F0 ± (F0 * factor)
        'p0_low': 200,   # P0 (nasal pole) region lower bound (Hz)
        'p0_high': 300,  # P0 region upper bound (Hz)
        'murmur_low': 0,     # Nasal murmur low band (Hz)
        'murmur_high': 500,  # Nasal murmur high band (Hz)
        'murmur_total_high': 5000,  # Total energy upper bound (Hz)
    },

    # -------------------------------------------------------------------------
    # Spectral tilt settings
    # -------------------------------------------------------------------------
    'spectral_tilt': {
        'low_band_start': 0,
        'low_band_end': 500,
        'high_band_start': 2000,
        'high_band_end': 4000,
    },

    # -------------------------------------------------------------------------
    # Annotation settings
    # -------------------------------------------------------------------------
    'annotation': {
        'default_tiers': [],  # Tier names to create automatically (e.g., ['words', 'phones'])
        'max_tiers': 5,
        'tier_height': 60,  # Approximate height in pixels
        'boundary_snap_threshold': 0.015,  # Snap to boundary within 15ms
        'click_threshold': 0.005,  # Clicks shorter than 5ms are not drags
        'autosave_interval': 60000,  # Autosave interval in milliseconds
    },

    # -------------------------------------------------------------------------
    # Display settings
    # -------------------------------------------------------------------------
    'display': {
        'waveform_max_points': 100000,  # Downsample waveform if longer
        'axis_width': 70,  # Width of Y-axis labels
    },
}


# =============================================================================
# CONFIG LOADING
# =============================================================================

def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries.

    Values from 'override' take precedence over 'base'.
    Nested dicts are merged recursively.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _find_config_file() -> Path | None:
    """Find the user's config file, if it exists."""
    candidates = [
        Path('./ozen.yaml'),
        Path('./ozen.json'),
        Path.home() / '.config' / 'ozen' / 'config.yaml',
        Path.home() / '.config' / 'ozen' / 'config.json',
        Path.home() / '.ozen.yaml',
        Path.home() / '.ozen.json',
    ]

    for path in candidates:
        if path.exists():
            return path
    return None


def _load_config_file(path: Path) -> dict:
    """Load configuration from a file."""
    with open(path, 'r', encoding='utf-8') as f:
        if path.suffix in ('.yaml', '.yml'):
            if HAS_YAML:
                return yaml.safe_load(f) or {}
            else:
                raise ImportError(
                    f"PyYAML is required to load {path}. "
                    "Install it with: pip install pyyaml"
                )
        else:
            return json.load(f)


def load_config(config_path: Path | str | None = None) -> dict:
    """
    Load configuration with user overrides.

    Args:
        config_path: Optional explicit path to config file. If provided,
                     this file will be loaded instead of searching default locations.

    Returns a dict with all settings, using defaults for any
    values not specified in the user's config file.
    """
    config = copy.deepcopy(DEFAULTS)

    # Use explicit path if provided, otherwise search default locations
    if config_path:
        config_file = Path(config_path)
        if not config_file.exists():
            print(f"Warning: Config file not found: {config_file}")
            return config
    else:
        config_file = _find_config_file()

    if config_file:
        try:
            user_config = _load_config_file(config_file)
            config = _deep_merge(config, user_config)
            print(f"Loaded config from: {config_file}")
        except Exception as e:
            print(f"Warning: Failed to load config from {config_file}: {e}")

    return config


def save_default_config(path: Path | str):
    """
    Save the default configuration to a file.

    Useful for creating a template config file that users can edit.
    """
    path = Path(path)

    with open(path, 'w', encoding='utf-8') as f:
        if path.suffix in ('.yaml', '.yml'):
            if HAS_YAML:
                yaml.dump(DEFAULTS, f, default_flow_style=False, sort_keys=False)
            else:
                raise ImportError("PyYAML is required to save YAML config")
        else:
            json.dump(DEFAULTS, f, indent=2)


# =============================================================================
# GLOBAL CONFIG INSTANCE
# =============================================================================

# Load config on module import
config = load_config()


def reload_config(config_path: Path | str | None = None):
    """
    Reload configuration from file.

    Args:
        config_path: Optional explicit path to config file. If provided,
                     this file will be loaded instead of searching default locations.
    """
    global config
    config = load_config(config_path)


def load_config_from_path(path: Path | str) -> dict:
    """
    Load configuration from a specific file path.

    Args:
        path: Path to the config file (YAML or JSON)

    Returns:
        Merged config dict with defaults

    Raises:
        FileNotFoundError: If the config file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    base_config = copy.deepcopy(DEFAULTS)
    user_config = _load_config_file(path)
    return _deep_merge(base_config, user_config)
