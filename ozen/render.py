"""Offline spectrogram renderer for publication-quality figures.

Renders spectrograms with acoustic overlays, TextGrid annotations,
and data collection points without requiring a GUI. Supports PNG,
PDF, SVG, and EPS output formats.

Usage:
    python -m ozen.render AUDIO_FILE -o OUTPUT [options]

Example:
    python -m ozen.render recording.wav -o fig.png --overlays pitch,formants
    python -m ozen.render recording.wav -o fig.pdf --overlays pitch --textgrid recording.TextGrid
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np

from .analysis.acoustic import (
    AcousticFeatures,
    compute_spectrogram,
    extract_features,
    switch_backend,
)
from .annotation.textgrid import read_textgrid
from .config import config, reload_config


# Overlay color defaults (RGB 0-1 tuples), derived from config colors
def _rgba_to_rgb01(rgba: list[int]) -> tuple[float, float, float]:
    """Convert [R,G,B,A] (0-255) to (r,g,b) (0-1)."""
    return (rgba[0] / 255, rgba[1] / 255, rgba[2] / 255)


def _get_overlay_colors() -> dict[str, tuple[float, float, float]]:
    """Get overlay colors from config, converted to matplotlib RGB tuples."""
    c = config['colors']
    return {
        'pitch': _rgba_to_rgb01(c['pitch']),
        'intensity': _rgba_to_rgb01(c['intensity']),
        'formant': _rgba_to_rgb01(c['formant']),
        'formant_wide': _rgba_to_rgb01(c['formant_wide']),
        'cog': _rgba_to_rgb01(c['cog']),
        'hnr': _rgba_to_rgb01(c['hnr']),
        'spectral_tilt': _rgba_to_rgb01(c['spectral_tilt']),
        'a1p0': _rgba_to_rgb01(c['a1p0']),
        'nasal_murmur': _rgba_to_rgb01(c['nasal_murmur']),
    }


# ---- Formant presets (same as spectrogram.py) ----
FORMANT_PRESET_SETTINGS = {
    'male': {'max_formant': 5000, 'num_formants': 5},
    'female': {'max_formant': 5500, 'num_formants': 5},
    'child': {'max_formant': 8000, 'num_formants': 5},
}


def _parse_point_spec(spec: str) -> tuple[str, str]:
    """Parse a --points argument: '[COLOR=]file.tsv'.

    Returns (color, path). Color defaults to 'orange'.
    """
    if '=' in spec:
        color_part, path_part = spec.split('=', 1)
        if mcolors.is_color_like(color_part):
            return color_part, path_part
    return 'orange', spec


def _load_points_tsv(path: str) -> list[dict]:
    """Load data points from a TSV file with 'time' and 'frequency' columns."""
    points = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                t = float(row['time'])
                freq = float(row['frequency'])
                points.append({'time': t, 'frequency': freq})
            except (KeyError, ValueError):
                continue
    return points


# ---- Overlay drawing helpers ----

def _draw_pitch(ax, features: AcousticFeatures, freq_start: float, freq_end: float,
                pitch_floor: float, pitch_ceiling: float, color):
    """Draw pitch overlay with log-scale mapping."""
    freq_range = freq_end - freq_start
    log_p_min = np.log(pitch_floor)
    log_p_max = np.log(pitch_ceiling)
    log_range = log_p_max - log_p_min

    with np.errstate(invalid='ignore'):
        scaled = (np.log(features.f0) - log_p_min) / log_range * freq_range + freq_start
    scaled = np.clip(scaled, freq_start, freq_end)

    # matplotlib naturally breaks lines at NaN
    ax.plot(features.times, scaled, color=color, linewidth=1.5, zorder=5)


def _add_pitch_right_axis(ax, freq_start: float, freq_end: float,
                          pitch_floor: float, pitch_ceiling: float, color):
    """Add a right-side Y-axis with log-spaced pitch tick labels."""
    ax2 = ax.twinx()
    freq_range = freq_end - freq_start
    log_p_min = np.log(pitch_floor)
    log_p_max = np.log(pitch_ceiling)
    log_range = log_p_max - log_p_min

    pitch_ticks = [t for t in [50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 800]
                   if pitch_floor <= t <= pitch_ceiling]

    tick_positions = []
    tick_labels = []
    for p in pitch_ticks:
        pos = (np.log(p) - log_p_min) / log_range * freq_range + freq_start
        tick_positions.append(pos)
        tick_labels.append(str(int(p)))

    ax2.set_ylim(freq_start, freq_end)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels)
    ax2.set_ylabel('Pitch (Hz)', color=color)
    ax2.tick_params(axis='y', colors=color)
    return ax2


def _draw_intensity(ax, features: AcousticFeatures, freq_start: float, freq_end: float, color):
    """Draw intensity overlay."""
    freq_range = freq_end - freq_start
    valid = ~np.isnan(features.intensity)
    scaled = (features.intensity - 30) / 60 * freq_range + freq_start
    scaled = np.clip(scaled, freq_start, freq_end)
    ax.plot(features.times[valid], scaled[valid], color=color, linewidth=1.5, zorder=5)


def _draw_formants(ax, features: AcousticFeatures, colors):
    """Draw formant scatter with bandwidth-based color."""
    bw_min, bw_max = 50, 400
    formant_color = colors['formant']
    wide_color = colors['formant_wide']

    for fkey, bkey in [('F1', 'B1'), ('F2', 'B2'), ('F3', 'B3'), ('F4', 'B4')]:
        vals = features.formants[fkey]
        valid = ~np.isnan(vals)
        if not np.any(valid):
            continue

        times_v = features.times[valid]
        freqs_v = vals[valid]
        bw_vals = features.bandwidths[bkey][valid]
        bw_vals = np.nan_to_num(bw_vals, nan=bw_max)

        # Interpolate color from formant (red) to formant_wide (pink) based on bandwidth
        t = np.clip((bw_vals - bw_min) / (bw_max - bw_min), 0, 1)
        point_colors = np.zeros((len(t), 3))
        for ch in range(3):
            point_colors[:, ch] = formant_color[ch] + t * (wide_color[ch] - formant_color[ch])

        ax.scatter(times_v, freqs_v, c=point_colors, s=6, zorder=6, linewidths=0)


def _draw_cog(ax, features: AcousticFeatures, color):
    """Draw center of gravity overlay."""
    valid = ~np.isnan(features.cog)
    ax.plot(features.times[valid], features.cog[valid], color=color, linewidth=1.5, zorder=5)


def _draw_hnr(ax, features: AcousticFeatures, freq_start: float, freq_end: float, color):
    """Draw HNR overlay (dashed)."""
    freq_range = freq_end - freq_start
    valid = ~np.isnan(features.hnr)
    scaled = (features.hnr + 10) / 50 * freq_range + freq_start
    scaled = np.clip(scaled, freq_start, freq_end)
    ax.plot(features.times[valid], scaled[valid], color=color, linewidth=1.5,
            linestyle='--', zorder=5)


def _draw_spectral_tilt(ax, features: AcousticFeatures, freq_start: float, freq_end: float, color):
    """Draw spectral tilt overlay."""
    freq_range = freq_end - freq_start
    valid = ~np.isnan(features.spectral_tilt)
    scaled = (features.spectral_tilt + 20) / 60 * freq_range + freq_start
    scaled = np.clip(scaled, freq_start, freq_end)
    ax.plot(features.times[valid], scaled[valid], color=color, linewidth=1.5, zorder=5)


def _draw_a1p0(ax, features: AcousticFeatures, freq_start: float, freq_end: float, color):
    """Draw A1-P0 nasal ratio overlay."""
    freq_range = freq_end - freq_start
    valid = ~np.isnan(features.nasal_ratio)
    if not np.any(valid):
        return
    scaled = (features.nasal_ratio + 20) / 40 * freq_range + freq_start
    scaled = np.clip(scaled, freq_start, freq_end)
    ax.plot(features.times[valid], scaled[valid], color=color, linewidth=1.5, zorder=5)


def _draw_nasal_murmur(ax, features: AcousticFeatures, freq_start: float, freq_end: float, color):
    """Draw nasal murmur ratio overlay (dashed)."""
    freq_range = freq_end - freq_start
    valid = ~np.isnan(features.nasal_murmur_ratio)
    if not np.any(valid):
        return
    scaled = features.nasal_murmur_ratio * freq_range + freq_start
    scaled = np.clip(scaled, freq_start, freq_end)
    ax.plot(features.times[valid], scaled[valid], color=color, linewidth=1.5,
            linestyle='--', zorder=5)


def _draw_annotation_tier(ax, tier, start_time: float, end_time: float):
    """Draw a single annotation tier in a subplot."""
    ax.set_xlim(start_time, end_time)
    ax.set_ylim(0, 1)
    ax.set_ylabel(tier.name, rotation=0, ha='right', va='center', fontsize=9)
    ax.set_yticks([])

    # Draw intervals
    for interval in tier.get_intervals():
        # Skip intervals outside view
        if interval.end <= start_time or interval.start >= end_time:
            continue

        # Clamp to view
        x_start = max(interval.start, start_time)
        x_end = min(interval.end, end_time)

        # Draw boundary lines
        if interval.start > start_time:
            ax.axvline(interval.start, color='#0064C8', linewidth=1, zorder=3)

        # Draw text label centered
        if interval.text:
            mid = (x_start + x_end) / 2
            ax.text(mid, 0.5, interval.text, ha='center', va='center',
                    fontsize=8, clip_on=True, zorder=4)

    # Draw right edge boundary
    if tier.end_time < end_time:
        ax.axvline(tier.end_time, color='#0064C8', linewidth=1, zorder=3)

    # Light background
    ax.set_facecolor('#F0F0F0')
    ax.tick_params(axis='x', labelbottom=False)


def _get_overlay_legend_info(overlay: str, pitch_floor: float, pitch_ceiling: float) -> tuple[str, str]:
    """Return (display_name, range_string) for an overlay."""
    info = {
        'pitch': ('Pitch (F0)', f'{pitch_floor:.0f}\u2013{pitch_ceiling:.0f} Hz, log'),
        'formants': ('Formants (F1\u2013F4)', 'Hz'),
        'intensity': ('Intensity', '30\u201390 dB'),
        'cog': ('Center of Gravity', 'Hz'),
        'hnr': ('HNR', '\u221210\u201340 dB'),
        'spectral_tilt': ('Spectral Tilt', '\u221220\u2013+40 dB'),
        'a1p0': ('A1\u2013P0 Nasal Ratio', '\u221220\u2013+20 dB'),
        'nasal_murmur': ('Nasal Murmur Ratio', '0\u20131'),
    }
    return info.get(overlay, (overlay, ''))


def _draw_legend(ax, overlays: list[str], overlay_colors: dict,
                 pitch_floor: float, pitch_ceiling: float):
    """Draw a legend box showing active overlays with colors and ranges."""
    from matplotlib.lines import Line2D

    handles = []
    labels = []

    for ov in overlays:
        name, range_str = _get_overlay_legend_info(ov, pitch_floor, pitch_ceiling)
        label = f'{name}  [{range_str}]'

        if ov == 'formants':
            handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=overlay_colors['formant'],
                            markersize=5, linestyle='None')
        elif ov in ('hnr', 'nasal_murmur'):
            handle = Line2D([0], [0], color=overlay_colors.get(ov, 'gray'), linewidth=1.5, linestyle='--')
        else:
            handle = Line2D([0], [0], color=overlay_colors.get(ov, 'gray'), linewidth=1.5)

        handles.append(handle)
        labels.append(label)

    if handles:
        ax.legend(handles, labels, loc='upper right', fontsize=7,
                  framealpha=0.85, edgecolor='gray', fancybox=False)


def _draw_data_points(ax, point_sets: list[tuple[str, list[dict]]], freq_start: float, freq_end: float):
    """Draw data collection point sets on the spectrogram axis."""
    for color, points in point_sets:
        for pt in points:
            ax.axvline(pt['time'], color=color, linewidth=1, alpha=0.7, zorder=7)
            ax.scatter([pt['time']], [pt['frequency']], color=color,
                       s=30, zorder=8, edgecolors='white', linewidths=0.5)


# ---- Main rendering function ----

def render_spectrogram(
    audio_path: str | Path,
    output_path: str | Path,
    start: float = 0.0,
    end: Optional[float] = None,
    backend: str = 'auto',
    overlays: Optional[list[str]] = None,
    preset: str = 'female',
    textgrid_path: Optional[str | Path] = None,
    tiers: Optional[list[str]] = None,
    point_sets: Optional[list[tuple[str, str]]] = None,
    max_freq: float = 5000.0,
    dynamic_range: float = 70.0,
    bandwidth: str = 'narrowband',
    colormap: str = 'Greys',
    pitch_floor: float = 50.0,
    pitch_ceiling: float = 400.0,
    width: float = 10.0,
    height: Optional[float] = None,
    dpi: int = 300,
    title: Optional[str] = None,
    legend: bool = False,
    config_path: Optional[str | Path] = None,
):
    """Render a publication-quality spectrogram image.

    This is the public API, importable from scripts:
        from ozen.render import render_spectrogram
        render_spectrogram('audio.wav', 'fig.png', overlays=['pitch', 'formants'])

    Args:
        audio_path: Path to audio file.
        output_path: Output image path (.png, .pdf, .svg, .eps).
        start: Start time in seconds.
        end: End time in seconds (None = full duration).
        backend: Acoustic backend name.
        overlays: List of overlay names to draw.
        preset: Formant preset ('female', 'male', 'child').
        textgrid_path: Path to TextGrid file.
        tiers: Tier names to display (None = all).
        point_sets: List of (color, tsv_path) tuples for data points.
        max_freq: Maximum frequency in Hz.
        dynamic_range: Dynamic range in dB.
        bandwidth: 'narrowband' or 'wideband'.
        colormap: Matplotlib colormap name or 'grayscale'.
        pitch_floor: Pitch display floor in Hz.
        pitch_ceiling: Pitch display ceiling in Hz.
        width: Figure width in inches.
        height: Figure height in inches (auto if None).
        dpi: DPI for raster output.
        title: Optional figure title.
        legend: If True, draw a legend showing overlay names, colors, and ranges.
        config_path: Path to ozen config YAML.
    """
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    overlays = overlays or []
    point_sets_raw = point_sets or []

    # 1. Setup: load config, switch backend
    if config_path:
        reload_config(config_path)

    if backend != 'auto':
        if not switch_backend(backend):
            print(f"Warning: Backend '{backend}' not available, using auto.", file=sys.stderr)

    # 2. Get audio duration
    import soundfile as sf
    info = sf.info(str(audio_path))
    duration = info.frames / info.samplerate

    if end is None or end > duration:
        end = duration
    if start < 0:
        start = 0.0
    if start >= end:
        print(f"Error: start ({start}) must be less than end ({end})", file=sys.stderr)
        sys.exit(1)

    # 3. Compute spectrogram
    window_length = 0.005 if bandwidth == 'wideband' else 0.025
    spec_times, spec_freqs, spec_db = compute_spectrogram(
        audio_path,
        window_length=window_length,
        max_frequency=max_freq,
        dynamic_range=dynamic_range,
        start_time=start,
        end_time=end,
    )

    freq_start = spec_freqs[0] if len(spec_freqs) > 0 else 0.0
    freq_end = spec_freqs[-1] if len(spec_freqs) > 0 else max_freq

    # 4. Extract acoustic features if overlays requested
    features = None
    if overlays:
        preset_settings = FORMANT_PRESET_SETTINGS.get(preset, FORMANT_PRESET_SETTINGS['female'])
        pitch_extract_floor = float(config['analysis']['default_pitch_floor'])
        pitch_extract_ceiling = float(config['analysis']['default_pitch_ceiling'])

        features = extract_features(
            audio_path,
            pitch_floor=pitch_extract_floor,
            pitch_ceiling=pitch_extract_ceiling,
            max_formant=preset_settings['max_formant'],
            start_time=start,
            end_time=end,
        )

    # 5. Load optional data
    annotations = None
    display_tiers = []
    if textgrid_path:
        annotations = read_textgrid(textgrid_path)
        all_tiers = annotations.get_tiers()
        if tiers:
            tier_set = set(tiers)
            display_tiers = [t for t in all_tiers if t.name in tier_set]
        else:
            display_tiers = all_tiers

    loaded_point_sets = []
    for color_spec, path_spec in point_sets_raw:
        pts = _load_points_tsv(path_spec)
        if pts:
            loaded_point_sets.append((color_spec, pts))

    # 6. Build matplotlib figure
    n_tiers = len(display_tiers)
    tier_height_ratio = 0.15  # each tier is 15% of spectrogram height

    if height is None:
        spec_height = width * 0.4  # default aspect ratio
        total_height = spec_height * (1 + n_tiers * tier_height_ratio)
    else:
        total_height = height

    # Create GridSpec: spectrogram on top, one row per tier below
    height_ratios = [1.0] + [tier_height_ratio] * n_tiers
    fig = plt.figure(figsize=(width, total_height))
    gs = gridspec.GridSpec(
        1 + n_tiers, 1,
        height_ratios=height_ratios,
        hspace=0.05,
    )

    # -- Spectrogram axis --
    ax_spec = fig.add_subplot(gs[0])

    # Normalize spectrogram to 0-1
    data_min = np.min(spec_db)
    data_max = np.max(spec_db)
    normalized = (spec_db - data_min) / (data_max - data_min + 1e-10)

    # Determine colormap
    if colormap.lower() == 'grayscale' or colormap == 'Greys':
        cmap_obj = 'Greys'
    else:
        cmap_obj = colormap

    # imshow expects (rows=freq, cols=time) with origin='lower'
    extent = [spec_times[0], spec_times[-1], freq_start, freq_end]
    ax_spec.imshow(
        normalized,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap=cmap_obj,
        interpolation='bilinear',
    )

    ax_spec.set_xlim(start, end)
    ax_spec.set_ylim(freq_start, freq_end)
    ax_spec.set_ylabel('Frequency (Hz)')

    if n_tiers == 0:
        ax_spec.set_xlabel('Time (s)')
    else:
        ax_spec.tick_params(axis='x', labelbottom=False)

    overlay_colors = _get_overlay_colors()

    # Draw overlays
    if features is not None:
        for ov in overlays:
            if ov == 'pitch':
                _draw_pitch(ax_spec, features, freq_start, freq_end,
                            pitch_floor, pitch_ceiling, overlay_colors['pitch'])
            elif ov == 'intensity':
                _draw_intensity(ax_spec, features, freq_start, freq_end,
                                overlay_colors['intensity'])
            elif ov == 'formants':
                _draw_formants(ax_spec, features, overlay_colors)
            elif ov == 'cog':
                _draw_cog(ax_spec, features, overlay_colors['cog'])
            elif ov == 'hnr':
                _draw_hnr(ax_spec, features, freq_start, freq_end,
                          overlay_colors['hnr'])
            elif ov == 'spectral_tilt':
                _draw_spectral_tilt(ax_spec, features, freq_start, freq_end,
                                    overlay_colors['spectral_tilt'])
            elif ov == 'a1p0':
                _draw_a1p0(ax_spec, features, freq_start, freq_end,
                           overlay_colors['a1p0'])
            elif ov == 'nasal_murmur':
                _draw_nasal_murmur(ax_spec, features, freq_start, freq_end,
                                   overlay_colors['nasal_murmur'])

        # Add pitch right axis if pitch overlay is active
        if 'pitch' in overlays:
            _add_pitch_right_axis(ax_spec, freq_start, freq_end,
                                  pitch_floor, pitch_ceiling, overlay_colors['pitch'])

    # Draw legend
    if legend and overlays:
        _draw_legend(ax_spec, overlays, overlay_colors, pitch_floor, pitch_ceiling)

    # Draw data points
    if loaded_point_sets:
        _draw_data_points(ax_spec, loaded_point_sets, freq_start, freq_end)

    # -- Annotation tier axes --
    for i, tier in enumerate(display_tiers):
        ax_tier = fig.add_subplot(gs[1 + i], sharex=ax_spec)
        _draw_annotation_tier(ax_tier, tier, start, end)
        if i == n_tiers - 1:
            ax_tier.set_xlabel('Time (s)')
            ax_tier.tick_params(axis='x', labelbottom=True)

    # Title
    if title:
        fig.suptitle(title, fontsize=12)

    # 7. Save
    fig.savefig(str(output_path), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---- CLI ----

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog='python -m ozen.render',
        description='Render publication-quality spectrograms with acoustic overlays.',
    )
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('-o', '--output', required=True, help='Output image path (.png, .pdf, .svg, .eps)')
    parser.add_argument('--start', type=float, default=0.0, help='Start time in seconds (default: 0)')
    parser.add_argument('--end', type=float, default=None, help='End time in seconds (default: full duration)')
    parser.add_argument('--backend', default='auto',
                        help='Acoustic backend (auto, praatfan, praatfan-rust, praatfan-gpl, praat)')
    parser.add_argument('--overlays', default='',
                        help='Comma-separated overlays: '
                             'pitch (F0), '
                             'formants (F1-F4), '
                             'intensity (dB), '
                             'cog (spectral centroid), '
                             'hnr (harmonics-to-noise), '
                             'spectral_tilt (low vs high energy), '
                             'a1p0 (nasal ratio), '
                             'nasal_murmur (low-freq energy ratio)')
    parser.add_argument('--preset', default='female', choices=['female', 'male', 'child'],
                        help='Formant preset (default: female)')
    parser.add_argument('--textgrid', default=None, help='Path to TextGrid file')
    parser.add_argument('--tiers', default=None, help='Comma-separated tier names to display')
    parser.add_argument('--points', action='append', default=None,
                        help='[COLOR=]file.tsv - data point file (repeatable)')
    parser.add_argument('--max-freq', type=float, default=5000.0, help='Max frequency in Hz (default: 5000)')
    parser.add_argument('--dynamic-range', type=float, default=70.0, help='Dynamic range in dB (default: 70)')
    parser.add_argument('--bandwidth', default='narrowband', choices=['narrowband', 'wideband'],
                        help='Bandwidth: narrowband (25ms) or wideband (5ms)')
    parser.add_argument('--colormap', default='Greys',
                        help='Colormap: grayscale, viridis, inferno, or any matplotlib name (default: Greys)')
    parser.add_argument('--pitch-floor', type=float, default=50.0, help='Pitch display floor in Hz (default: 50)')
    parser.add_argument('--pitch-ceiling', type=float, default=400.0, help='Pitch display ceiling in Hz (default: 400)')
    parser.add_argument('--width', type=float, default=10.0, help='Figure width in inches (default: 10)')
    parser.add_argument('--height', type=float, default=None, help='Figure height in inches (default: auto)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for raster output (default: 300)')
    parser.add_argument('--title', default=None, help='Figure title')
    parser.add_argument('--legend', action='store_true', help='Show legend with overlay names, colors, and ranges')
    parser.add_argument('-c', '--config', default=None, help='Path to ozen config YAML')
    parser.add_argument('--man', action='store_true', help='Show full manual page')

    return parser.parse_args(argv)


MAN_PAGE = """\
OZEN-RENDER(1)                  Ozen Manual                  OZEN-RENDER(1)

NAME
    ozen-render - render publication-quality spectrograms with acoustic
    overlays, annotations, and data points

SYNOPSIS
    python -m ozen.render AUDIO_FILE -o OUTPUT [OPTIONS]

DESCRIPTION
    Generates spectrogram images suitable for papers and presentations.
    Supports PNG, PDF, SVG, and EPS output. Can overlay acoustic
    measurements, TextGrid annotations, and data collection points.

    Runs headless (no GUI required), making it suitable for batch
    processing and scripting.

AUDIO & TIME
    AUDIO_FILE              Input audio file (WAV, FLAC, OGG, etc.)
    -o, --output PATH       Output image path (.png, .pdf, .svg, .eps)
    --start SECONDS         Start time (default: 0)
    --end SECONDS           End time (default: full duration)

SPECTROGRAM
    --max-freq HZ           Maximum frequency (default: 5000)
    --dynamic-range DB      Dynamic range in dB (default: 70)
    --bandwidth TYPE        narrowband (25ms window) or wideband (5ms)
                            (default: narrowband)
    --colormap NAME         grayscale, viridis, inferno, or any matplotlib
                            colormap name (default: Greys)

ACOUSTIC OVERLAYS
    --overlays LIST         Comma-separated list of overlays to draw.
    --backend NAME          Acoustic backend: auto, praatfan, praatfan-rust,
                            praatfan-gpl, praat (default: auto)
    --preset TYPE           Formant preset: female, male, child
                            (affects max_formant for formant analysis)

    Available overlays and their display ranges:

    Name            Description                     Range
    ──────────────  ──────────────────────────────  ─────────────────
    pitch           Fundamental frequency (F0)      pitch-floor to
                                                    pitch-ceiling Hz
                                                    (log scale)
    formants        Formant frequencies F1–F4       direct Hz
                    (color: red=narrow bandwidth,   (scatter plot)
                    pink=wide bandwidth)
    intensity       Sound pressure level            30–90 dB
    cog             Center of gravity (spectral     direct Hz
                    centroid)
    hnr             Harmonics-to-noise ratio        −10–40 dB
                    (dashed line)
    spectral_tilt   Low vs high frequency energy    −20–+40 dB
    a1p0            A1–P0 nasal ratio (requires     −20–+20 dB
                    voicing detection)
    nasal_murmur    Low-frequency energy ratio      0–1
                    (dashed line)

    Overlays are scaled to fit the spectrogram frequency axis. Pitch
    uses a logarithmic scale; other scaled overlays use linear mapping.
    Formants and CoG are plotted at their actual frequency values.

PITCH DISPLAY
    --pitch-floor HZ        Pitch display minimum (default: 50)
    --pitch-ceiling HZ      Pitch display maximum (default: 400)

    When the pitch overlay is active, a right Y-axis is added with
    log-spaced tick labels showing pitch values in Hz.

    Note: pitch-floor/ceiling control the *display* range. The
    *extraction* range is set by the config (analysis.default_pitch_floor
    and analysis.default_pitch_ceiling, default 75–600 Hz).

ANNOTATIONS
    --textgrid PATH         Path to Praat TextGrid file
    --tiers LIST            Comma-separated tier names to display
                            (default: all tiers in the TextGrid)

    Each tier is rendered as a labeled row below the spectrogram with
    boundary lines and centered text labels.

DATA POINTS
    --points [COLOR=]FILE   Data point TSV file. Repeatable.

    The TSV file must have 'time' and 'frequency' column headers.
    Points are drawn as vertical lines with circle markers.

    Colors can be any matplotlib color name or hex code:
        --points red=vowels.tsv
        --points "#4488CC"=stops.tsv
        --points nasals.tsv              (default: orange)

    Multiple point sets can be overlaid:
        --points red=vowels.tsv --points blue=stops.tsv

FIGURE
    --width INCHES          Figure width (default: 10)
    --height INCHES         Figure height (default: auto)
    --dpi NUMBER            DPI for raster formats (default: 300)
    --title TEXT            Figure title
    --legend                Add legend showing overlay names, colors,
                            and value ranges

CONFIGURATION
    -c, --config PATH       Path to Ozen config YAML file. Overrides
                            default config search (./ozen.yaml,
                            ~/.config/ozen/config.yaml, ~/.ozen.yaml).

    Colors for overlays are read from the config file's 'colors'
    section. The config also controls formant filter thresholds,
    pitch extraction range, and other analysis parameters.

EXAMPLES
    Basic spectrogram:
        python -m ozen.render speech.wav -o fig.png

    With pitch and formants:
        python -m ozen.render speech.wav -o fig.png \\
            --overlays pitch,formants --legend

    Windowed view with annotations:
        python -m ozen.render speech.wav -o fig.pdf \\
            --start 0.5 --end 2.0 \\
            --overlays pitch,formants,intensity \\
            --textgrid speech.TextGrid --tiers words,phones

    Male voice, wideband, custom colormap:
        python -m ozen.render speech.wav -o fig.png \\
            --preset male --bandwidth wideband \\
            --overlays pitch,formants --colormap inferno

    Multiple colored point sets:
        python -m ozen.render speech.wav -o fig.png \\
            --overlays pitch,formants \\
            --points red=vowels.tsv --points blue=stops.tsv

    All overlays with legend and title:
        python -m ozen.render speech.wav -o fig.png \\
            --overlays pitch,formants,intensity,cog,hnr,spectral_tilt,a1p0,nasal_murmur \\
            --legend --title "Acoustic Analysis"

PYTHON API
    The renderer can also be used as a library:

        from ozen.render import render_spectrogram

        render_spectrogram(
            'speech.wav', 'fig.png',
            overlays=['pitch', 'formants'],
            textgrid_path='speech.TextGrid',
            legend=True,
        )

SEE ALSO
    python -m ozen          Launch the Ozen GUI
    python -m ozen --help   GUI command-line options
"""


def main(argv: Optional[list[str]] = None):
    """CLI entry point."""
    # Handle --man before argparse (which requires positional args)
    check_argv = argv if argv is not None else sys.argv[1:]
    if '--man' in check_argv:
        print(MAN_PAGE)
        return

    args = parse_args(argv)

    overlays = [o.strip() for o in args.overlays.split(',') if o.strip()] if args.overlays else []
    tiers_list = [t.strip() for t in args.tiers.split(',') if t.strip()] if args.tiers else None

    # Parse point specs
    point_sets = []
    if args.points:
        for spec in args.points:
            color, path = _parse_point_spec(spec)
            point_sets.append((color, path))

    render_spectrogram(
        audio_path=args.audio_file,
        output_path=args.output,
        start=args.start,
        end=args.end,
        backend=args.backend,
        overlays=overlays,
        preset=args.preset,
        textgrid_path=args.textgrid,
        tiers=tiers_list,
        point_sets=point_sets if point_sets else None,
        max_freq=args.max_freq,
        dynamic_range=args.dynamic_range,
        bandwidth=args.bandwidth,
        colormap=args.colormap,
        pitch_floor=args.pitch_floor,
        pitch_ceiling=args.pitch_ceiling,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        title=args.title,
        legend=args.legend,
        config_path=args.config,
    )


if __name__ == '__main__':
    main()
