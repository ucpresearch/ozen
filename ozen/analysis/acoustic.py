"""
Acoustic analysis using praatfan (with pluggable backends) and scipy.

This module provides acoustic feature extraction for speech analysis.
It uses praatfan for core acoustic measurements, which supports multiple
backends including pure Python (MIT), Rust-accelerated (MIT), and
parselmouth (GPL).

Features extracted:
    - Pitch (F0): Fundamental frequency using Praat's autocorrelation method
    - Formants (F1-F4): Vocal tract resonances using Burg's method
    - Formant bandwidths (B1-B4): Width of formant peaks
    - Intensity: Sound pressure level in dB
    - HNR: Harmonics-to-noise ratio (voice quality measure)
    - Center of Gravity (CoG): Spectral centroid
    - Spectral tilt: Balance between low and high frequencies
    - A1-P0 nasal ratio: Nasality measure (amplitude at F0 vs ~250Hz)
    - Nasal murmur ratio: Low-frequency energy proportion

The module also provides spectrogram computation using either:
    - scipy with Gaussian window (default, high resolution)
    - Praat's spectrogram algorithm (optional)

Backend Selection:
    The acoustic analysis backend can be configured via the 'analysis.acoustic_backend'
    config option:
    - 'auto': Automatically select best available backend
    - 'praatfan': Pure Python implementation (MIT license)
    - 'praatfan-rust': Rust-accelerated implementation (MIT license)
    - 'parselmouth': Original Praat bindings (GPL license)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from praatfan import Sound as PraatfanSound, set_backend, get_backend, get_available_backends
from scipy import signal
import soundfile as sf

from ..config import config


# Backend initialization flag
_backend_initialized = False

# Display name mapping (internal name -> display name)
# Backends:
#   praatfan      - Pure Python - slow but portable (MIT)
#   praatfan_rust - Rust implementation - fast (MIT)
#   parselmouth   - Original Praat bindings (GPL)
BACKEND_DISPLAY_NAMES = {
    'praatfan': 'Praatfan (slow)',
    'praatfan_rust': 'Praatfan (fast)',
    'praatfan_gpl': 'Praatfan (GPL)',
    'parselmouth': 'Praat',
}

# Reverse mapping (display name -> internal name)
BACKEND_INTERNAL_NAMES = {v: k for k, v in BACKEND_DISPLAY_NAMES.items()}


def get_backend_display_name(internal_name: str) -> str:
    """Convert internal backend name to display name."""
    return BACKEND_DISPLAY_NAMES.get(internal_name, internal_name)


def get_backend_internal_name(display_name: str) -> str:
    """Convert display name to internal backend name."""
    return BACKEND_INTERNAL_NAMES.get(display_name, display_name)


def get_available_backends_display() -> list[str]:
    """Get list of available backends as display names, with 'Praat' last."""
    available = get_available_backends()
    display_names = [get_backend_display_name(b) for b in available]
    # Sort so 'Praat' is always last
    return sorted(display_names, key=lambda x: (x == 'Praat', x))


def _ensure_backend():
    """Initialize acoustic analysis backend from config.

    This function is called before any acoustic analysis to ensure
    the correct backend is selected based on configuration.

    Backend priority for 'auto' mode:
        1. praatfan_rust (fast, MIT)
        2. praatfan_gpl (fast, GPL)
        3. parselmouth (reference implementation, GPL)
        4. praatfan (slow, MIT fallback)
    """
    global _backend_initialized
    if _backend_initialized:
        return

    desired = config['analysis'].get('acoustic_backend', 'auto')
    available = get_available_backends()

    if desired == 'auto':
        # Auto-select best available backend (prefer fast Rust implementation)
        priority_order = ['praatfan_rust', 'parselmouth', 'praatfan']
        for backend in priority_order:
            if backend in available:
                set_backend(backend)
                print(f"Acoustic backend auto-selected: {get_backend_display_name(backend)}")
                break
    else:
        # Convert display name to internal name if needed (e.g., 'Praat' -> 'parselmouth')
        internal_name = get_backend_internal_name(desired)
        if internal_name in available:
            set_backend(internal_name)
            print(f"Acoustic backend set to: {desired}")
        else:
            print(f"Warning: Requested backend '{desired}' not available. "
                  f"Available: {available}. Using auto-selection.")
            # Fall back to auto-selection
            priority_order = ['praatfan_rust', 'parselmouth', 'praatfan']
            for backend in priority_order:
                if backend in available:
                    set_backend(backend)
                    print(f"Acoustic backend auto-selected: {get_backend_display_name(backend)}")
                    break

    _backend_initialized = True


def get_current_backend() -> str:
    """Get the currently active acoustic analysis backend (internal name).

    Returns:
        Internal name of the current backend (e.g., 'praatfan', 'parselmouth')
    """
    _ensure_backend()
    return get_backend()


def get_current_backend_display() -> str:
    """Get the currently active acoustic analysis backend (display name).

    Returns:
        Display name of the current backend (e.g., 'praatfan', 'praat')
    """
    return get_backend_display_name(get_current_backend())


def switch_backend(backend: str) -> bool:
    """Switch to a different acoustic analysis backend.

    Args:
        backend: Name of the backend (accepts both display and internal names)

    Returns:
        True if switch was successful, False otherwise
    """
    global _backend_initialized
    # Convert display name to internal name if needed
    internal_name = get_backend_internal_name(backend)
    available = get_available_backends()
    if internal_name in available:
        set_backend(internal_name)
        _backend_initialized = True
        return True
    return False


@dataclass
class AcousticFeatures:
    """
    Container for time-aligned acoustic features.

    All arrays are aligned to the same time axis (self.times).
    NaN values indicate frames where a feature could not be computed
    (e.g., pitch during unvoiced segments).

    Attributes:
        times: Time points in seconds for each frame
        f0: Fundamental frequency in Hz (NaN for unvoiced)
        intensity: Sound intensity in dB
        hnr: Harmonics-to-noise ratio in dB (voice quality)
        formants: Dict with 'F1' through 'F4' frequency arrays
        bandwidths: Dict with 'B1' through 'B4' bandwidth arrays
        cog: Center of gravity (spectral centroid) in Hz
        spectral_std: Standard deviation of spectrum
        skewness: Spectral skewness (asymmetry)
        kurtosis: Spectral kurtosis (peakedness)
        nasal_murmur_ratio: Ratio of low-freq (0-500Hz) to total energy
        spectral_tilt: Low vs high frequency balance in dB
        nasal_ratio: A1-P0 nasal ratio in dB (requires voicing)
    """

    times: np.ndarray
    f0: np.ndarray
    intensity: np.ndarray
    hnr: np.ndarray
    formants: dict[str, np.ndarray]
    bandwidths: dict[str, np.ndarray]
    cog: np.ndarray
    spectral_std: np.ndarray
    skewness: np.ndarray
    kurtosis: np.ndarray
    nasal_murmur_ratio: np.ndarray
    spectral_tilt: np.ndarray
    nasal_ratio: np.ndarray


def extract_features(
    audio_path: str | Path,
    time_step: float = 0.01,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
    max_formant: float = 5500.0,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    progress_callback: Optional[callable] = None
) -> AcousticFeatures:
    """
    Extract acoustic features from an audio file.

    Args:
        audio_path: Path to audio file
        time_step: Analysis time step in seconds
        pitch_floor: Minimum F0 for pitch analysis
        pitch_ceiling: Maximum F0 for pitch analysis
        max_formant: Maximum formant frequency (Hz)
        start_time: Start time for analysis (None = beginning)
        end_time: End time for analysis (None = end)
        progress_callback: Optional callback(progress: float) for progress updates

    Returns:
        AcousticFeatures with all extracted measurements
    """
    _ensure_backend()

    # Load audio and convert to mono if needed (praatfan backends require mono)
    samples, sample_rate = sf.read(str(audio_path), dtype='float64')
    if samples.ndim > 1:
        # Convert stereo to mono by averaging channels
        samples = np.mean(samples, axis=1)
    samples = np.ascontiguousarray(samples)

    total_duration = len(samples) / sample_rate
    snd = PraatfanSound(samples, sample_rate)

    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = total_duration

    # Extract the region of interest
    if start_time > 0 or end_time < total_duration:
        snd = snd.extract_part(start_time, end_time)
        analysis_duration = end_time - start_time
    else:
        analysis_duration = total_duration

    # Create analysis objects using praatfan's unified API
    # These compute the full analysis once; we then query values at each time point
    pitch_obj = snd.to_pitch_ac(
        time_step=0.0,  # Auto time step
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling
    )
    intensity = snd.to_intensity(
        minimum_pitch=pitch_floor,
        time_step=0.01
    )
    formants = snd.to_formant_burg(
        time_step=0.005,
        max_number_of_formants=5,
        maximum_formant=max_formant,
        window_length=0.025,
        pre_emphasis_from=50.0
    )
    harmonicity = snd.to_harmonicity_ac(
        time_step=0.01,
        minimum_pitch=pitch_floor,
        silence_threshold=0.1,
        periods_per_window=4.5
    )

    # Extract pitch using praatfan's unified API
    # Some backends return 0 for unvoiced, others return NaN - normalize to NaN
    pitch_times = pitch_obj.xs()
    pitch_values = pitch_obj.values()
    # Convert 0 to NaN (0 means unvoiced in Praat's pitch representation)
    pitch_values = np.where(pitch_values == 0, np.nan, pitch_values)

    # Generate time points for other features
    times = np.arange(0, analysis_duration, time_step)
    n_frames = len(times)

    # Helper to check if a pitch value is valid (not NaN and > 0)
    def is_voiced(val):
        return not np.isnan(val) and val > 0

    # Resample pitch to our time grid using log-scale interpolation
    # (pitch perception is logarithmic - we hear intervals as ratios)
    # Only interpolate between voiced frames; preserve NaN for unvoiced regions
    f0_vals = np.full(n_frames, np.nan)
    for i, t in enumerate(times):
        idx = np.searchsorted(pitch_times, t)
        if idx == 0:
            # Before first frame - use first value if voiced
            if is_voiced(pitch_values[0]):
                f0_vals[i] = pitch_values[0]
        elif idx >= len(pitch_times):
            # After last frame - use last value if voiced
            if is_voiced(pitch_values[-1]):
                f0_vals[i] = pitch_values[-1]
        else:
            # Between two frames - interpolate only if both are voiced
            val_before = pitch_values[idx - 1]
            val_after = pitch_values[idx]
            if is_voiced(val_before) and is_voiced(val_after):
                # Log-scale interpolation (linear in log space)
                t_before = pitch_times[idx - 1]
                t_after = pitch_times[idx]
                weight = (t - t_before) / (t_after - t_before)
                log_before = np.log(val_before)
                log_after = np.log(val_after)
                f0_vals[i] = np.exp(log_before + weight * (log_after - log_before))
            elif is_voiced(val_before):
                # Only before is voiced - use it
                f0_vals[i] = val_before
            elif is_voiced(val_after):
                # Only after is voiced - use it
                f0_vals[i] = val_after
            # else: both unvoiced - leave as NaN

    # === Extract batch features using praatfan's direct methods ===
    # This is much faster than per-frame call() queries

    # Intensity - interpolate to our time grid
    int_times = intensity.xs()
    int_values = intensity.values()
    intensity_vals = np.interp(times, int_times, int_values, left=np.nan, right=np.nan)

    # HNR - interpolate to our time grid
    hnr_times = harmonicity.xs()
    hnr_values = harmonicity.values()
    # Replace -200 (Praat's "undefined" for HNR) with NaN
    hnr_values = np.where(hnr_values <= -200, np.nan, hnr_values)
    hnr_vals = np.interp(times, hnr_times, hnr_values, left=np.nan, right=np.nan)

    # Formants and bandwidths - extract each formant, then interpolate
    formant_times = formants.xs()

    # Interpolate each formant to our time grid (formant_values takes 1-indexed formant number)
    f1_vals = np.interp(times, formant_times, formants.formant_values(1), left=np.nan, right=np.nan)
    f2_vals = np.interp(times, formant_times, formants.formant_values(2), left=np.nan, right=np.nan)
    f3_vals = np.interp(times, formant_times, formants.formant_values(3), left=np.nan, right=np.nan)
    f4_vals = np.interp(times, formant_times, formants.formant_values(4), left=np.nan, right=np.nan)

    bw1_vals = np.interp(times, formant_times, formants.bandwidth_values(1), left=np.nan, right=np.nan)
    bw2_vals = np.interp(times, formant_times, formants.bandwidth_values(2), left=np.nan, right=np.nan)
    bw3_vals = np.interp(times, formant_times, formants.bandwidth_values(3), left=np.nan, right=np.nan)
    bw4_vals = np.interp(times, formant_times, formants.bandwidth_values(4), left=np.nan, right=np.nan)

    # Replace 0 values with NaN (0 means undefined for formants)
    f1_vals = np.where(f1_vals == 0, np.nan, f1_vals)
    f2_vals = np.where(f2_vals == 0, np.nan, f2_vals)
    f3_vals = np.where(f3_vals == 0, np.nan, f3_vals)
    f4_vals = np.where(f4_vals == 0, np.nan, f4_vals)

    # === Spectral features using per-frame spectrum computation ===
    # Use per-frame FFT (extract window → spectrum → query) instead of batch
    # methods, which on some backends use a coarse spectrogram approximation
    # that produces different values from the standard per-frame approach.
    window_duration = 0.025
    n_frames = len(times)

    cog_vals = np.full(n_frames, np.nan)
    std_vals = np.full(n_frames, np.nan)
    skew_vals = np.full(n_frames, np.nan)
    kurt_vals = np.full(n_frames, np.nan)
    nasal_vals = np.full(n_frames, np.nan)
    tilt_vals = np.full(n_frames, np.nan)
    a1p0_vals = np.full(n_frames, np.nan)

    for i, t in enumerate(times):
        spectrum = snd.get_spectrum_at_time(t, window_duration)

        # Spectral moments
        cog_vals[i] = spectrum.get_center_of_gravity(2.0)
        std_vals[i] = spectrum.get_standard_deviation(2.0)
        skew_vals[i] = spectrum.get_skewness(2.0)
        kurt_vals[i] = spectrum.get_kurtosis(2.0)

        # Band energies for derived features
        band_low = spectrum.get_band_energy(0, 500)
        band_total = spectrum.get_band_energy(0, 5000)
        band_high = spectrum.get_band_energy(2000, 4000)
        band_nasal = spectrum.get_band_energy(200, 300)

        # Nasal murmur ratio: low-freq energy / total energy
        if band_total > 0:
            nasal_vals[i] = band_low / band_total

        # Spectral tilt: 10*log10(low) - 10*log10(high)
        if band_low > 0 and band_high > 0:
            tilt_vals[i] = 10 * np.log10(band_low) - 10 * np.log10(band_high)

        # A1-P0 nasal ratio: amplitude at F0 vs amplitude at ~250Hz
        f0 = f0_vals[i]
        if not np.isnan(f0) and f0 > 0:
            a1_band = spectrum.get_band_energy(f0 * 0.9, f0 * 1.1)
            if a1_band > 0 and band_nasal > 0:
                a1p0_vals[i] = 10 * np.log10(a1_band) - 10 * np.log10(band_nasal)

    if progress_callback:
        progress_callback(1.0)

    # Post-process formants: filter by bandwidth and frequency range
    # Use more lenient bandwidth thresholds (Praat-like)
    # Post-process formants using filter thresholds from config
    filters = config['formant_filters']
    f1_vals, bw1_vals = _filter_formant(f1_vals, bw1_vals, **filters['F1'])
    f2_vals, bw2_vals = _filter_formant(f2_vals, bw2_vals, **filters['F2'])
    f3_vals, bw3_vals = _filter_formant(f3_vals, bw3_vals, **filters['F3'])
    f4_vals, bw4_vals = _filter_formant(f4_vals, bw4_vals, **filters['F4'])

    # Apply median filtering to smooth outliers (window of 5 for better smoothing)
    f1_vals = _median_filter(f1_vals, window=5)
    f2_vals = _median_filter(f2_vals, window=5)
    f3_vals = _median_filter(f3_vals, window=5)

    # Interpolate small gaps (up to 3 frames) for continuity
    f1_vals = _interpolate_gaps(f1_vals, max_gap=3)
    f2_vals = _interpolate_gaps(f2_vals, max_gap=3)
    f3_vals = _interpolate_gaps(f3_vals, max_gap=3)

    # Adjust times to absolute values if we extracted a subset
    times = times + start_time

    return AcousticFeatures(
        times=times,
        f0=f0_vals,
        intensity=intensity_vals,
        hnr=hnr_vals,
        formants={'F1': f1_vals, 'F2': f2_vals, 'F3': f3_vals, 'F4': f4_vals},
        bandwidths={'B1': bw1_vals, 'B2': bw2_vals, 'B3': bw3_vals, 'B4': bw4_vals},
        cog=cog_vals,
        spectral_std=std_vals,
        skewness=skew_vals,
        kurtosis=kurt_vals,
        nasal_murmur_ratio=nasal_vals,
        spectral_tilt=tilt_vals,
        nasal_ratio=a1p0_vals
    )


def _filter_formant(
    formant: np.ndarray,
    bandwidth: np.ndarray | None,
    min_freq: float,
    max_freq: float,
    max_bandwidth: float | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Filter formant values based on frequency range and bandwidth.

    Values outside the expected range or with too high bandwidth are set to NaN.

    Args:
        formant: Array of formant frequency values
        bandwidth: Array of bandwidth values (can be None)
        min_freq: Minimum allowed frequency (Hz)
        max_freq: Maximum allowed frequency (Hz)
        max_bandwidth: Maximum allowed bandwidth (Hz), or None to skip filter
    """
    formant = formant.copy()
    if bandwidth is not None:
        bandwidth = bandwidth.copy()

    # Filter by frequency range
    out_of_range = (formant < min_freq) | (formant > max_freq)
    formant[out_of_range] = np.nan
    if bandwidth is not None:
        bandwidth[out_of_range] = np.nan

    # Filter by bandwidth (high bandwidth = unreliable)
    if max_bandwidth is not None and bandwidth is not None:
        high_bw = bandwidth > max_bandwidth
        formant[high_bw] = np.nan
        bandwidth[high_bw] = np.nan

    return formant, bandwidth


def _median_filter(data: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Apply median filter to smooth outliers while preserving NaN values.

    Uses nanmedian to properly ignore NaN values in the median calculation,
    avoiding distortion near gaps.
    """
    result = data.copy()
    n = len(data)
    half_window = window // 2

    for i in range(n):
        if np.isnan(data[i]):
            continue  # Preserve NaN positions

        # Get window bounds
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)

        # Compute median ignoring NaN values
        window_data = data[start:end]
        median_val = np.nanmedian(window_data)

        if not np.isnan(median_val):
            result[i] = median_val

    return result


def _interpolate_gaps(data: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """
    Interpolate small gaps (NaN runs) in the data.

    Only fills gaps of max_gap or fewer consecutive NaN values.
    """
    result = data.copy()
    n = len(data)

    # Find NaN positions
    is_nan = np.isnan(result)

    # Find start and end of each NaN run
    i = 0
    while i < n:
        if is_nan[i]:
            # Found start of NaN run
            gap_start = i
            while i < n and is_nan[i]:
                i += 1
            gap_end = i  # One past the last NaN

            gap_length = gap_end - gap_start

            # Only interpolate if gap is small enough and has valid values on both sides
            if gap_length <= max_gap:
                if gap_start > 0 and gap_end < n:
                    # Linear interpolation
                    start_val = result[gap_start - 1]
                    end_val = result[gap_end]
                    if not np.isnan(start_val) and not np.isnan(end_val):
                        for j in range(gap_start, gap_end):
                            t = (j - gap_start + 1) / (gap_length + 1)
                            result[j] = start_val + t * (end_val - start_val)
        else:
            i += 1

    return result


def compute_spectrogram(
    audio_path: str | Path,
    window_length: float = 0.025,
    max_frequency: float = 5000.0,
    dynamic_range: float = 70.0,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    pre_emphasis: float = 0.97,
    use_praat: bool = False  # scipy with Gaussian window gives better resolution
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram using Praat (via praatfan) or scipy.

    Args:
        audio_path: Path to audio file
        window_length: Analysis window length in seconds (0.005 = wideband, 0.025 = narrowband)
        max_frequency: Maximum frequency to display
        dynamic_range: Dynamic range in dB
        start_time: Start time (None = beginning)
        end_time: End time (None = end)
        pre_emphasis: Pre-emphasis coefficient (0.97 typical for speech)
        use_praat: If True, use Praat's spectrogram algorithm (recommended for speech)

    Returns:
        (times, frequencies, spectrogram_db) - spectrogram in dB
    """
    if use_praat:
        return _compute_spectrogram_praat(
            audio_path, window_length, max_frequency, dynamic_range,
            start_time, end_time, pre_emphasis
        )
    else:
        return _compute_spectrogram_scipy(
            audio_path, window_length, max_frequency, dynamic_range,
            start_time, end_time, pre_emphasis
        )


def _compute_spectrogram_praat(
    audio_path: str | Path,
    window_length: float = 0.025,
    max_frequency: float = 5000.0,
    dynamic_range: float = 70.0,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    pre_emphasis: float = 0.97
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectrogram using Praat's algorithm (Gaussian window)."""
    _ensure_backend()

    # Note: pre-emphasis not yet available in praatfan unified API
    # The spectrogram will be computed without pre-emphasis
    # (scipy version applies pre-emphasis manually and is used by default)

    # Load sound with praatfan (supports partial loading via start_time/end_time)
    snd = PraatfanSound(str(audio_path), start_time=start_time, end_time=end_time)

    # Compute spectrogram using praatfan's direct method
    # Use small time_step and frequency_step for high resolution display
    time_step = 0.001  # 1ms time step for smooth display
    freq_step = 5.0    # 5 Hz frequency resolution
    spectrogram = snd.to_spectrogram(
        window_length=window_length,
        maximum_frequency=max_frequency,
        time_step=time_step,
        frequency_step=freq_step
    )

    # Extract data using praatfan's unified API
    values = spectrogram.values()  # Shape: (n_freqs, n_times)
    times = spectrogram.xs()
    freqs = spectrogram.ys()

    # Convert to dB (Praat uses Pa^2/Hz, convert to dB)
    values_db = 10 * np.log10(values + 1e-30)

    # Apply dynamic range
    max_power = np.max(values_db)
    values_db = np.clip(values_db, max_power - dynamic_range, max_power)

    # Shift times to absolute values if we loaded a partial segment
    if start_time is not None and start_time > 0:
        times = times + start_time

    return times, freqs, values_db


def _compute_spectrogram_scipy(
    audio_path: str | Path,
    window_length: float = 0.025,
    max_frequency: float = 5000.0,
    dynamic_range: float = 70.0,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    pre_emphasis: float = 0.97
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectrogram using scipy with Gaussian window (Praat-like)."""
    # Get file info without loading the entire file
    info = sf.info(str(audio_path))
    sample_rate = info.samplerate
    total_frames = info.frames
    total_duration = total_frames / sample_rate

    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = total_duration

    # Ensure minimum duration for spectrogram computation
    # Need at least 2x window length for meaningful spectrogram
    min_duration = window_length * 3
    segment_duration = end_time - start_time
    if segment_duration < min_duration:
        # Expand the range symmetrically to meet minimum
        expand = (min_duration - segment_duration) / 2
        start_time = max(0, start_time - expand)
        end_time = min(total_duration, end_time + expand)
        # If still too short (near file boundaries), expand the other direction
        if end_time - start_time < min_duration:
            if start_time == 0:
                end_time = min(total_duration, min_duration)
            else:
                start_time = max(0, end_time - min_duration)

    # Calculate sample range to read (only read the portion we need)
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Ensure we have enough samples
    if end_sample - start_sample < int(window_length * sample_rate) * 2:
        raise ValueError(f"Audio segment too short for spectrogram: {end_time - start_time:.3f}s")

    # Read only the required portion of the audio file
    samples, _ = sf.read(
        str(audio_path),
        dtype='float64',
        start=start_sample,
        stop=end_sample
    )

    # Convert to mono if stereo
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    # Apply pre-emphasis filter to boost high frequencies (better for speech)
    if pre_emphasis > 0:
        samples = np.append(samples[0], samples[1:] - pre_emphasis * samples[:-1])

    # Compute spectrogram parameters
    nperseg = int(window_length * sample_rate)
    nperseg = max(256, nperseg)  # Minimum window size

    # Use larger nfft for better frequency resolution
    nfft = max(1024, 2 ** int(np.ceil(np.log2(nperseg * 2))))

    # Calculate overlap to get ~2000 time points for display (enough for most screens)
    # hop_size = (n_samples) / n_time_points
    n_samples = len(samples)
    target_time_points = 2000
    hop_size = max(1, n_samples // target_time_points)
    noverlap = max(0, nperseg - hop_size)

    # Create Gaussian window (Praat uses Gaussian)
    # std = nperseg/6 gives good frequency smoothing similar to Praat
    gaussian_window = signal.windows.gaussian(nperseg, std=nperseg/6)

    # Compute spectrogram with Gaussian window
    frequencies, times, Sxx = signal.spectrogram(
        samples,
        fs=sample_rate,
        window=gaussian_window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling='spectrum',
        mode='psd'
    )

    # Filter to max frequency
    freq_mask = frequencies <= max_frequency
    frequencies = frequencies[freq_mask]
    Sxx = Sxx[freq_mask, :]

    # Convert to dB
    power_db = 10 * np.log10(Sxx + 1e-10)

    # Apply dynamic range
    max_power = np.max(power_db)
    power_db = np.clip(power_db, max_power - dynamic_range, max_power)

    # Shift times to absolute values
    times = times + start_time

    return times, frequencies, power_db
