"""
Acoustic analysis using Parselmouth (Praat) and scipy.

This module provides acoustic feature extraction for speech analysis.
It uses Parselmouth (Python bindings for Praat) for core acoustic
measurements and scipy for spectrogram computation.

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
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy import signal
import soundfile as sf

from ..config import config


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
    snd = parselmouth.Sound(str(audio_path))
    total_duration = call(snd, "Get total duration")

    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = total_duration

    # Extract the region of interest
    if start_time > 0 or end_time < total_duration:
        snd = call(snd, "Extract part", start_time, end_time, "rectangular", 1.0, "no")
        analysis_duration = end_time - start_time
    else:
        analysis_duration = total_duration

    # Create Praat analysis objects
    # These compute the full analysis once; we then query values at each time point
    pitch = call(snd, "To Pitch", 0.01, pitch_floor, pitch_ceiling)  # Autocorrelation method
    intensity = call(snd, "To Intensity", pitch_floor, 0.01)  # RMS energy
    formants = call(snd, "To Formant (burg)", 0.005, 5, max_formant, 0.025, 50)  # Burg's method
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, pitch_floor, 0.1, 1.0)  # Cross-correlation HNR

    # Generate time points
    times = np.arange(0, analysis_duration, time_step)
    n_frames = len(times)

    # Initialize arrays
    f0_vals = np.full(n_frames, np.nan)
    intensity_vals = np.full(n_frames, np.nan)
    hnr_vals = np.full(n_frames, np.nan)
    f1_vals = np.full(n_frames, np.nan)
    f2_vals = np.full(n_frames, np.nan)
    f3_vals = np.full(n_frames, np.nan)
    f4_vals = np.full(n_frames, np.nan)
    bw1_vals = np.full(n_frames, np.nan)
    bw2_vals = np.full(n_frames, np.nan)
    bw3_vals = np.full(n_frames, np.nan)
    bw4_vals = np.full(n_frames, np.nan)
    cog_vals = np.full(n_frames, np.nan)
    std_vals = np.full(n_frames, np.nan)
    skew_vals = np.full(n_frames, np.nan)
    kurt_vals = np.full(n_frames, np.nan)
    nasal_vals = np.full(n_frames, np.nan)
    tilt_vals = np.full(n_frames, np.nan)
    a1p0_vals = np.full(n_frames, np.nan)  # A1-P0 nasal ratio

    window_duration = 0.025  # Window size for spectral analysis

    # Main extraction loop: query each feature at each time point
    # This is slow but provides maximum accuracy
    for i, t in enumerate(times):
        if progress_callback and i % 100 == 0:
            progress_callback(i / n_frames)

        # === Basic features from pre-computed Praat objects ===
        f0_val = call(pitch, "Get value at time", t, "Hertz", "Linear")
        f0_vals[i] = f0_val if f0_val else np.nan

        int_val = call(intensity, "Get value at time", t, "Cubic")
        intensity_vals[i] = int_val if int_val else np.nan

        hnr_val = call(harmonicity, "Get value at time", t, "Linear")
        hnr_vals[i] = hnr_val if hnr_val else np.nan

        # Formants and bandwidths
        f1_vals[i] = call(formants, "Get value at time", 1, t, "Hertz", "Linear") or np.nan
        f2_vals[i] = call(formants, "Get value at time", 2, t, "Hertz", "Linear") or np.nan
        f3_vals[i] = call(formants, "Get value at time", 3, t, "Hertz", "Linear") or np.nan
        f4_vals[i] = call(formants, "Get value at time", 4, t, "Hertz", "Linear") or np.nan
        bw1_vals[i] = call(formants, "Get bandwidth at time", 1, t, "Hertz", "Linear") or np.nan
        bw2_vals[i] = call(formants, "Get bandwidth at time", 2, t, "Hertz", "Linear") or np.nan
        bw3_vals[i] = call(formants, "Get bandwidth at time", 3, t, "Hertz", "Linear") or np.nan
        bw4_vals[i] = call(formants, "Get bandwidth at time", 4, t, "Hertz", "Linear") or np.nan

        # Spectral moments (from short-time spectrum)
        t_start = max(0, t - window_duration / 2)
        t_end = min(analysis_duration, t + window_duration / 2)
        segment = call(snd, "Extract part", t_start, t_end, "rectangular", 1.0, "no")
        spectrum = call(segment, "To Spectrum", "yes")

        cog_vals[i] = call(spectrum, "Get centre of gravity", 2) or np.nan
        std_vals[i] = call(spectrum, "Get standard deviation", 2) or np.nan
        skew_vals[i] = call(spectrum, "Get skewness", 2) or np.nan
        kurt_vals[i] = call(spectrum, "Get kurtosis", 2) or np.nan

        # Nasal-related features
        low_freq_energy = call(spectrum, "Get band energy", 0, 500)
        total_energy = call(spectrum, "Get band energy", 0, 5000)
        nasal_vals[i] = low_freq_energy / total_energy if total_energy > 0 else np.nan

        band_low = call(spectrum, "Get band energy", 0, 500)
        band_high = call(spectrum, "Get band energy", 2000, 4000)
        if band_low > 0 and band_high > 0:
            tilt_vals[i] = 10 * np.log10(band_low) - 10 * np.log10(band_high)
        else:
            tilt_vals[i] = np.nan

        # A1-P0 nasal ratio: amplitude of first harmonic minus nasal pole amplitude
        # A1 is at F0 frequency, P0 is in the ~250 Hz nasal region
        if f0_val and not np.isnan(f0_val) and f0_val > 0:
            # Get amplitude at F0 (A1 - first harmonic)
            a1_amp = call(spectrum, "Get band energy", f0_val * 0.9, f0_val * 1.1)
            # Get amplitude at nasal pole region (around 250 Hz, typical P0 region)
            p0_amp = call(spectrum, "Get band energy", 200, 300)
            if a1_amp > 0 and p0_amp > 0:
                a1p0_vals[i] = 10 * np.log10(a1_amp) - 10 * np.log10(p0_amp)
            else:
                a1p0_vals[i] = np.nan
        else:
            a1p0_vals[i] = np.nan

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
    """
    from scipy.ndimage import median_filter as scipy_median

    result = data.copy()

    # Only filter non-NaN values
    valid = ~np.isnan(data)
    if np.sum(valid) < window:
        return result

    # Create a version with NaN replaced by interpolation for filtering
    temp = data.copy()

    # Simple approach: filter and keep NaN positions
    # Use scipy's median filter on valid stretches
    filtered = scipy_median(np.nan_to_num(data, nan=0), size=window)

    # Only update positions that were originally valid
    result[valid] = filtered[valid]

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
    Compute spectrogram using Praat (via parselmouth) or scipy.

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
    # Load sound with parselmouth
    snd = parselmouth.Sound(str(audio_path))

    # Apply pre-emphasis if requested
    if pre_emphasis > 0:
        snd = call(snd, "Filter (pre-emphasis)", 50.0)

    total_duration = call(snd, "Get total duration")

    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = total_duration

    # Extract region of interest
    if start_time > 0 or end_time < total_duration:
        snd = call(snd, "Extract part", start_time, end_time, "rectangular", 1.0, "no")

    # Compute spectrogram using Praat
    # Parameters: time_step, max_freq, window_length, freq_step, window_shape
    # Use small time_step and freq_step for high resolution display
    time_step = 0.001  # 1ms time step for smooth display
    freq_step = 5.0    # 5 Hz frequency resolution (gives ~1000 freq bins for 5kHz)
    spectrogram = call(snd, "To Spectrogram", time_step, max_frequency,
                       window_length, freq_step, "Gaussian")

    # Extract data
    values = np.array(spectrogram.values)  # Shape: (n_freqs, n_times)
    n_freqs, n_times = values.shape

    # Get time and frequency arrays
    # Use center times for each frame
    times = np.array([spectrogram.get_time_from_frame_number(i + 1) for i in range(n_times)])
    freqs = np.linspace(spectrogram.ymin, spectrogram.ymax, n_freqs)

    # Convert to dB (Praat uses Pa^2/Hz, convert to dB)
    values_db = 10 * np.log10(values + 1e-30)

    # Apply dynamic range
    max_power = np.max(values_db)
    values_db = np.clip(values_db, max_power - dynamic_range, max_power)

    # Shift times to absolute values
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
    # Load audio
    samples, sample_rate = sf.read(str(audio_path), dtype='float64')

    # Convert to mono if stereo
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    total_duration = len(samples) / sample_rate

    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = total_duration

    # Extract region of interest
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    samples = samples[start_sample:end_sample]

    # Apply pre-emphasis filter to boost high frequencies (better for speech)
    if pre_emphasis > 0:
        samples = np.append(samples[0], samples[1:] - pre_emphasis * samples[:-1])

    # Compute spectrogram parameters
    nperseg = int(window_length * sample_rate)
    nperseg = max(256, nperseg)  # Minimum window size

    # Use larger nfft for better frequency resolution
    nfft = max(1024, 2 ** int(np.ceil(np.log2(nperseg * 4))))

    # High overlap for smooth display (95% like Praat)
    noverlap = int(nperseg * 0.95)

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
