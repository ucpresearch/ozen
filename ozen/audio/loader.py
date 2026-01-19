"""Audio file loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class AudioData:
    """Container for loaded audio data."""

    samples: np.ndarray  # Audio samples (mono or stereo)
    sample_rate: int
    duration: float  # Duration in seconds
    channels: int
    file_path: Path

    @property
    def times(self) -> np.ndarray:
        """Return time array for samples."""
        return np.arange(len(self.samples)) / self.sample_rate

    def get_mono(self) -> np.ndarray:
        """Return mono version of audio (average if stereo)."""
        if self.channels == 1:
            return self.samples
        return np.mean(self.samples, axis=1)


def load_audio(file_path: str | Path) -> AudioData:
    """
    Load an audio file and return AudioData.

    Supports formats: WAV, FLAC, OGG, etc. (via soundfile/libsndfile)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    samples, sample_rate = sf.read(file_path, dtype='float64')

    # Handle mono vs stereo
    if samples.ndim == 1:
        channels = 1
    else:
        channels = samples.shape[1]

    duration = len(samples) / sample_rate

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration=duration,
        channels=channels,
        file_path=file_path
    )
