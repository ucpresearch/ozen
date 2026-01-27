"""Acoustic analysis modules."""

from .acoustic import (
    AcousticFeatures,
    extract_features,
    compute_spectrogram,
    get_current_backend,
    get_current_backend_display,
    switch_backend,
    get_available_backends_display,
    get_backend_display_name,
    get_backend_internal_name,
)
from praatfan import get_available_backends

__all__ = [
    'AcousticFeatures',
    'extract_features',
    'compute_spectrogram',
    'get_current_backend',
    'get_current_backend_display',
    'switch_backend',
    'get_available_backends',
    'get_available_backends_display',
    'get_backend_display_name',
    'get_backend_internal_name',
]
