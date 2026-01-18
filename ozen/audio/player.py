"""
Audio playback using sounddevice.

This module provides low-latency audio playback functionality using the
sounddevice library (Python bindings for PortAudio). It supports:
- Playing full files or selected regions
- Pause/resume functionality
- Position tracking with callbacks
- Thread-safe playback state management
"""

from typing import Optional, Callable
import threading
import time

import numpy as np
import sounddevice as sd

from .loader import AudioData


class AudioPlayer:
    """
    Audio playback controller using sounddevice.

    This class manages audio playback with support for:
    - Playing entire files or time-range selections
    - Pause/resume/stop controls
    - Real-time position tracking via callbacks
    - Thread-safe state management (audio runs on a separate thread)

    Attributes:
        is_playing: True if audio is currently playing
        current_time: Current playback position in seconds
    """

    def __init__(self):
        self._audio_data: AudioData | None = None
        self._is_playing: bool = False
        self._start_time: float = 0.0  # Start of playback region (seconds)
        self._end_time: float = 0.0    # End of playback region (seconds)
        self._playback_started_at: float = 0.0  # Wall clock time when playback started
        self._playback_offset: float = 0.0  # Position offset when playback started
        self._lock = threading.Lock()

        # User-provided callbacks for playback events
        self._on_position_changed: Callable[[float], None] | None = None
        self._on_playback_finished: Callable[[], None] | None = None

    def set_audio_data(self, audio_data: AudioData):
        """Set the audio data to play."""
        self.stop()
        self._audio_data = audio_data
        self._start_time = 0.0
        self._end_time = audio_data.duration
        self._playback_offset = 0.0

    def set_position_callback(self, callback: Callable[[float], None]):
        """Set callback for position updates during playback.

        Note: This callback is no longer called from the audio thread
        to avoid thread-safety issues. Use the current_time property
        with a QTimer for position updates instead.
        """
        self._on_position_changed = callback

    def set_finished_callback(self, callback: Callable[[], None]):
        """Set callback for when playback finishes."""
        self._on_playback_finished = callback

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def current_time(self) -> float:
        if self._audio_data is None:
            return 0.0
        if not self._is_playing:
            return self._playback_offset
        # Calculate position based on wall clock time
        elapsed = time.time() - self._playback_started_at
        pos = self._playback_offset + elapsed
        # Clamp to region bounds
        return min(pos, self._end_time)

    def play(self, start_time: Optional[float] = None, end_time: Optional[float] = None):
        """
        Start playback.

        Args:
            start_time: Start time in seconds (None = current position)
            end_time: End time in seconds (None = end of file)
        """
        if self._audio_data is None:
            return

        self.stop()

        duration = self._audio_data.duration
        sr = self._audio_data.sample_rate

        if start_time is not None:
            self._start_time = max(0.0, min(start_time, duration))
        else:
            self._start_time = self._playback_offset

        if end_time is not None:
            self._end_time = max(self._start_time, min(end_time, duration))
        else:
            self._end_time = duration

        self._playback_offset = self._start_time
        self._playback_started_at = time.time()
        self._is_playing = True

        # Extract the region to play
        start_frame = int(self._start_time * sr)
        end_frame = int(self._end_time * sr)

        # Get audio data and convert to float32
        samples = self._audio_data.samples[start_frame:end_frame]
        if samples.ndim == 1:
            # Mono - duplicate to stereo for macOS compatibility
            samples = np.column_stack([samples, samples])
        # Ensure contiguous float32 array
        samples = np.ascontiguousarray(samples, dtype=np.float32)

        def playback_thread():
            try:
                # Small delay for macOS CoreAudio to settle after stop()
                time.sleep(0.05)
                sd.play(samples, sr)
                sd.wait()
            except Exception:
                pass
            finally:
                with self._lock:
                    if self._is_playing:  # Only if not manually stopped
                        self._is_playing = False
                        self._playback_offset = self._end_time
                        if self._on_playback_finished:
                            self._on_playback_finished()

        thread = threading.Thread(target=playback_thread, daemon=True)
        thread.start()

    def pause(self):
        """Pause playback."""
        if self._is_playing:
            with self._lock:
                # Record current position before stopping
                self._playback_offset = self.current_time
                self._is_playing = False
            sd.stop()

    def stop(self):
        """Stop playback and reset position."""
        with self._lock:
            self._is_playing = False
            self._playback_offset = self._start_time
        sd.stop()

    def seek(self, time_pos: float):
        """Seek to a specific time."""
        if self._audio_data is None:
            return
        self._playback_offset = max(0.0, min(time_pos, self._audio_data.duration))

    def toggle_play_pause(self, start_time: Optional[float] = None,
                          end_time: Optional[float] = None):
        """Toggle between play and pause."""
        if self._is_playing:
            self.pause()
        else:
            self.play(start_time, end_time)
