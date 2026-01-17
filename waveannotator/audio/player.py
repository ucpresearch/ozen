"""Audio playback using sounddevice."""

from typing import Optional, Callable
import threading

import numpy as np
import sounddevice as sd

from .loader import AudioData


class AudioPlayer:
    """Audio playback controller."""

    def __init__(self):
        self._audio_data: AudioData | None = None
        self._is_playing: bool = False
        self._current_frame: int = 0
        self._start_frame: int = 0
        self._end_frame: int = 0
        self._stream: sd.OutputStream | None = None
        self._lock = threading.Lock()

        # Callbacks
        self._on_position_changed: Callable[[float], None] | None = None
        self._on_playback_finished: Callable[[], None] | None = None

    def set_audio_data(self, audio_data: AudioData):
        """Set the audio data to play."""
        self.stop()
        self._audio_data = audio_data
        self._current_frame = 0
        self._start_frame = 0
        self._end_frame = len(audio_data.samples)

    def set_position_callback(self, callback: Callable[[float], None]):
        """Set callback for position updates during playback."""
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
        return self._current_frame / self._audio_data.sample_rate

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

        sr = self._audio_data.sample_rate
        total_samples = len(self._audio_data.samples)

        if start_time is not None:
            self._start_frame = int(start_time * sr)
        else:
            self._start_frame = self._current_frame

        if end_time is not None:
            self._end_frame = int(end_time * sr)
        else:
            self._end_frame = total_samples

        # Clamp to valid range
        self._start_frame = max(0, min(self._start_frame, total_samples))
        self._end_frame = max(self._start_frame, min(self._end_frame, total_samples))

        self._current_frame = self._start_frame
        self._is_playing = True

        # Get audio data
        mono = self._audio_data.get_mono()

        def callback(outdata, frames, time_info, status):
            with self._lock:
                if not self._is_playing:
                    outdata.fill(0)
                    raise sd.CallbackStop()

                remaining = self._end_frame - self._current_frame
                if remaining <= 0:
                    outdata.fill(0)
                    self._is_playing = False
                    raise sd.CallbackStop()

                chunk_size = min(frames, remaining)
                outdata[:chunk_size, 0] = mono[self._current_frame:self._current_frame + chunk_size]
                if chunk_size < frames:
                    outdata[chunk_size:].fill(0)

                self._current_frame += chunk_size

                # Notify position change
                if self._on_position_changed:
                    current_time = self._current_frame / sr
                    # Schedule callback on main thread would be better
                    # but for now direct call
                    self._on_position_changed(current_time)

        def finished_callback():
            self._is_playing = False
            if self._on_playback_finished:
                self._on_playback_finished()

        self._stream = sd.OutputStream(
            samplerate=sr,
            channels=1,
            callback=callback,
            finished_callback=finished_callback
        )
        self._stream.start()

    def pause(self):
        """Pause playback."""
        if self._stream is not None and self._is_playing:
            self._is_playing = False
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def stop(self):
        """Stop playback and reset position."""
        self.pause()
        self._current_frame = self._start_frame

    def seek(self, time: float):
        """Seek to a specific time."""
        if self._audio_data is None:
            return
        self._current_frame = int(time * self._audio_data.sample_rate)
        self._current_frame = max(0, min(self._current_frame,
                                         len(self._audio_data.samples)))

    def toggle_play_pause(self, start_time: Optional[float] = None,
                          end_time: Optional[float] = None):
        """Toggle between play and pause."""
        if self._is_playing:
            self.pause()
        else:
            self.play(start_time, end_time)
