"""Waveform display widget using pyqtgraph."""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor

from ..audio.loader import AudioData


class WaveformWidget(pg.PlotWidget):
    """Widget for displaying audio waveform."""

    # Signals
    time_range_changed = pyqtSignal(float, float)  # (start, end)
    cursor_moved = pyqtSignal(float)  # time position
    selection_changed = pyqtSignal(float, float)  # (start, end)
    selection_clicked = pyqtSignal()  # emitted when user clicks inside selection

    def __init__(self, parent=None):
        super().__init__(parent)

        self._audio_data: AudioData | None = None
        self._cursor_time: float = 0.0
        self._selection_start: float | None = None
        self._selection_end: float | None = None
        self._is_dragging: bool = False
        self._click_inside_selection: bool = False
        self._click_start_pos: float | None = None

        self._setup_plot()
        self._setup_cursor()
        self._setup_selection()

    def _setup_plot(self):
        """Configure the plot appearance (Praat-like: white bg, black plot)."""
        self.setBackground('w')  # White background
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setLabel('left', 'Amplitude')
        self.setLabel('bottom', 'Time', units='s')

        # Set fixed axis widths for alignment with spectrogram
        self.plotItem.getAxis('left').setWidth(70)
        # Add a right axis placeholder to match spectrogram's pitch axis
        self.plotItem.showAxis('right')
        self.plotItem.getAxis('right').setWidth(70)
        self.plotItem.getAxis('right').setStyle(showValues=False)
        self.plotItem.getAxis('right').setTicks([])

        # Disable default mouse drag (we handle selection ourselves)
        self.setMouseEnabled(x=False, y=False)
        self.plotItem.vb.setDefaultPadding(0)

        # Create the waveform plot item (black line, thicker)
        self._waveform_curve = self.plot(
            pen=pg.mkPen(color='k', width=2)
        )

        # Connect view range changes
        self.sigXRangeChanged.connect(self._on_x_range_changed)

    def _setup_cursor(self):
        """Setup the playback cursor line."""
        self._cursor_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen(color=(200, 0, 0), width=2),
            movable=False
        )
        self.addItem(self._cursor_line)

    def _setup_selection(self):
        """Setup the selection region."""
        self._selection_region = pg.LinearRegionItem(
            values=[0, 0],
            brush=pg.mkBrush(180, 180, 255, 100),
            pen=pg.mkPen(color=(80, 80, 180), width=2),
            movable=True
        )
        self._selection_region.hide()
        self._selection_region.sigRegionChanged.connect(self._on_selection_changed)
        self.addItem(self._selection_region)

    def set_audio_data(self, audio_data: AudioData):
        """Load audio data for display."""
        self._audio_data = audio_data
        mono = audio_data.get_mono()

        # Downsample for display if very long
        max_points = 100000
        if len(mono) > max_points:
            step = len(mono) // max_points
            display_samples = mono[::step]
            display_times = audio_data.times[::step]
        else:
            display_samples = mono
            display_times = audio_data.times

        self._waveform_curve.setData(display_times, display_samples)

        # Set X range starting exactly at 0
        self.setXRange(0, audio_data.duration, padding=0)

        # Auto-scale Y to fit waveform
        amp_max = np.max(np.abs(mono))
        self.setYRange(-amp_max * 1.1, amp_max * 1.1, padding=0)

    def set_cursor_position(self, time: float):
        """Set the playback cursor position."""
        self._cursor_time = time
        self._cursor_line.setPos(time)
        self.cursor_moved.emit(time)

    def set_selection(self, start: float, end: float):
        """Set the selection region."""
        self._selection_start = min(start, end)
        self._selection_end = max(start, end)
        self._selection_region.setRegion([self._selection_start, self._selection_end])
        self._selection_region.show()
        self.selection_changed.emit(self._selection_start, self._selection_end)

    def clear_selection(self):
        """Clear the selection region."""
        self._selection_start = None
        self._selection_end = None
        self._selection_region.hide()

    def get_selection(self) -> tuple[float, float] | None:
        """Get current selection or None if no selection."""
        if self._selection_start is not None and self._selection_end is not None:
            return (self._selection_start, self._selection_end)
        return None

    def get_view_range(self) -> tuple[float, float]:
        """Get the current visible time range."""
        view_range = self.viewRange()
        return (view_range[0][0], view_range[0][1])

    def set_x_range(self, start: float, end: float):
        """Set the visible time range."""
        self.setXRange(start, end, padding=0)

    def _on_x_range_changed(self):
        """Handle view range changes."""
        x_range = self.viewRange()[0]
        self.time_range_changed.emit(x_range[0], x_range[1])

    def _on_selection_changed(self):
        """Handle selection region changes from user dragging."""
        region = self._selection_region.getRegion()
        self._selection_start = region[0]
        self._selection_end = region[1]
        self.selection_changed.emit(region[0], region[1])

    def mousePressEvent(self, ev):
        """Handle mouse press for selection (Praat-like behavior)."""
        if ev.button() == Qt.MouseButton.LeftButton:
            pos = self.plotItem.vb.mapSceneToView(ev.position())
            x = pos.x()
            self._click_start_pos = x
            self._click_inside_selection = False

            # Check if clicking inside existing selection
            if self._selection_region.isVisible():
                region = self._selection_region.getRegion()
                if region[0] <= x <= region[1]:
                    # Mark that we clicked inside selection
                    self._click_inside_selection = True
                    super().mousePressEvent(ev)
                    return

            # Start new selection
            self._selection_start = x
            self._is_dragging = True
            self._selection_region.setRegion([x, x])
            self._selection_region.show()
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        """Handle mouse move for selection."""
        if self._is_dragging and self._selection_start is not None:
            pos = self.plotItem.vb.mapSceneToView(ev.position())
            self._selection_region.setRegion([self._selection_start, pos.x()])
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        """Handle mouse release for selection."""
        if ev.button() == Qt.MouseButton.LeftButton:
            pos = self.plotItem.vb.mapSceneToView(ev.position())
            x = pos.x()

            # Check if this was a click inside selection (not a drag)
            if self._click_inside_selection and self._click_start_pos is not None:
                if abs(x - self._click_start_pos) < 0.01:
                    # It was just a click, not a drag - play the selection
                    self.selection_clicked.emit()
                    ev.accept()
                    self._click_inside_selection = False
                    self._click_start_pos = None
                    return
                self._click_inside_selection = False
                self._click_start_pos = None
                super().mouseReleaseEvent(ev)
                return

            if self._is_dragging:
                self._selection_end = x
                self._is_dragging = False

                if self._selection_start is None:
                    ev.accept()
                    return

                if abs(self._selection_end - self._selection_start) < 0.005:
                    # Click without drag - move cursor instead
                    cursor_pos = self._selection_start
                    self.clear_selection()
                    self.set_cursor_position(cursor_pos)
                else:
                    # Normalize selection (start < end)
                    if self._selection_start > self._selection_end:
                        self._selection_start, self._selection_end = self._selection_end, self._selection_start
                    self._selection_region.setRegion([self._selection_start, self._selection_end])
                    self.selection_changed.emit(self._selection_start, self._selection_end)
                ev.accept()
            else:
                super().mouseReleaseEvent(ev)
        else:
            super().mouseReleaseEvent(ev)

    def wheelEvent(self, ev):
        """Handle scroll wheel: vertical = zoom, horizontal = pan."""
        if self._audio_data is None:
            return

        x_min, x_max = self.get_view_range()
        x_range = x_max - x_min

        delta_x = ev.angleDelta().x()
        delta_y = ev.angleDelta().y()

        # Horizontal scroll = pan only (no zoom)
        if delta_x != 0:
            pan_amount = x_range * 0.05 * (-delta_x / 120.0)
            new_min = x_min + pan_amount
            new_max = x_max + pan_amount

            # Clamp to audio bounds
            if new_min < 0:
                new_max -= new_min
                new_min = 0
            if new_max > self._audio_data.duration:
                new_min -= (new_max - self._audio_data.duration)
                new_max = self._audio_data.duration
                new_min = max(0, new_min)

            self.setXRange(new_min, new_max, padding=0)

        # Vertical scroll = zoom only (slower zoom), centered on mouse position
        if delta_y != 0 and delta_x == 0:
            # Get mouse position in view coordinates
            pos = self.plotItem.vb.mapSceneToView(ev.position())
            mouse_x = pos.x()

            # Clamp mouse position to valid range
            mouse_x = max(x_min, min(x_max, mouse_x))

            if delta_y > 0:
                factor = 0.9  # Zoom in
            else:
                factor = 1.1  # Zoom out

            # Calculate new range keeping mouse position fixed
            # mouse_x should be at the same relative position after zoom
            left_frac = (mouse_x - x_min) / x_range
            new_range = x_range * factor

            new_min = mouse_x - left_frac * new_range
            new_max = mouse_x + (1 - left_frac) * new_range

            # Clamp to audio bounds
            if new_min < 0:
                new_max -= new_min
                new_min = 0
            if new_max > self._audio_data.duration:
                new_min -= (new_max - self._audio_data.duration)
                new_max = self._audio_data.duration
                new_min = max(0, new_min)

            self.setXRange(new_min, new_max, padding=0)

        ev.accept()
