"""Spectrogram display widget with overlay tracks."""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor, QTransform, QFont

from ..analysis.acoustic import AcousticFeatures


def create_spectrogram_colormap():
    """Create a Praat-like grayscale colormap (white=low, black=high)."""
    positions = [0.0, 1.0]
    colors = [
        (255, 255, 255, 255),  # White (low energy)
        (0, 0, 0, 255),        # Black (high energy)
    ]
    return pg.ColorMap(positions, colors)


def create_viridis_colormap():
    """Create a viridis-like colormap."""
    positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = [
        (68, 1, 84, 255),
        (59, 82, 139, 255),
        (33, 145, 140, 255),
        (94, 201, 98, 255),
        (253, 231, 37, 255),
    ]
    return pg.ColorMap(positions, colors)


def create_inferno_colormap():
    """Create an inferno-like colormap."""
    positions = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = [
        (0, 0, 4, 255),
        (87, 16, 110, 255),
        (188, 55, 84, 255),
        (249, 142, 9, 255),
        (252, 255, 164, 255),
    ]
    return pg.ColorMap(positions, colors)


COLORMAPS = {
    'grayscale': create_spectrogram_colormap,
    'viridis': create_viridis_colormap,
    'inferno': create_inferno_colormap,
}

# Formant presets for different voice types
FORMANT_PRESETS = {
    'male': {'max_formant': 5000, 'num_formants': 5},
    'female': {'max_formant': 5500, 'num_formants': 5},
    'child': {'max_formant': 8000, 'num_formants': 5},
}


class SpectrogramWidget(pg.GraphicsLayoutWidget):
    """Widget for displaying spectrogram with acoustic overlays."""

    # Signals
    time_range_changed = pyqtSignal(float, float)
    cursor_moved = pyqtSignal(float)
    selection_changed = pyqtSignal(float, float)
    selection_clicked = pyqtSignal()  # emitted when user clicks inside selection

    def __init__(self, parent=None):
        super().__init__(parent)

        self._spectrogram_data: np.ndarray | None = None
        self._times: np.ndarray | None = None
        self._frequencies: np.ndarray | None = None
        self._freq_start: float = 0.0  # Start of frequency range
        self._freq_end: float = 5000.0  # End of frequency range
        self._features: AcousticFeatures | None = None
        self._cursor_time: float = 0.0
        self._duration: float = 0.0
        self._is_dragging: bool = False
        self._selection_start: float | None = None
        self._selection_end: float | None = None
        self._click_inside_selection: bool = False
        self._click_start_pos: float | None = None

        # Track visibility settings
        self._track_visibility = {
            'pitch': True,
            'formants': True,
            'intensity': False,
            'cog': False,
            'hnr': False,
        }

        self._setup_layout()
        self._setup_spectrogram()
        self._setup_pitch_axis()
        self._setup_overlays()
        self._setup_selection()
        self._setup_cursor()
        self._setup_tooltip()

    def _setup_tooltip(self):
        """Setup tooltip for showing acoustic values."""
        self.setMouseTracking(True)
        # Create a text item for displaying values (more visible than tooltip)
        self._info_label = pg.TextItem(
            text="",
            color=(0, 0, 0),
            anchor=(0, 0),
            fill=pg.mkBrush(255, 255, 255, 200)
        )
        self._info_label.setFont(QFont("Monospace", 8))
        self._info_label.hide()
        self._plot.addItem(self._info_label)

    def _setup_layout(self):
        """Setup the graphics layout (Praat-like white background)."""
        self.setBackground('w')

        # Create the main plot
        self._plot = self.addPlot(row=0, col=0)
        self._plot.setLabel('left', 'Frequency', units='Hz')
        self._plot.setLabel('bottom', 'Time', units='s')
        self._plot.showGrid(x=True, y=False, alpha=0.2)

        # Set fixed Y-axis width for alignment with waveform
        self._plot.getAxis('left').setWidth(70)

        # Disable default mouse drag (we handle selection ourselves)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.vb.setDefaultPadding(0)
        # Hide the autorange "A" button
        self._plot.hideButtons()

        # Connect view range changes
        self._plot.sigXRangeChanged.connect(self._on_x_range_changed)

    def _setup_pitch_axis(self):
        """Setup pitch display (on main plot, scaled to frequency range)."""
        # Fixed pitch range like Praat (50-800 Hz)
        self._pitch_min = 50.0
        self._pitch_max = 800.0

        # Use the plot's built-in right axis for pitch scale
        self._plot.showAxis('right')
        self._pitch_axis = self._plot.getAxis('right')
        self._pitch_axis.setLabel('Pitch (Hz)', color='#0000C8')
        self._pitch_axis.setWidth(70)
        self._pitch_axis.setPen(pg.mkPen(color=(0, 0, 200)))
        self._pitch_axis.setTextPen(pg.mkPen(color=(0, 0, 200)))

    def _setup_spectrogram(self):
        """Setup the spectrogram image item."""
        self._spectrogram_img = pg.ImageItem()
        self._plot.addItem(self._spectrogram_img)

        # Use grayscale by default (Praat-like)
        cmap = create_spectrogram_colormap()
        self._spectrogram_img.setLookupTable(cmap.getLookupTable())

    def _setup_selection(self):
        """Setup selection region."""
        self._selection_region = pg.LinearRegionItem(
            values=[0, 0],
            brush=pg.mkBrush(180, 180, 255, 100),
            pen=pg.mkPen(color=(80, 80, 180), width=2),
            movable=True
        )
        self._selection_region.hide()
        self._selection_region.sigRegionChanged.connect(self._on_selection_changed)
        self._plot.addItem(self._selection_region)

    def _setup_overlays(self):
        """Setup overlay plot items with distinct colors and thick lines."""
        # Pitch track - blue line (scaled to frequency range for display)
        self._pitch_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color=(0, 0, 200), width=3),
            name='Pitch'
        )
        self._plot.addItem(self._pitch_curve)

        # Intensity - yellow line
        self._intensity_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color=(220, 180, 0), width=3),
            name='Intensity'
        )
        self._plot.addItem(self._intensity_curve)

        # Formants - red dots, color fades to pink/white with higher bandwidth
        self._formant_items = {}
        # Constant size for all formants (F4 slightly smaller as it's less reliable)
        formant_sizes = {'F1': 10, 'F2': 10, 'F3': 9, 'F4': 7}

        for formant_key in ['F1', 'F2', 'F3', 'F4']:
            scatter = pg.ScatterPlotItem(
                pen=None,
                brush=pg.mkBrush(200, 0, 0, 255),  # Default red, will be updated per-point
                size=formant_sizes[formant_key],
                name=formant_key
            )
            self._plot.addItem(scatter)
            self._formant_items[formant_key] = {'scatter': scatter, 'size': formant_sizes[formant_key]}

        # Center of Gravity - green
        self._cog_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color=(0, 180, 0), width=3),
            name='CoG'
        )
        self._plot.addItem(self._cog_curve)

        # HNR - bright purple dotted
        self._hnr_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color=(180, 0, 220), width=3, style=Qt.PenStyle.DashLine),
            name='HNR'
        )
        self._plot.addItem(self._hnr_curve)

    def _setup_cursor(self):
        """Setup playback cursor."""
        self._cursor_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen(color=(200, 0, 0), width=2),
            movable=False
        )
        self._plot.addItem(self._cursor_line)

    def set_spectrogram(
        self,
        times: np.ndarray,
        frequencies: np.ndarray,
        spectrogram_db: np.ndarray
    ):
        """Set the spectrogram data."""
        self._times = times
        self._frequencies = frequencies
        self._spectrogram_data = spectrogram_db

        # Store duration for zoom limits
        self._duration = times[-1] if len(times) > 0 else 0

        # Normalize to 0-1 range for colormap
        data_min = np.min(spectrogram_db)
        data_max = np.max(spectrogram_db)
        normalized = (spectrogram_db - data_min) / (data_max - data_min + 1e-10)

        self._spectrogram_img.setImage(normalized.T)

        # Set transform - start exactly at 0
        t_start = 0  # Always start at 0 for alignment
        t_end = times[-1] if len(times) > 0 else 1
        f_start = frequencies[0] if len(frequencies) > 0 else 0
        f_end = frequencies[-1] if len(frequencies) > 0 else 5000

        # Store frequency range for pitch scaling
        self._freq_start = f_start
        self._freq_end = f_end

        # The spectrogram data may not start exactly at 0, so we need to adjust
        actual_t_start = times[0] if len(times) > 0 else 0
        t_scale = (t_end - actual_t_start) / max(len(times), 1)
        f_scale = (f_end - f_start) / max(len(frequencies), 1)

        transform = QTransform()
        transform.translate(actual_t_start, f_start)
        transform.scale(t_scale, f_scale)
        self._spectrogram_img.setTransform(transform)

        # Set view ranges starting at 0
        self._plot.setXRange(0, t_end, padding=0)
        self._plot.setYRange(f_start, f_end, padding=0)

    def _update_pitch_axis_ticks(self):
        """Update the right axis to show pitch values instead of frequency."""
        # Create tick values at nice pitch intervals
        pitch_ticks = [50, 100, 150, 200, 300, 400, 500, 600, 800]

        # Convert pitch values to frequency axis positions
        p_min = self._pitch_min
        p_max = self._pitch_max
        f_start = self._freq_start
        f_end = self._freq_end
        freq_range = f_end - f_start

        ticks = []
        for pitch in pitch_ticks:
            if p_min <= pitch <= p_max:
                # Map pitch (p_min to p_max) to frequency axis (f_start to f_end)
                freq_pos = (pitch - p_min) / (p_max - p_min) * freq_range + f_start
                ticks.append((freq_pos, str(int(pitch))))

        self._pitch_axis.setTicks([ticks])

    def set_features(self, features: AcousticFeatures):
        """Set acoustic features for overlay display."""
        self._features = features
        self._update_overlays()

    def _update_overlays(self):
        """Update overlay plots with current feature data."""
        if self._features is None:
            return

        f = self._features

        # Pitch - scaled to display on spectrogram with fixed range (50-800 Hz like Praat)
        if self._track_visibility['pitch'] and self._frequencies is not None:
            valid = ~np.isnan(f.f0)
            if np.any(valid):
                times_valid = f.times[valid]
                f0_valid = f.f0[valid]

                # Fixed pitch range like Praat
                p_min = self._pitch_min  # 50 Hz
                p_max = self._pitch_max  # 800 Hz

                # Use stored frequency range for proper scaling
                f_start = self._freq_start
                f_end = self._freq_end
                freq_range = f_end - f_start

                # Map pitch (50-800 Hz) to frequency axis (f_start - f_end)
                scaled_pitch = (f0_valid - p_min) / (p_max - p_min) * freq_range + f_start
                # Clip to valid range
                scaled_pitch = np.clip(scaled_pitch, f_start, f_end)

                self._pitch_curve.setData(times_valid, scaled_pitch)

                # Update right axis tick labels to show pitch values
                self._update_pitch_axis_ticks()
            self._pitch_curve.show()
        else:
            self._pitch_curve.hide()

        # Intensity
        if self._track_visibility['intensity'] and self._frequencies is not None:
            valid = ~np.isnan(f.intensity)
            f_start = self._freq_start
            f_end = self._freq_end
            freq_range = f_end - f_start
            scaled_intensity = (f.intensity - 30) / 60 * freq_range + f_start
            scaled_intensity = np.clip(scaled_intensity, f_start, f_end)
            self._intensity_curve.setData(f.times[valid], scaled_intensity[valid])
            self._intensity_curve.show()
        else:
            self._intensity_curve.hide()

        # Formants with bandwidth-based coloring (higher bandwidth = more pink/white)
        if self._track_visibility['formants']:
            bw_keys = {'F1': 'B1', 'F2': 'B2', 'F3': 'B3', 'F4': None}
            # Bandwidth thresholds for color scaling
            bw_min = 50   # Below this = full red
            bw_max = 400  # Above this = full pink/white

            for formant_key, items in self._formant_items.items():
                formant_vals = f.formants[formant_key]
                valid = ~np.isnan(formant_vals)

                if np.any(valid):
                    times_valid = f.times[valid]
                    freqs_valid = formant_vals[valid]
                    size = items['size']

                    bw_key = bw_keys.get(formant_key)
                    if bw_key and bw_key in f.bandwidths:
                        bw_vals = f.bandwidths[bw_key][valid]
                        bw_vals = np.nan_to_num(bw_vals, nan=bw_max)

                        # Create colors: red (200,0,0) -> pink/white (255,180,180)
                        # Higher bandwidth = more white mixed in
                        t = np.clip((bw_vals - bw_min) / (bw_max - bw_min), 0, 1)
                        # Interpolate from red to pink
                        r = (200 + t * 55).astype(int)   # 200 -> 255
                        g = (0 + t * 180).astype(int)    # 0 -> 180
                        b = (0 + t * 180).astype(int)    # 0 -> 180

                        brushes = [pg.mkBrush(r[i], g[i], b[i], 255) for i in range(len(r))]
                        items['scatter'].setData(times_valid, freqs_valid, size=size, brush=brushes)
                    else:
                        # No bandwidth info - use default red
                        items['scatter'].setData(times_valid, freqs_valid, size=size)

                items['scatter'].show()
        else:
            for items in self._formant_items.values():
                items['scatter'].hide()

        # Center of Gravity
        if self._track_visibility['cog']:
            valid = ~np.isnan(f.cog)
            self._cog_curve.setData(f.times[valid], f.cog[valid])
            self._cog_curve.show()
        else:
            self._cog_curve.hide()

        # HNR
        if self._track_visibility['hnr'] and self._frequencies is not None:
            valid = ~np.isnan(f.hnr)
            f_start = self._freq_start
            f_end = self._freq_end
            freq_range = f_end - f_start
            scaled_hnr = (f.hnr + 10) / 50 * freq_range + f_start
            scaled_hnr = np.clip(scaled_hnr, f_start, f_end)
            self._hnr_curve.setData(f.times[valid], scaled_hnr[valid])
            self._hnr_curve.show()
        else:
            self._hnr_curve.hide()

    def set_track_visible(self, track_name: str, visible: bool):
        """Set visibility of an overlay track."""
        if track_name in self._track_visibility:
            self._track_visibility[track_name] = visible
            self._update_overlays()

    def set_colormap(self, name: str):
        """Change the spectrogram colormap."""
        if name in COLORMAPS:
            cmap = COLORMAPS[name]()
            self._spectrogram_img.setLookupTable(cmap.getLookupTable())

    def set_cursor_position(self, time: float):
        """Set playback cursor position."""
        self._cursor_time = time
        self._cursor_line.setPos(time)
        self.cursor_moved.emit(time)

    def get_view_range(self) -> tuple[float, float]:
        """Get current visible time range."""
        view_range = self._plot.viewRange()
        return (view_range[0][0], view_range[0][1])

    def set_x_range(self, start: float, end: float):
        """Set the visible time range."""
        self._plot.setXRange(start, end, padding=0)

    def set_selection(self, start: float, end: float):
        """Set the selection region."""
        self._selection_start = min(start, end)
        self._selection_end = max(start, end)
        self._selection_region.setRegion([self._selection_start, self._selection_end])
        self._selection_region.show()

    def clear_selection(self):
        """Clear the selection region."""
        self._selection_start = None
        self._selection_end = None
        self._selection_region.hide()

    def get_selection(self) -> tuple[float, float] | None:
        """Get current selection."""
        if self._selection_start is not None and self._selection_end is not None:
            return (self._selection_start, self._selection_end)
        return None

    def _on_x_range_changed(self):
        """Handle view range changes."""
        x_range = self._plot.viewRange()[0]
        self.time_range_changed.emit(x_range[0], x_range[1])

    def _on_selection_changed(self):
        """Handle selection region changes."""
        region = self._selection_region.getRegion()
        self._selection_start = region[0]
        self._selection_end = region[1]
        self.selection_changed.emit(region[0], region[1])

    def mousePressEvent(self, ev):
        """Handle mouse press for selection (Praat-like)."""
        if ev.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(ev.position().toPoint())
            view_pos = self._plot.vb.mapSceneToView(scene_pos)
            x = view_pos.x()
            self._click_start_pos = x
            self._click_inside_selection = False

            # Check if clicking inside existing selection
            if self._selection_region.isVisible():
                region = self._selection_region.getRegion()
                if region[0] <= x <= region[1]:
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
        """Handle mouse move for selection and info display."""
        scene_pos = self.mapToScene(ev.position().toPoint())
        view_pos = self._plot.vb.mapSceneToView(scene_pos)
        x, y = view_pos.x(), view_pos.y()

        # Update info label with acoustic values
        if self._features is not None:
            self._update_info_label(x, y)

        if self._is_dragging and self._selection_start is not None:
            self._selection_region.setRegion([self._selection_start, x])
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        """Handle mouse release for selection."""
        if ev.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(ev.position().toPoint())
            view_pos = self._plot.vb.mapSceneToView(scene_pos)
            x = view_pos.x()

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
                    # Single click - move cursor
                    cursor_pos = self._selection_start
                    self.clear_selection()
                    self.set_cursor_position(cursor_pos)
                else:
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
        if self._duration <= 0:
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
            if new_max > self._duration:
                new_min -= (new_max - self._duration)
                new_max = self._duration
                new_min = max(0, new_min)

            self._plot.setXRange(new_min, new_max, padding=0)

        # Vertical scroll = zoom only, centered on mouse position
        if delta_y != 0 and delta_x == 0:
            # Get mouse position in view coordinates
            scene_pos = self.mapToScene(ev.position().toPoint())
            view_pos = self._plot.vb.mapSceneToView(scene_pos)
            mouse_x = view_pos.x()

            # Clamp mouse position to valid range
            mouse_x = max(x_min, min(x_max, mouse_x))

            if delta_y > 0:
                factor = 0.9  # Zoom in
            else:
                factor = 1.1  # Zoom out

            # Calculate new range keeping mouse position fixed
            left_frac = (mouse_x - x_min) / x_range
            new_range = x_range * factor

            new_min = mouse_x - left_frac * new_range
            new_max = mouse_x + (1 - left_frac) * new_range

            # Clamp to audio bounds
            if new_min < 0:
                new_max -= new_min
                new_min = 0
            if new_max > self._duration:
                new_min -= (new_max - self._duration)
                new_max = self._duration
                new_min = max(0, new_min)

            self._plot.setXRange(new_min, new_max, padding=0)

        ev.accept()

    def _get_acoustic_values_at_time(self, time: float) -> dict:
        """Get acoustic feature values at a specific time."""
        values = {}

        if self._features is None or self._times is None:
            return values

        f = self._features

        # Find the nearest time index
        time_idx = np.searchsorted(f.times, time)
        if time_idx >= len(f.times):
            time_idx = len(f.times) - 1
        if time_idx > 0 and abs(f.times[time_idx - 1] - time) < abs(f.times[time_idx] - time):
            time_idx -= 1

        # Get values if within reasonable range
        if abs(f.times[time_idx] - time) < 0.05:  # Within 50ms
            # Pitch
            if not np.isnan(f.f0[time_idx]):
                values['Pitch'] = f"{f.f0[time_idx]:.1f} Hz"

            # Intensity
            if not np.isnan(f.intensity[time_idx]):
                values['Intensity'] = f"{f.intensity[time_idx]:.1f} dB"

            # Formants
            for key in ['F1', 'F2', 'F3', 'F4']:
                if key in f.formants and not np.isnan(f.formants[key][time_idx]):
                    values[key] = f"{f.formants[key][time_idx]:.0f} Hz"

            # HNR
            if not np.isnan(f.hnr[time_idx]):
                values['HNR'] = f"{f.hnr[time_idx]:.1f} dB"

            # CoG
            if not np.isnan(f.cog[time_idx]):
                values['CoG'] = f"{f.cog[time_idx]:.0f} Hz"

        return values

    def _update_info_label(self, time: float, freq: float):
        """Update the info label with acoustic values."""
        # Hide if outside valid frequency range
        if freq < self._freq_start or freq > self._freq_end:
            self._info_label.hide()
            return

        lines = [f"Time: {time:.3f}s", f"Freq: {freq:.0f} Hz"]

        values = self._get_acoustic_values_at_time(time)
        for key, val in values.items():
            lines.append(f"{key}: {val}")

        if len(lines) > 2:  # Only show if we have acoustic data
            self._info_label.setText("\n".join(lines))
            # Position in top-right of view
            view_range = self._plot.viewRange()
            x_max = view_range[0][1]
            y_max = view_range[1][1]
            self._info_label.setPos(x_max, y_max)
            self._info_label.setAnchor((1, 0))  # Right-top anchor
            self._info_label.show()
        else:
            self._info_label.hide()

    def leaveEvent(self, ev):
        """Hide info label when mouse leaves widget."""
        self._info_label.hide()
        super().leaveEvent(ev)
