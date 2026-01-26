"""
Spectrogram display widget with overlay tracks.

This module provides the SpectrogramWidget class for displaying spectrograms
with acoustic feature overlays. It's designed for speech analysis and
integrates with the rest of Ozen.

Features:
    - Time-frequency spectrogram display (computed externally)
    - Multiple colormap options (grayscale, viridis, inferno)
    - Overlay tracks for acoustic features:
        * Pitch (F0) - blue line with separate Y-axis
        * Formants (F1-F4) - red dots with bandwidth-based transparency
        * Intensity - yellow line
        * Center of Gravity (CoG) - green line
        * HNR (harmonics-to-noise) - purple dashed line
        * Spectral tilt - orange line
        * A1-P0 nasal ratio - cyan line
        * Nasal murmur ratio - magenta dashed line
    - Hover tooltip showing acoustic values at cursor
    - Synchronized view with waveform and annotation editor
    - Selection region for playback

Architecture:
    The widget inherits from pyqtgraph's GraphicsLayoutWidget to support
    multiple plot areas (main spectrogram + pitch axis). Acoustic overlays
    are rendered as PlotCurveItem objects that scale to the spectrogram's
    frequency range.

Signals:
    time_range_changed(start, end): Emitted when the visible range changes
    cursor_moved(time): Emitted when cursor position changes
    selection_changed(start, end): Emitted when selection changes
    selection_clicked(): Emitted when user clicks inside selection
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal, Qt, QRectF, QPointF
from PyQt6.QtGui import QColor, QTransform, QFont, QFontDatabase, QPainterPath, QPolygonF
from PyQt6.QtWidgets import QGraphicsItem, QMenu

from ..analysis.acoustic import AcousticFeatures
from ..config import config
from .data_points import DataPoint, DataPointCollection


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


class PlayButtonItem(QGraphicsItem):
    """Play button triangle drawn in pixel coordinates, matching annotation editor style."""

    # Button size in pixels
    WIDTH = 12
    HEIGHT = 14

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hovered = False
        self._color = QColor(0, 120, 0, 230)
        self._hover_color = QColor(0, 180, 0, 255)
        self._border_color = QColor(0, 80, 0, 255)
        self.setZValue(2000)
        self.hide()

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.WIDTH, self.HEIGHT)

    def paint(self, painter, option, widget=None):
        # Draw right-pointing triangle, same style as annotation editor
        triangle = QPolygonF([
            QPointF(0, 0),                          # Top left
            QPointF(0, self.HEIGHT),                # Bottom left
            QPointF(self.WIDTH, self.HEIGHT / 2),  # Right point
        ])

        if self._hovered:
            painter.setBrush(self._hover_color)
        else:
            painter.setBrush(self._color)
        painter.setPen(pg.mkPen(color=self._border_color, width=1))
        painter.drawPolygon(triangle)

    def set_hovered(self, hovered: bool):
        if self._hovered != hovered:
            self._hovered = hovered
            self.update()

    def contains(self, point: QPointF) -> bool:
        return self.boundingRect().contains(point)


class DataPointItem(QGraphicsItem):
    """Visual representation of a data collection point.

    Displays as a vertical line spanning the frequency range with a
    circle marker at the clicked frequency position. The color changes
    when hovered.
    """

    def __init__(self, data_point: DataPoint, view_box, freq_range: tuple[float, float], parent=None):
        super().__init__(parent)
        self._data_point = data_point
        self._view_box = view_box
        self._freq_start, self._freq_end = freq_range
        self._hovered = False

        colors = config['colors']
        self._color = QColor(*colors['data_point'][:4])
        self._hover_color = QColor(*colors['data_point_hover'][:4])
        self._line_width = colors['data_point_line_width']
        self._marker_size = colors['data_point_marker_size']

        self.setZValue(1500)  # Above spectrogram, below cursor
        self.setAcceptHoverEvents(True)

    @property
    def data_point(self) -> DataPoint:
        return self._data_point

    @property
    def point_id(self) -> int:
        return self._data_point.id

    def update_freq_range(self, freq_start: float, freq_end: float):
        """Update the frequency range for proper positioning."""
        self._freq_start = freq_start
        self._freq_end = freq_end
        self.prepareGeometryChange()
        self.update()

    def boundingRect(self) -> QRectF:
        # Return a rect in scene coordinates that covers the line and marker
        # The line spans the full frequency range at a specific time
        time = self._data_point.time
        # Add some padding for the marker
        marker_padding = 0.02  # In time units
        return QRectF(
            time - marker_padding,
            self._freq_start,
            marker_padding * 2,
            self._freq_end - self._freq_start
        )

    def paint(self, painter, option, widget=None):
        time = self._data_point.time
        freq = self._data_point.frequency

        color = self._hover_color if self._hovered else self._color
        pen = pg.mkPen(color=color, width=self._line_width)
        painter.setPen(pen)

        # Draw vertical line spanning full frequency range
        painter.drawLine(
            QPointF(time, self._freq_start),
            QPointF(time, self._freq_end)
        )

        # Draw circle marker at the frequency position
        brush = pg.mkBrush(color)
        painter.setBrush(brush)

        # Calculate marker size in data coordinates
        # We need to convert pixel size to data coordinates
        view_rect = self._view_box.viewRect()
        if view_rect.width() > 0 and view_rect.height() > 0:
            # Get view size in pixels
            view_size = self._view_box.size()
            if view_size.width() > 0 and view_size.height() > 0:
                # Convert pixel size to data coordinates
                x_scale = view_rect.width() / view_size.width()
                y_scale = view_rect.height() / view_size.height()
                marker_x_radius = self._marker_size * x_scale / 2
                marker_y_radius = self._marker_size * y_scale / 2
            else:
                marker_x_radius = 0.01
                marker_y_radius = 50
        else:
            marker_x_radius = 0.01
            marker_y_radius = 50

        painter.drawEllipse(
            QPointF(time, freq),
            marker_x_radius,
            marker_y_radius
        )

    def set_hovered(self, hovered: bool):
        if self._hovered != hovered:
            self._hovered = hovered
            self.update()

    def set_position(self, time: float, frequency: float):
        """Update the point's position."""
        self._data_point.time = time
        self._data_point.frequency = frequency
        self.prepareGeometryChange()
        self.update()

    def contains_point(self, time: float, freq: float, time_tol: float = 0.02, freq_tol: float = 100) -> bool:
        """Check if a position is close to this data point."""
        return (
            abs(time - self._data_point.time) <= time_tol and
            abs(freq - self._data_point.frequency) <= freq_tol
        )


class SpectrogramWidget(pg.GraphicsLayoutWidget):
    """Widget for displaying spectrogram with acoustic overlays."""

    # Signals
    time_range_changed = pyqtSignal(float, float)
    cursor_moved = pyqtSignal(float)
    selection_changed = pyqtSignal(float, float)
    selection_clicked = pyqtSignal()  # emitted when user clicks inside selection

    # Data point signals
    point_added = pyqtSignal(object)  # DataPoint
    point_removed = pyqtSignal(object)  # DataPoint
    point_moved = pyqtSignal(object)  # DataPoint

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
        self._click_start_pos: float | None = None
        self._play_button_hovered: bool = False

        # Track visibility settings
        self._track_visibility = {
            'pitch': True,
            'formants': True,
            'intensity': False,
            'cog': False,
            'hnr': False,
            'spectral_tilt': False,
            'a1p0': False,  # A1-P0 nasal ratio (requires voicing)
            'nasal_murmur': False,  # Low-freq energy ratio
        }

        # Data collection points
        self._data_points = DataPointCollection()
        self._data_point_items: dict[int, DataPointItem] = {}  # point_id -> item
        self._hovered_point_id: int | None = None
        self._dragging_point_id: int | None = None
        self._drag_start_pos: tuple[float, float] | None = None

        # Callback to get annotation intervals at a time position
        self._annotation_provider: callable | None = None

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
        font_name = config['fonts']['monospace']
        font_size = config['fonts']['monospace_size']
        if font_name:
            font = QFont(font_name, font_size)
        else:
            font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
            font.setPointSize(font_size)
        self._info_label.setFont(font)
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
        # Pitch range from config
        self._pitch_min = float(config['pitch']['display_floor'])
        self._pitch_max = float(config['pitch']['display_ceiling'])

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
        """Setup selection region and play button."""
        colors = config['colors']
        self._selection_region = pg.LinearRegionItem(
            values=[0, 0],
            brush=pg.mkBrush(*colors['selection_fill']),
            pen=pg.mkPen(color=colors['selection_border'][:3], width=colors['selection_border_width']),
            movable=True
        )
        self._selection_region.hide()
        self._selection_region.sigRegionChanged.connect(self._on_selection_changed)
        self._plot.addItem(self._selection_region)

        # Play button (triangle) - appears on selection, lower left corner
        # Uses custom QGraphicsItem for pixel-perfect rendering like annotation editor
        self._play_button = PlayButtonItem()
        self._plot.scene().addItem(self._play_button)

    def _setup_overlays(self):
        """Setup overlay plot items with colors from config."""
        colors = config['colors']

        # Pitch track - scaled to frequency range for display
        # Use connect='finite' to break line at NaN (unvoiced) values
        self._pitch_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color=colors['pitch'][:3], width=colors['pitch_width']),
            connect='finite',
            name='Pitch'
        )
        self._plot.addItem(self._pitch_curve)

        # Intensity
        self._intensity_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color=colors['intensity'][:3], width=colors['intensity_width']),
            name='Intensity'
        )
        self._plot.addItem(self._intensity_curve)

        # Formants - color fades based on bandwidth (narrow=red, wide=pink)
        self._formant_items = {}
        formant_size = colors['formant_size']

        for formant_key in ['F1', 'F2', 'F3', 'F4']:
            scatter = pg.ScatterPlotItem(
                pen=None,
                brush=pg.mkBrush(*colors['formant']),
                size=formant_size,
                name=formant_key
            )
            self._plot.addItem(scatter)
            self._formant_items[formant_key] = {'scatter': scatter, 'size': formant_size}

        # Center of Gravity
        self._cog_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color=colors['cog'][:3], width=colors['cog_width']),
            name='CoG'
        )
        self._plot.addItem(self._cog_curve)

        # HNR (dashed line)
        self._hnr_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color=colors['hnr'][:3], width=colors['hnr_width'], style=Qt.PenStyle.DashLine),
            name='HNR'
        )
        self._plot.addItem(self._hnr_curve)

        # Spectral tilt
        self._tilt_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color=colors['spectral_tilt'][:3], width=colors['spectral_tilt_width']),
            name='Spectral Tilt'
        )
        self._plot.addItem(self._tilt_curve)

        # A1-P0 nasal ratio (requires voicing)
        self._a1p0_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color=colors['a1p0'][:3], width=colors['a1p0_width']),
            name='A1-P0'
        )
        self._plot.addItem(self._a1p0_curve)

        # Nasal murmur ratio (dashed line)
        self._nasal_murmur_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color=colors['nasal_murmur'][:3], width=colors['nasal_murmur_width'], style=Qt.PenStyle.DashLine),
            name='Nasal Murmur'
        )
        self._plot.addItem(self._nasal_murmur_curve)

    def _setup_cursor(self):
        """Setup playback cursor."""
        colors = config['colors']
        self._cursor_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen(color=colors['cursor'][:3], width=colors['cursor_width']),
            movable=False
        )
        self._cursor_line.setZValue(1000)  # Ensure cursor is always on top
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
        # Create tick values at nice pitch intervals based on range
        p_min = self._pitch_min
        p_max = self._pitch_max
        # Generate ticks: every 50 Hz up to 200, then every 100 Hz
        pitch_ticks = [t for t in [50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 800]
                       if p_min <= t <= p_max]

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

        # Pitch - scaled to display on spectrogram with configurable range
        # Keep NaN values so connect='finite' breaks the line at unvoiced frames
        if self._track_visibility['pitch'] and self._frequencies is not None:
            if len(f.f0) > 0:
                # Pitch range from config
                p_min = self._pitch_min
                p_max = self._pitch_max

                # Use stored frequency range for proper scaling
                f_start = self._freq_start
                f_end = self._freq_end
                freq_range = f_end - f_start

                # Map pitch to frequency axis for display (NaN values stay NaN)
                scaled_pitch = (f.f0 - p_min) / (p_max - p_min) * freq_range + f_start
                # Clip valid values to range (NaN stays NaN)
                scaled_pitch = np.clip(scaled_pitch, f_start, f_end)

                self._pitch_curve.setData(f.times, scaled_pitch)

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
            bw_keys = {'F1': 'B1', 'F2': 'B2', 'F3': 'B3', 'F4': 'B4'}
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

        # Spectral tilt (dB, typically -20 to +20 range)
        if self._track_visibility['spectral_tilt'] and self._frequencies is not None:
            valid = ~np.isnan(f.spectral_tilt)
            f_start = self._freq_start
            f_end = self._freq_end
            freq_range = f_end - f_start
            # Map spectral tilt (-20 to +40 dB) to frequency range
            scaled_tilt = (f.spectral_tilt + 20) / 60 * freq_range + f_start
            scaled_tilt = np.clip(scaled_tilt, f_start, f_end)
            self._tilt_curve.setData(f.times[valid], scaled_tilt[valid])
            self._tilt_curve.show()
        else:
            self._tilt_curve.hide()

        # A1-P0 nasal ratio (requires voicing/pitch detection)
        if self._track_visibility['a1p0'] and self._frequencies is not None:
            try:
                valid = ~np.isnan(f.nasal_ratio)
                if np.any(valid):
                    f_start = self._freq_start
                    f_end = self._freq_end
                    freq_range = f_end - f_start
                    # A1-P0 in dB: map -20 to +20 range
                    scaled_a1p0 = (f.nasal_ratio + 20) / 40 * freq_range + f_start
                    scaled_a1p0 = np.clip(scaled_a1p0, f_start, f_end)
                    self._a1p0_curve.setData(f.times[valid], scaled_a1p0[valid])
                    self._a1p0_curve.show()
                else:
                    self._a1p0_curve.hide()
            except AttributeError:
                self._a1p0_curve.hide()
        else:
            self._a1p0_curve.hide()

        # Nasal murmur ratio (low-freq energy / total energy)
        if self._track_visibility['nasal_murmur'] and self._frequencies is not None:
            valid = ~np.isnan(f.nasal_murmur_ratio)
            if np.any(valid):
                f_start = self._freq_start
                f_end = self._freq_end
                freq_range = f_end - f_start
                # Murmur ratio 0-1: direct mapping to frequency range
                scaled_murmur = f.nasal_murmur_ratio * freq_range + f_start
                scaled_murmur = np.clip(scaled_murmur, f_start, f_end)
                self._nasal_murmur_curve.setData(f.times[valid], scaled_murmur[valid])
                self._nasal_murmur_curve.show()
            else:
                self._nasal_murmur_curve.hide()
        else:
            self._nasal_murmur_curve.hide()

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

    # -------------------------------------------------------------------------
    # Data Collection Points
    # -------------------------------------------------------------------------

    def set_annotation_provider(self, provider: callable):
        """Set a callback to get annotation intervals at a time position.

        The callback should accept a time (float) and return a dict mapping
        tier names to interval text at that time.

        Args:
            provider: Callable[[float], dict[str, str]]
        """
        self._annotation_provider = provider

    def get_data_points(self) -> DataPointCollection:
        """Get the data point collection."""
        return self._data_points

    def _add_data_point(self, time: float, frequency: float):
        """Add a data point at the given position."""
        # Get acoustic values at this time
        acoustic_values = self._get_acoustic_values_at_time_raw(time)

        # Add the point to the collection (annotations are looked up at export time)
        point = self._data_points.add_point(
            time=time,
            frequency=frequency,
            acoustic_values=acoustic_values
        )

        # Create visual item
        self._create_data_point_item(point)

        # Emit signal
        self.point_added.emit(point)

    def _remove_data_point(self, point_id: int):
        """Remove a data point by ID."""
        point = self._data_points.remove_point(point_id)
        if point:
            # Remove visual item
            if point_id in self._data_point_items:
                item = self._data_point_items.pop(point_id)
                self._plot.removeItem(item)

            # Emit signal
            self.point_removed.emit(point)

    def _create_data_point_item(self, point: DataPoint):
        """Create a visual item for a data point."""
        item = DataPointItem(
            point,
            self._plot.vb,
            (self._freq_start, self._freq_end)
        )
        self._data_point_items[point.id] = item
        self._plot.addItem(item)

    def _get_data_point_at_position(self, time: float, freq: float) -> DataPoint | None:
        """Find a data point near the given position."""
        # Calculate tolerances based on current view
        view_range = self._plot.viewRange()
        time_range = view_range[0][1] - view_range[0][0]
        freq_range = view_range[1][1] - view_range[1][0]

        # 2% of view range as tolerance
        time_tol = time_range * 0.02
        freq_tol = freq_range * 0.05

        return self._data_points.get_point_at_position(time, freq, time_tol, freq_tol)

    def _update_data_point_hover(self, time: float, freq: float):
        """Update hover state of data points."""
        point = self._get_data_point_at_position(time, freq)
        new_hovered_id = point.id if point else None

        if new_hovered_id != self._hovered_point_id:
            # Update old hovered item
            if self._hovered_point_id is not None and self._hovered_point_id in self._data_point_items:
                self._data_point_items[self._hovered_point_id].set_hovered(False)

            # Update new hovered item
            if new_hovered_id is not None and new_hovered_id in self._data_point_items:
                self._data_point_items[new_hovered_id].set_hovered(True)

            self._hovered_point_id = new_hovered_id

    def undo_data_point(self) -> bool:
        """Undo the last data point action.

        Returns:
            True if an action was undone, False otherwise
        """
        # Get the last action before undoing
        if not self._data_points._undo_stack:
            return False

        action_type, data = self._data_points._undo_stack[-1]

        result = self._data_points.undo()
        if result:
            if action_type == 'add':
                # Point was removed by undo - remove visual
                point = data['point']
                if point.id in self._data_point_items:
                    item = self._data_point_items.pop(point.id)
                    self._plot.removeItem(item)
                self.point_removed.emit(point)

            elif action_type == 'remove':
                # Point was re-added by undo - create visual
                point = data['point']
                self._create_data_point_item(point)
                self.point_added.emit(point)

            elif action_type == 'move':
                # Point was moved back - update visual
                point_id = data['point_id']
                if point_id in self._data_point_items:
                    item = self._data_point_items[point_id]
                    item.set_position(data['old_time'], data['old_frequency'])
                    point = self._data_points.get_point_by_id(point_id)
                    if point:
                        self.point_moved.emit(point)

        return result

    def clear_data_points(self):
        """Remove all data points."""
        for item in self._data_point_items.values():
            self._plot.removeItem(item)
        self._data_point_items.clear()
        self._data_points.clear()
        self._hovered_point_id = None
        self._dragging_point_id = None

    def _get_acoustic_values_at_time_raw(self, time: float) -> dict:
        """Get raw acoustic values at a time (numeric only, for export)."""
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
                values['Pitch'] = round(f.f0[time_idx], 1)

            # Intensity
            if not np.isnan(f.intensity[time_idx]):
                values['Intensity'] = round(f.intensity[time_idx], 1)

            # Formants
            for key in ['F1', 'F2', 'F3', 'F4']:
                if key in f.formants and not np.isnan(f.formants[key][time_idx]):
                    values[key] = round(f.formants[key][time_idx], 0)

            # HNR
            if not np.isnan(f.hnr[time_idx]):
                values['HNR'] = round(f.hnr[time_idx], 1)

            # CoG
            if not np.isnan(f.cog[time_idx]):
                values['CoG'] = round(f.cog[time_idx], 0)

            # Spectral tilt
            if not np.isnan(f.spectral_tilt[time_idx]):
                values['Tilt'] = round(f.spectral_tilt[time_idx], 1)

            # Nasal ratio (A1-P0)
            try:
                if not np.isnan(f.nasal_ratio[time_idx]):
                    values['A1-P0'] = round(f.nasal_ratio[time_idx], 1)
            except AttributeError:
                pass

            # Nasal murmur ratio
            if not np.isnan(f.nasal_murmur_ratio[time_idx]):
                values['Nasal'] = round(f.nasal_murmur_ratio[time_idx], 3)

        return values

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
        self._update_play_button()

    def clear_selection(self):
        """Clear the selection region."""
        self._selection_start = None
        self._selection_end = None
        self._selection_region.hide()
        self._play_button.hide()

    def _update_play_button(self):
        """Update play button position to lower left of selection."""
        if self._selection_start is None or self._selection_end is None:
            self._play_button.hide()
            return
        # Convert selection start to scene coordinates
        view_box = self._plot.vb
        y_range = self._plot.viewRange()[1]
        # Position at lower left of selection, with small pixel margin
        data_point = QPointF(self._selection_start, y_range[0])
        scene_point = view_box.mapViewToScene(data_point)
        # Add pixel margins (matching annotation editor style)
        margin_x = 6  # pixels from left edge of selection
        margin_y = 4  # pixels from bottom
        self._play_button.setPos(scene_point.x() + margin_x,
                                  scene_point.y() - PlayButtonItem.HEIGHT - margin_y)
        self._play_button.show()

    def _is_over_play_button(self, x: float, y: float) -> bool:
        """Check if data coordinates are over the play button."""
        if not self._play_button.isVisible():
            return False
        # Convert data coords to scene coords
        view_box = self._plot.vb
        scene_point = view_box.mapViewToScene(QPointF(x, y))
        # Check if within button bounds
        btn_rect = self._play_button.sceneBoundingRect()
        return btn_rect.contains(scene_point)

    def get_selection(self) -> tuple[float, float] | None:
        """Get current selection."""
        if self._selection_start is not None and self._selection_end is not None:
            return (self._selection_start, self._selection_end)
        return None

    def _on_x_range_changed(self):
        """Handle view range changes."""
        x_range = self._plot.viewRange()[0]
        self._update_play_button()  # Reposition play button for new view
        self.time_range_changed.emit(x_range[0], x_range[1])

    def _on_selection_changed(self):
        """Handle selection region changes."""
        region = self._selection_region.getRegion()
        self._selection_start = region[0]
        self._selection_end = region[1]
        self._update_play_button()
        self.selection_changed.emit(region[0], region[1])

    def mousePressEvent(self, ev):
        """Handle mouse press for selection and data point dragging."""
        if ev.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(ev.position().toPoint())
            view_pos = self._plot.vb.mapSceneToView(scene_pos)
            x, y = view_pos.x(), view_pos.y()
            self._click_start_pos = x

            # Check if clicking on play button
            if self._is_over_play_button(x, y):
                self.selection_clicked.emit()
                ev.accept()
                return

            # Check if clicking on a data point for dragging
            point = self._get_data_point_at_position(x, y)
            if point:
                self._dragging_point_id = point.id
                self._drag_start_pos = (point.time, point.frequency)
                ev.accept()
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
        """Handle mouse move for selection, cursor tracking, data points, and info display."""
        scene_pos = self.mapToScene(ev.position().toPoint())
        view_pos = self._plot.vb.mapSceneToView(scene_pos)
        x, y = view_pos.x(), view_pos.y()

        # Update cursor position on hover
        if self._duration > 0:
            self._cursor_time = x
            self._cursor_line.setPos(x)
            self.cursor_moved.emit(x)

        # Update info label with acoustic values
        if self._features is not None:
            self._update_info_label(x, y)

        # Update play button hover state
        over_button = self._is_over_play_button(x, y)
        if over_button != self._play_button_hovered:
            self._play_button_hovered = over_button
            self._play_button.set_hovered(over_button)

        # Update data point hover state
        if not self._dragging_point_id:
            self._update_data_point_hover(x, y)

        # Handle data point dragging
        if self._dragging_point_id is not None:
            # Clamp to valid range
            x = max(0, min(self._duration, x))
            y = max(self._freq_start, min(self._freq_end, y))

            # Update visual position
            if self._dragging_point_id in self._data_point_items:
                self._data_point_items[self._dragging_point_id].set_position(x, y)
            ev.accept()
            return

        if self._is_dragging and self._selection_start is not None:
            self._selection_region.setRegion([self._selection_start, x])
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        """Handle mouse release for selection and data point dragging."""
        if ev.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(ev.position().toPoint())
            view_pos = self._plot.vb.mapSceneToView(scene_pos)
            x, y = view_pos.x(), view_pos.y()

            # Handle data point drag release
            if self._dragging_point_id is not None:
                # Clamp to valid range
                x = max(0, min(self._duration, x))
                y = max(self._freq_start, min(self._freq_end, y))

                # Move the point in the collection (this adds to undo stack)
                self._data_points.move_point(self._dragging_point_id, x, y)

                # Update acoustic values at new position (annotations looked up at export)
                point = self._data_points.get_point_by_id(self._dragging_point_id)
                if point:
                    point.acoustic_values = self._get_acoustic_values_at_time_raw(x)
                    self.point_moved.emit(point)

                self._dragging_point_id = None
                self._drag_start_pos = None
                ev.accept()
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
                    self._update_play_button()
                    self.selection_changed.emit(self._selection_start, self._selection_end)
                ev.accept()
            else:
                super().mouseReleaseEvent(ev)
        else:
            super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        """Handle double-click to add data points."""
        if ev.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(ev.position().toPoint())
            view_pos = self._plot.vb.mapSceneToView(scene_pos)
            x, y = view_pos.x(), view_pos.y()

            # Check if within valid range
            if 0 <= x <= self._duration and self._freq_start <= y <= self._freq_end:
                self._add_data_point(x, y)
                ev.accept()
                return

        super().mouseDoubleClickEvent(ev)

    def contextMenuEvent(self, ev):
        """Handle right-click context menu for data points."""
        scene_pos = self.mapToScene(ev.pos())
        view_pos = self._plot.vb.mapSceneToView(scene_pos)
        x, y = view_pos.x(), view_pos.y()

        # Check if right-clicking on a data point
        point = self._get_data_point_at_position(x, y)
        if point:
            menu = QMenu(self)
            remove_action = menu.addAction("Remove")

            action = menu.exec(ev.globalPos())
            if action == remove_action:
                self._remove_data_point(point.id)
            ev.accept()
            return

        super().contextMenuEvent(ev)

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

            # Formants with bandwidths
            for key in ['F1', 'F2', 'F3', 'F4']:
                if key in f.formants and not np.isnan(f.formants[key][time_idx]):
                    formant_val = f"{f.formants[key][time_idx]:.0f} Hz"
                    # Add bandwidth if available
                    bw_key = 'B' + key[1]  # F1 -> B1, etc.
                    if bw_key in f.bandwidths and not np.isnan(f.bandwidths[bw_key][time_idx]):
                        formant_val += f" (bw:{f.bandwidths[bw_key][time_idx]:.0f})"
                    values[key] = formant_val

            # HNR
            if not np.isnan(f.hnr[time_idx]):
                values['HNR'] = f"{f.hnr[time_idx]:.1f} dB"

            # CoG
            if not np.isnan(f.cog[time_idx]):
                values['CoG'] = f"{f.cog[time_idx]:.0f} Hz"

            # Spectral tilt
            if not np.isnan(f.spectral_tilt[time_idx]):
                values['Tilt'] = f"{f.spectral_tilt[time_idx]:.1f} dB"

            # Nasal ratio (A1-P0) - requires pitch detection
            try:
                if not np.isnan(f.nasal_ratio[time_idx]):
                    values['A1-P0'] = f"{f.nasal_ratio[time_idx]:.1f} dB"
            except AttributeError:
                pass  # Old features without nasal_ratio

            # Nasal murmur ratio (low-freq energy ratio)
            if not np.isnan(f.nasal_murmur_ratio[time_idx]):
                values['Nasal'] = f"{f.nasal_murmur_ratio[time_idx]:.2f}"

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
