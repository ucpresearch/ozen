"""
Annotation editor widget for displaying and editing annotation tiers.

This module provides the visual interface for creating and editing
annotation tiers (similar to Praat's TextGrid editor). It consists of:

    TierItem: A pyqtgraph GraphicsObject that renders a single tier
        as a horizontal band with interval boundaries and labels.

    AnnotationEditorWidget: A pyqtgraph GraphicsLayoutWidget that manages
        multiple TierItem objects and handles user interaction.

Features:
    - Visual display of interval tiers with labels and durations
    - Click to select intervals, double-click to add boundaries
    - Drag boundaries to move them
    - Inline text editing for interval labels
    - Play buttons for individual intervals
    - Keyboard shortcuts (Enter to add boundary, Delete to remove)
    - Undo support for boundary and text changes
    - Synchronized cursor with waveform/spectrogram views

Architecture:
    The editor inherits from pg.GraphicsLayoutWidget and uses custom
    GraphicsObjects (TierItem) for rendering. This allows efficient
    zooming/panning while maintaining crisp text rendering. Mouse events
    are handled at the widget level and dispatched to appropriate tier items.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal, Qt, QRectF, QPointF, QLineF, QEvent
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPolygonF
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QMenu

from .tier import Tier, Interval, AnnotationSet


class TierItem(pg.GraphicsObject):
    """
    A single annotation tier displayed as horizontal band with intervals.

    This GraphicsObject renders one tier of annotations, including:
    - Background color for the tier band
    - Vertical lines for interval boundaries
    - Text labels for each interval
    - Duration display for each interval
    - Play buttons for selected intervals
    - Visual highlighting for selected intervals and hovered boundaries

    The paint() method is optimized to only render intervals that are
    currently visible in the view, enabling smooth performance with
    many intervals.
    """

    # Tier height in pixels (approximately)
    TIER_HEIGHT = 60
    # Play button size
    PLAY_BUTTON_HEIGHT = 0.18  # fraction of tier height (Y axis)
    PLAY_BUTTON_WIDTH = 0.008  # width in seconds (X axis) - about 8ms

    def __init__(self, tier: Tier, y_pos: float, height: float, parent=None):
        super().__init__(parent)
        self.tier = tier
        self._y_pos = y_pos
        self._height = height
        self._selected_interval: int | None = None
        self._hovered_boundary: int | None = None
        self._hovered_play_button: int | None = None  # interval index of hovered play button

        # Colors
        self._bg_color = QColor(240, 240, 240)
        self._border_color = QColor(100, 100, 100)
        self._boundary_color = QColor(0, 100, 200)
        self._selected_color = QColor(200, 220, 255)
        self._text_color = QColor(0, 0, 0)
        self._label_bg_color = QColor(255, 255, 255, 200)
        self._play_button_color = QColor(0, 120, 0, 230)  # Dark green for play button
        self._play_button_hover_color = QColor(0, 180, 0, 255)  # Brighter on hover
        self._play_button_border_color = QColor(0, 80, 0, 255)  # Border color

        # Text items for interval labels (managed separately)
        self._text_items: list[pg.TextItem] = []
        # Text items for interval durations
        self._duration_items: list[pg.TextItem] = []
        # Text item for tier name
        self._name_item: pg.TextItem | None = None

    def boundingRect(self) -> QRectF:
        """Return the bounding rectangle of this tier."""
        # Use a very wide range for X since we don't know the time extent
        return QRectF(0, self._y_pos, 100000, self._height)

    def paint(self, painter: QPainter, option, widget=None):
        """Draw the tier with intervals and boundaries."""
        if self.tier is None:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get the view's visible range
        view = self.getViewBox()
        if view is None:
            return

        view_range = view.viewRange()
        x_min, x_max = view_range[0]

        # Draw background
        rect = QRectF(x_min, self._y_pos, x_max - x_min, self._height)
        painter.fillRect(rect, self._bg_color)

        # Draw intervals
        intervals = self.tier.get_intervals()
        font = QFont("Arial", 9)
        painter.setFont(font)

        for i, interval in enumerate(intervals):
            # Skip intervals outside view
            if interval.end < x_min or interval.start > x_max:
                continue

            # Interval rectangle
            int_rect = QRectF(
                interval.start,
                self._y_pos,
                interval.end - interval.start,
                self._height
            )

            # Draw selection highlight
            if i == self._selected_interval:
                painter.fillRect(int_rect, self._selected_color)

                # Draw play button (triangle) only for selected interval
                # Button width adapts to zoom level (1% of visible range, min 0.005s, max 0.025s)
                view_width = x_max - x_min
                btn_width = max(0.005, min(0.025, view_width * 0.01))
                btn_height = self._height * self.PLAY_BUTTON_HEIGHT
                btn_margin_y = btn_height * 0.2
                btn_margin_x = btn_width * 0.8
                btn_left = interval.start + btn_margin_x
                btn_bottom = self._y_pos + btn_margin_y
                btn_top = btn_bottom + btn_height

                # Only draw if interval is wide enough (at least 2x button width)
                if interval.end - interval.start > btn_width * 2:
                    # Create triangle pointing right
                    triangle = QPolygonF([
                        QPointF(btn_left, btn_top),
                        QPointF(btn_left, btn_bottom),
                        QPointF(btn_left + btn_width, btn_bottom + btn_height / 2),
                    ])

                    # Use hover color if hovered
                    if i == self._hovered_play_button:
                        painter.setBrush(self._play_button_hover_color)
                    else:
                        painter.setBrush(self._play_button_color)
                    # Draw with border for visibility
                    painter.setPen(pg.mkPen(color=self._play_button_border_color, width=1))
                    painter.drawPolygon(triangle)

            # Text is drawn using separate TextItems (see update_text_items)

        # Reset brush before drawing other elements
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # Draw boundaries
        painter.setPen(pg.mkPen(color=self._boundary_color, width=2))
        for i, boundary in enumerate(self.tier.boundaries):
            if x_min <= boundary <= x_max:
                # Highlight hovered boundary
                if i == self._hovered_boundary:
                    painter.setPen(pg.mkPen(color=(255, 100, 0), width=3))
                else:
                    painter.setPen(pg.mkPen(color=self._boundary_color, width=2))

                painter.drawLine(
                    QPointF(boundary, self._y_pos),
                    QPointF(boundary, self._y_pos + self._height)
                )

        # Draw tier border
        painter.setPen(pg.mkPen(color=self._border_color, width=1))
        painter.drawRect(rect)

        # Tier name is drawn using TextItem (see update_text_items)

    def set_selected_interval(self, index: int | None):
        """Set the selected interval."""
        self._selected_interval = index
        self.update()

    def set_hovered_boundary(self, index: int | None):
        """Set the hovered boundary for highlighting."""
        self._hovered_boundary = index
        self.update()

    def set_hovered_play_button(self, index: int | None):
        """Set the hovered play button for highlighting."""
        self._hovered_play_button = index
        self.update()

    def get_play_button_at(self, x: float, y: float, view_range: tuple[float, float] = None) -> int | None:
        """Check if the point (x, y) is over a play button. Returns interval index or None.

        Play button only exists for the selected interval.
        """
        if self.tier is None or self._selected_interval is None:
            return None

        intervals = self.tier.get_intervals()
        if self._selected_interval >= len(intervals):
            return None

        interval = intervals[self._selected_interval]

        # Calculate adaptive button width based on view range
        if view_range:
            view_width = view_range[1] - view_range[0]
            btn_width = max(0.005, min(0.025, view_width * 0.01))
        else:
            btn_width = self.PLAY_BUTTON_WIDTH

        btn_height = self._height * self.PLAY_BUTTON_HEIGHT
        btn_margin_y = btn_height * 0.2
        btn_margin_x = btn_width * 0.8  # Must match paint() method

        # Skip if interval too narrow
        if interval.end - interval.start <= btn_width * 2:
            return None

        btn_left = interval.start + btn_margin_x
        btn_right = btn_left + btn_width
        btn_bottom = self._y_pos + btn_margin_y
        btn_top = btn_bottom + btn_height

        if btn_left <= x <= btn_right and btn_bottom <= y <= btn_top:
            return self._selected_interval

        return None

    def update_text_items(self, plot_widget, view_range=None):
        """Update text items for interval labels, durations, and tier name."""
        # Remove old text items
        for item in self._text_items:
            plot_widget.removeItem(item)
        self._text_items.clear()

        for item in self._duration_items:
            plot_widget.removeItem(item)
        self._duration_items.clear()

        if self._name_item is not None:
            plot_widget.removeItem(self._name_item)
            self._name_item = None

        # Create tier name label at the left edge of visible area
        if view_range is not None:
            x_min = view_range[0]
            x_max = view_range[1]
        else:
            x_min = 0
            x_max = 1000

        name_item = pg.TextItem(
            text=self.tier.name,
            color=(0, 0, 0),
            anchor=(0, 0.5),  # Left-center anchor
            fill=pg.mkBrush(255, 255, 255, 200)  # White background
        )
        name_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        name_item.setPos(x_min + 0.01, self._y_pos + self._height - 10)
        plot_widget.addItem(name_item)
        self._name_item = name_item

        # Create text items for each interval
        intervals = self.tier.get_intervals()
        for i, interval in enumerate(intervals):
            # Skip intervals outside view
            if interval.end < x_min or interval.start > x_max:
                continue

            center_x = (interval.start + interval.end) / 2

            # Interval label (center)
            if interval.text:
                center_y = self._y_pos + self._height / 2
                text_item = pg.TextItem(
                    text=interval.text,
                    color=(0, 0, 0),
                    anchor=(0.5, 0.5)  # Center anchor
                )
                text_item.setPos(center_x, center_y)
                plot_widget.addItem(text_item)
                self._text_items.append(text_item)

            # Duration label (bottom) - small gray text
            duration = interval.end - interval.start
            if duration >= 0.01:  # Only show if >= 10ms
                duration_text = f"{duration*1000:.0f}ms" if duration < 1 else f"{duration:.2f}s"
                duration_item = pg.TextItem(
                    text=duration_text,
                    color=(120, 120, 120),  # Gray
                    anchor=(0.5, 1.0)  # Bottom-center anchor
                )
                duration_item.setFont(QFont("Arial", 7))
                duration_item.setPos(center_x, self._y_pos + 8)
                plot_widget.addItem(duration_item)
                self._duration_items.append(duration_item)


class AnnotationEditorWidget(pg.GraphicsLayoutWidget):
    """Widget for displaying and editing annotation tiers."""

    # Signals
    time_range_changed = pyqtSignal(float, float)  # (start, end)
    cursor_moved = pyqtSignal(float)  # time position
    selection_changed = pyqtSignal(float, float)  # (start, end)
    selection_clicked = pyqtSignal()  # emitted when user clicks inside selection

    # Annotation-specific signals
    boundary_added = pyqtSignal(int, float)  # (tier_index, time)
    boundary_removed = pyqtSignal(int, int)  # (tier_index, boundary_index)
    boundary_moved = pyqtSignal(int, int, float)  # (tier_index, boundary_index, new_time)
    interval_text_changed = pyqtSignal(int, int, str)  # (tier_index, interval_index, text) - per character
    text_edit_finished = pyqtSignal(int, int)  # (tier_index, interval_index) - when edit session ends
    interval_selected = pyqtSignal(int, int)  # (tier_index, interval_index)
    interval_play_requested = pyqtSignal(float, float)  # (start, end)

    # Constants
    TIER_HEIGHT = 60  # pixels per tier

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plot = None  # Will be created in _setup_plot

        self._annotations: AnnotationSet | None = None
        self._duration: float = 0.0
        self._cursor_time: float = 0.0
        self._selection_start: float | None = None
        self._selection_end: float | None = None
        self._is_dragging: bool = False
        self._dragging_boundary: tuple[int, int] | None = None  # (tier_idx, boundary_idx)
        self._click_start_pos: float | None = None

        # Selected interval for editing
        self._selected_tier_idx: int | None = None
        self._selected_interval_idx: int | None = None

        # Currently hovered tier and boundary (for keyboard operations like Delete)
        self._hovered_tier_idx: int | None = None
        self._hovered_boundary_idx: int | None = None

        self._tier_items: list[TierItem] = []

        self._setup_plot()
        self._setup_cursor()
        self._setup_selection()
        self._setup_text_editor()

        # Enable mouse tracking for hover cursor
        self.setMouseTracking(True)

        # Enable keyboard focus for text input
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Flag to prevent Enter from adding boundary after text editor closes
        self._ignore_next_enter = False

        # Undo stack: list of (action_type, tier_idx, data) tuples
        # action_type: 'add_boundary', 'remove_boundary', 'set_text'
        self._undo_stack: list[tuple] = []
        self._max_undo = 100  # Maximum number of undo steps

        # Track original text when editing starts (for undo)
        self._text_edit_original: str | None = None
        self._text_edit_tier_idx: int | None = None
        self._text_edit_interval_idx: int | None = None

    def _setup_text_editor(self):
        """Setup inline text editor for interval labels."""
        self._text_editor = QLineEdit(self)
        self._text_editor.setStyleSheet("""
            QLineEdit {
                background-color: white;
                border: 2px solid #0064C8;
                border-radius: 3px;
                padding: 2px 5px;
                font-size: 12px;
            }
        """)
        self._text_editor.hide()
        self._text_editor.returnPressed.connect(self._on_text_editor_enter)
        self._text_editor.textChanged.connect(self._on_text_editor_changed)
        # Finalize edit when focus is lost (clicking elsewhere)
        self._text_editor.installEventFilter(self)

    def _setup_plot(self):
        """Configure the plot appearance."""
        self.setBackground('w')

        # Create the plot inside the layout (like SpectrogramWidget)
        self._plot = self.addPlot(row=0, col=0)
        self._plot.showGrid(x=True, y=False, alpha=0.2)
        self._plot.setLabel('bottom', 'Time', units='s')

        # Hide Y axis labels (tiers are labeled internally)
        self._plot.getAxis('left').setWidth(70)
        self._plot.getAxis('left').setStyle(showValues=False)
        self._plot.getAxis('left').setTicks([])
        self._plot.setLabel('left', '')  # Clear any left axis label

        # Add right axis placeholder for alignment
        self._plot.showAxis('right')
        self._plot.getAxis('right').setWidth(70)
        self._plot.getAxis('right').setStyle(showValues=False)
        self._plot.getAxis('right').setTicks([])

        # Disable default mouse handling
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.vb.setDefaultPadding(0)

        # Disable pyqtgraph's menu and keyboard handling so we can handle text input
        self._plot.setMenuEnabled(False)
        self._plot.vb.setMenuEnabled(False)
        # Disable ViewBox keyboard shortcuts (like 'a' for autorange)
        self._plot.vb.state['enableMenu'] = False
        # Hide the autorange "A" button
        self._plot.hideButtons()

        # Connect view range changes
        self._plot.sigXRangeChanged.connect(self._on_x_range_changed)

    def _setup_cursor(self):
        """Setup the playback cursor line."""
        self._cursor_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen(color=(200, 0, 0), width=2),
            movable=False
        )
        self._cursor_line.setZValue(1000)  # Ensure cursor is always on top
        self._plot.addItem(self._cursor_line)

    def _setup_selection(self):
        """Setup the selection region."""
        self._selection_region = pg.LinearRegionItem(
            values=[0, 0],
            brush=pg.mkBrush(180, 180, 255, 80),
            pen=pg.mkPen(color=(80, 80, 180), width=1),
            movable=True
        )
        self._selection_region.hide()
        self._selection_region.sigRegionChanged.connect(self._on_selection_changed)
        self._plot.addItem(self._selection_region)

    def set_annotations(self, annotations: AnnotationSet):
        """Set the annotation set to display."""
        self._annotations = annotations
        self._duration = annotations.duration
        self._rebuild_tier_items()
        self._update_y_range()
        # Set X range if we have a valid duration
        if self._duration > 0:
            self._plot.setXRange(0, self._duration, padding=0)
        # Initial refresh to render tier labels
        self.refresh()

    def set_duration(self, duration: float):
        """Set the total duration (for display even without annotations)."""
        self._duration = duration
        if self._annotations:
            self._annotations.duration = duration
        self._plot.setXRange(0, duration, padding=0)

    def _rebuild_tier_items(self):
        """Rebuild the tier display items."""
        # Remove old items and their associated text items
        for item in self._tier_items:
            # Remove text items that were added to the plot
            for text_item in item._text_items:
                self._plot.removeItem(text_item)
            for duration_item in item._duration_items:
                self._plot.removeItem(duration_item)
            if item._name_item is not None:
                self._plot.removeItem(item._name_item)
            # Remove the tier item itself
            self._plot.removeItem(item)
        self._tier_items.clear()

        if self._annotations is None:
            return

        # Create new tier items from bottom to top (tier 0 at top)
        num_tiers = self._annotations.num_tiers
        for i, tier in enumerate(self._annotations.get_tiers()):
            y_pos = (num_tiers - 1 - i) * self.TIER_HEIGHT
            item = TierItem(tier, y_pos, self.TIER_HEIGHT)
            self._tier_items.append(item)
            self._plot.addItem(item)

    def _update_y_range(self):
        """Update the Y axis range based on number of tiers."""
        if self._annotations is None or self._annotations.num_tiers == 0:
            self._plot.setYRange(0, self.TIER_HEIGHT, padding=0)
        else:
            total_height = self._annotations.num_tiers * self.TIER_HEIGHT
            self._plot.setYRange(0, total_height, padding=0.02)

    def refresh(self):
        """Refresh the display after data changes."""
        view_range = self._plot.viewRange()[0]  # Get current X range
        for item in self._tier_items:
            item.prepareGeometryChange()  # Force geometry update
            item.update()
            item.update_text_items(self._plot, view_range)  # Update text labels
        # Also update the plot widget itself
        self.viewport().update()

    def set_cursor_position(self, time: float):
        """Set the playback cursor position."""
        self._cursor_time = time
        self._cursor_line.setPos(time)

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
        """Get current selection or None."""
        if self._selection_start is not None and self._selection_end is not None:
            return (self._selection_start, self._selection_end)
        return None

    def get_view_range(self) -> tuple[float, float]:
        """Get the current visible time range."""
        view_range = self._plot.viewRange()
        return (view_range[0][0], view_range[0][1])

    def set_x_range(self, start: float, end: float):
        """Set the visible time range."""
        self._plot.setXRange(start, end, padding=0)

    def _on_x_range_changed(self):
        """Handle view range changes."""
        x_range = self._plot.viewRange()[0]
        self.time_range_changed.emit(x_range[0], x_range[1])
        # Redraw tier items with new view and update tier names position
        for item in self._tier_items:
            item.update()
            item.update_text_items(self._plot, x_range)

    def _on_selection_changed(self):
        """Handle selection region changes."""
        region = self._selection_region.getRegion()
        self._selection_start = region[0]
        self._selection_end = region[1]
        self.selection_changed.emit(region[0], region[1])

    def _get_tier_at_y(self, y: float) -> int | None:
        """Get the tier index at the given Y position."""
        if self._annotations is None:
            return None

        num_tiers = self._annotations.num_tiers
        tier_idx = num_tiers - 1 - int(y / self.TIER_HEIGHT)

        if 0 <= tier_idx < num_tiers:
            return tier_idx
        return None

    # Snap-to-grid constants
    SNAP_THRESHOLD = 0.015  # 15ms snap threshold

    def _get_snap_position(self, tier_idx: int, time: float) -> float:
        """Get snapped position if near a boundary in upper tiers.

        Returns the snapped time if within threshold of an upper tier boundary,
        otherwise returns the original time.
        """
        if self._annotations is None or tier_idx <= 0:
            return time

        # Check boundaries in all upper tiers (lower indices)
        for upper_idx in range(tier_idx):
            upper_tier = self._annotations.get_tier(upper_idx)
            for boundary in upper_tier.boundaries:
                if abs(boundary - time) <= self.SNAP_THRESHOLD:
                    return boundary

        return time

    def _find_boundary_near(self, tier_idx: int, time: float, tolerance: float = 0.01) -> int | None:
        """Find boundary index near the given time."""
        if self._annotations is None:
            return None

        tier = self._annotations.get_tier(tier_idx)
        for i, boundary in enumerate(tier.boundaries):
            if abs(boundary - time) <= tolerance:
                return i
        return None

    def mousePressEvent(self, ev):
        """Handle mouse press - single click selects interval or drags boundary."""
        if ev.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(ev.position().toPoint())
            pos = self._plot.vb.mapSceneToView(scene_pos)
            x, y = pos.x(), pos.y()
            self._click_start_pos = x

            tier_idx = self._get_tier_at_y(y)

            if tier_idx is not None and self._annotations:
                # Check if clicking on a play button
                if tier_idx < len(self._tier_items):
                    view_range = self.get_view_range()
                    play_btn_interval = self._tier_items[tier_idx].get_play_button_at(x, y, view_range)
                    if play_btn_interval is not None:
                        # Play this interval
                        tier = self._annotations.get_tier(tier_idx)
                        interval = tier.get_interval(play_btn_interval)
                        self.interval_play_requested.emit(interval.start, interval.end)
                        ev.accept()
                        return

                # Check if clicking near a boundary (for dragging)
                boundary_idx = self._find_boundary_near(tier_idx, x)
                if boundary_idx is not None:
                    self._dragging_boundary = (tier_idx, boundary_idx)
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                    ev.accept()
                    return

                # Single click on tier = select interval
                tier = self._annotations.get_tier(tier_idx)
                try:
                    interval_idx, interval = tier.get_interval_at_time(x)

                    # Clear previous selection
                    self._clear_interval_selection()

                    # Select this interval
                    self._selected_tier_idx = tier_idx
                    self._selected_interval_idx = interval_idx

                    # Highlight in the tier item
                    if tier_idx < len(self._tier_items):
                        self._tier_items[tier_idx].set_selected_interval(interval_idx)

                    # Set selection region to match interval
                    self._selection_start = interval.start
                    self._selection_end = interval.end
                    self._selection_region.setRegion([interval.start, interval.end])
                    self._selection_region.show()

                    # Emit signals
                    self.interval_selected.emit(tier_idx, interval_idx)
                    self.selection_changed.emit(interval.start, interval.end)

                    # Show text editor for this interval
                    self._show_text_editor()

                except ValueError:
                    pass

            # Update cursor position
            self._cursor_time = x
            self._cursor_line.setPos(x)
            self.cursor_moved.emit(x)
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        """Handle mouse move - cursor follows mouse over tiers."""
        scene_pos = self.mapToScene(ev.position().toPoint())
        pos = self._plot.vb.mapSceneToView(scene_pos)
        x, y = pos.x(), pos.y()

        if self._dragging_boundary is not None:
            # Dragging a boundary
            tier_idx, boundary_idx = self._dragging_boundary
            tier = self._annotations.get_tier(tier_idx)

            # Snap to upper tier boundaries if close
            snapped_x = self._get_snap_position(tier_idx, x)

            if tier.move_boundary(boundary_idx, snapped_x):
                self.refresh()
                self.boundary_moved.emit(tier_idx, boundary_idx, x)

                # Update selection region if there's a selected interval on this tier
                if (self._selected_tier_idx == tier_idx and
                    self._selected_interval_idx is not None):
                    try:
                        interval = tier.get_interval(self._selected_interval_idx)
                        self._selection_start = interval.start
                        self._selection_end = interval.end
                        self._selection_region.setRegion([interval.start, interval.end])
                        self.selection_changed.emit(interval.start, interval.end)
                    except (IndexError, ValueError):
                        pass

            # Update cursor position to follow the drag
            self._cursor_time = x
            self._cursor_line.setPos(x)
            self.cursor_moved.emit(x)

            ev.accept()
            return

        # Update cursor line to follow mouse when hovering over tiers
        tier_idx = self._get_tier_at_y(y)
        self._hovered_tier_idx = tier_idx  # Track hovered tier for keyboard operations
        if tier_idx is not None and self._annotations:
            # Show cursor at mouse position and notify other widgets
            self._cursor_time = x
            self._cursor_line.setPos(x)
            self.cursor_moved.emit(x)  # Sync cursor with waveform/spectrogram

            # Check for play button hover
            if tier_idx < len(self._tier_items):
                view_range = self.get_view_range()
                play_btn_interval = self._tier_items[tier_idx].get_play_button_at(x, y, view_range)
                if play_btn_interval is not None:
                    self.setCursor(Qt.CursorShape.PointingHandCursor)
                    self._tier_items[tier_idx].set_hovered_play_button(play_btn_interval)
                    self._tier_items[tier_idx].set_hovered_boundary(None)
                    self._hovered_boundary_idx = None
                else:
                    self._tier_items[tier_idx].set_hovered_play_button(None)

                    # Check for boundary proximity for drag cursor
                    boundary_idx = self._find_boundary_near(tier_idx, x)
                    if boundary_idx is not None:
                        self.setCursor(Qt.CursorShape.SizeHorCursor)
                        self._tier_items[tier_idx].set_hovered_boundary(boundary_idx)
                        self._hovered_boundary_idx = boundary_idx
                    else:
                        self.setCursor(Qt.CursorShape.CrossCursor)
                        self._tier_items[tier_idx].set_hovered_boundary(None)
                        self._hovered_boundary_idx = None

            # Clear hover state on other tiers
            for i, item in enumerate(self._tier_items):
                if i != tier_idx:
                    item.set_hovered_boundary(None)
                    item.set_hovered_play_button(None)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._hovered_boundary_idx = None
            # Clear hovered state on all tiers
            for item in self._tier_items:
                item.set_hovered_boundary(None)
                item.set_hovered_play_button(None)

        ev.accept()

    def mouseReleaseEvent(self, ev):
        """Handle mouse release."""
        if ev.button() == Qt.MouseButton.LeftButton:
            if self._dragging_boundary is not None:
                self._dragging_boundary = None
                self.setCursor(Qt.CursorShape.CrossCursor)
                ev.accept()
                return
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)

    def enterEvent(self, ev):
        """Grab focus when mouse enters so keyboard shortcuts work."""
        self.setFocus()
        super().enterEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        """Handle double-click to add a boundary."""
        if ev.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(ev.position().toPoint())
            pos = self._plot.vb.mapSceneToView(scene_pos)
            x, y = pos.x(), pos.y()

            tier_idx = self._get_tier_at_y(y)
            if tier_idx is not None and self._annotations:
                # Double click = add boundary at this position
                tier = self._annotations.get_tier(tier_idx)

                # Snap to upper tier boundaries if close
                snapped_x = self._get_snap_position(tier_idx, x)

                try:
                    tier.add_boundary(snapped_x)
                    self._push_undo('add_boundary', tier_idx, snapped_x)
                    self.refresh()
                    self.boundary_added.emit(tier_idx, snapped_x)

                    # Update selection to the interval at the click position
                    # (the interval that was split now has new bounds)
                    if self._selected_tier_idx == tier_idx:
                        try:
                            interval_idx, interval = tier.get_interval_at_time(snapped_x)
                            self._selected_interval_idx = interval_idx
                            self._selection_start = interval.start
                            self._selection_end = interval.end
                            self._selection_region.setRegion([interval.start, interval.end])
                            if tier_idx < len(self._tier_items):
                                self._tier_items[tier_idx].set_selected_interval(interval_idx)
                            self.selection_changed.emit(interval.start, interval.end)
                        except ValueError:
                            pass
                except ValueError:
                    pass  # Boundary already exists or outside range

            ev.accept()
        else:
            super().mouseDoubleClickEvent(ev)

    def contextMenuEvent(self, ev):
        """Handle right-click context menu for removing boundaries."""
        scene_pos = self.mapToScene(ev.pos())
        pos = self._plot.vb.mapSceneToView(scene_pos)
        x, y = pos.x(), pos.y()

        tier_idx = self._get_tier_at_y(y)
        if tier_idx is None or self._annotations is None:
            super().contextMenuEvent(ev)
            return

        tier = self._annotations.get_tier(tier_idx)

        # Check if near a boundary
        boundary_idx = self._find_boundary_near(tier_idx, x)
        if boundary_idx is not None:
            boundary_time = tier.boundaries[boundary_idx]

            menu = QMenu(self)
            remove_action = menu.addAction("Remove")

            action = menu.exec(ev.globalPos())
            if action == remove_action:
                self._push_undo('remove_boundary', tier_idx, boundary_time)
                tier.remove_boundary(boundary_idx)
                self._clear_interval_selection()
                self.refresh()
                self.boundary_removed.emit(tier_idx, boundary_time)

            ev.accept()
            return

        super().contextMenuEvent(ev)

    def _clear_interval_selection(self):
        """Clear the currently selected interval."""
        if self._selected_tier_idx is not None and self._selected_tier_idx < len(self._tier_items):
            self._tier_items[self._selected_tier_idx].set_selected_interval(None)
        self._selected_tier_idx = None
        self._selected_interval_idx = None

    def get_selected_interval(self) -> tuple[int, int, str] | None:
        """Get the currently selected interval info.

        Returns (tier_idx, interval_idx, current_text) or None.
        """
        if self._selected_tier_idx is None or self._selected_interval_idx is None:
            return None
        if self._annotations is None:
            return None
        try:
            tier = self._annotations.get_tier(self._selected_tier_idx)
            text = tier.get_interval_text(self._selected_interval_idx)
            return (self._selected_tier_idx, self._selected_interval_idx, text)
        except (IndexError, ValueError):
            return None

    def keyPressEvent(self, ev):
        """Handle keyboard shortcuts and text input for selected interval."""
        if self._annotations is None:
            super().keyPressEvent(ev)
            return

        # Ctrl+Z / Cmd+Z: don't handle here, let it propagate to main window
        if ev.key() == Qt.Key.Key_Z and (ev.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier)):
            ev.ignore()  # Explicitly ignore so it propagates
            return

        # If text editor is active, let it handle the event (don't add boundaries etc.)
        if self.is_editing_text():
            # Handle Escape to close editor
            if ev.key() == Qt.Key.Key_Escape:
                self._hide_text_editor()
                self._clear_interval_selection()
                self.clear_selection()
                self.refresh()
                ev.accept()
                return
            # Handle Delete to remove hovered boundary (even while editing text)
            if ev.key() == Qt.Key.Key_Delete and self._hovered_boundary_idx is not None:
                self._delete_hovered_boundary()
                ev.accept()
                return
            # All other keys are handled by the QLineEdit
            return

        # Check if we have a selected interval for text editing
        if self._selected_tier_idx is not None and self._selected_interval_idx is not None:
            tier = self._annotations.get_tier(self._selected_tier_idx)
            # Validate that the selected interval still exists
            intervals = tier.get_intervals()
            if self._selected_interval_idx >= len(intervals):
                # Selection is invalid, clear it
                self._clear_interval_selection()
                self.clear_selection()
                self.refresh()
                super().keyPressEvent(ev)
                return
            current_text = tier.get_interval_text(self._selected_interval_idx)

            # Handle text input
            if ev.key() == Qt.Key.Key_Backspace:
                # Delete last character
                if current_text:
                    new_text = current_text[:-1]
                    tier.set_interval_text(self._selected_interval_idx, new_text)
                    self.refresh()
                    self.interval_text_changed.emit(
                        self._selected_tier_idx, self._selected_interval_idx, new_text
                    )
                ev.accept()
                return
            elif ev.key() == Qt.Key.Key_Escape:
                # Deselect interval
                self._clear_interval_selection()
                self.clear_selection()
                self.refresh()
                ev.accept()
                return
            elif ev.key() == Qt.Key.Key_Return or ev.key() == Qt.Key.Key_Enter:
                # Confirm and deselect
                self._clear_interval_selection()
                self.refresh()
                ev.accept()
                return
            elif ev.key() == Qt.Key.Key_Space:
                # Play the selected interval
                interval = tier.get_interval(self._selected_interval_idx)
                self.interval_play_requested.emit(interval.start, interval.end)
                ev.accept()
                return
            elif ev.text() and ev.text().isprintable():
                # Add character to text
                new_text = current_text + ev.text()
                tier.set_interval_text(self._selected_interval_idx, new_text)
                self.refresh()
                self.interval_text_changed.emit(
                    self._selected_tier_idx, self._selected_interval_idx, new_text
                )
                ev.accept()
                return

        # No interval selected - handle other shortcuts

        # Enter/Return - add boundary at cursor (legacy, now double-click does this)
        if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            # Check if we should ignore this Enter (from text editor closing)
            if self._ignore_next_enter:
                self._ignore_next_enter = False
                ev.accept()
                return
            # Use hovered tier (where mouse is), then selected tier, then active tier
            if self._hovered_tier_idx is not None:
                tier_idx = self._hovered_tier_idx
            elif self._selected_tier_idx is not None:
                tier_idx = self._selected_tier_idx
            else:
                tier_idx = self._annotations.active_tier_index

            if tier_idx is not None and tier_idx < self._annotations.num_tiers:
                tier = self._annotations.get_tier(tier_idx)
                # Snap to upper tier boundaries if close
                snapped_time = self._get_snap_position(tier_idx, self._cursor_time)
                try:
                    tier.add_boundary(snapped_time)
                    self._push_undo('add_boundary', tier_idx, snapped_time)
                    self.refresh()
                    self.boundary_added.emit(tier_idx, snapped_time)
                except ValueError:
                    pass  # Boundary already exists or outside range
            ev.accept()

        # Delete key - remove the hovered boundary (the one that's highlighted)
        elif ev.key() == Qt.Key.Key_Delete:
            self._delete_hovered_boundary()
            ev.accept()

        # Number keys 1-5 - switch active tier
        elif Qt.Key.Key_1 <= ev.key() <= Qt.Key.Key_5:
            tier_num = ev.key() - Qt.Key.Key_1
            if tier_num < self._annotations.num_tiers:
                self._annotations.active_tier_index = tier_num
            ev.accept()

        else:
            super().keyPressEvent(ev)

    def wheelEvent(self, ev):
        """Handle scroll wheel: vertical = zoom, horizontal = pan."""
        x_min, x_max = self.get_view_range()
        x_range = x_max - x_min

        delta_x = ev.angleDelta().x()
        delta_y = ev.angleDelta().y()

        # Horizontal scroll = pan only (no zoom)
        if delta_x != 0:
            pan_amount = x_range * 0.05 * (-delta_x / 120.0)
            new_min = x_min + pan_amount
            new_max = x_max + pan_amount

            # Clamp to bounds
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
            pos = self._plot.vb.mapSceneToView(scene_pos)
            mouse_x = pos.x()

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

            # Clamp to bounds
            if new_min < 0:
                new_max -= new_min
                new_min = 0
            if new_max > self._duration:
                new_min -= (new_max - self._duration)
                new_max = self._duration
                new_min = max(0, new_min)

            self._plot.setXRange(new_min, new_max, padding=0)

        ev.accept()

    def add_boundary_at_cursor(self):
        """Add a boundary at the current cursor position."""
        if self._annotations is None:
            return

        tier = self._annotations.active_tier
        if tier is None:
            return

        try:
            tier.add_boundary(self._cursor_time)
            self._push_undo('add_boundary', self._annotations.active_tier_index, self._cursor_time)
            self.refresh()
            self.boundary_added.emit(
                self._annotations.active_tier_index,
                self._cursor_time
            )
        except ValueError:
            pass

    def remove_nearest_boundary(self):
        """Remove the boundary nearest to the cursor."""
        if self._annotations is None:
            return

        tier_idx = self._annotations.active_tier_index
        tier = self._annotations.active_tier
        if tier is None:
            return

        boundary_idx, _, _ = tier.find_nearest_boundary(self._cursor_time)
        if boundary_idx >= 0:
            tier.remove_boundary(boundary_idx)
            self.refresh()
            self.boundary_removed.emit(tier_idx, boundary_idx)

    def _delete_hovered_boundary(self):
        """Delete the currently hovered boundary (highlighted in orange)."""
        if self._annotations is None:
            return False

        if (self._hovered_tier_idx is None or
            self._hovered_boundary_idx is None or
            self._hovered_tier_idx >= self._annotations.num_tiers):
            return False

        tier_idx = self._hovered_tier_idx
        boundary_idx = self._hovered_boundary_idx
        tier = self._annotations.get_tier(tier_idx)
        # Save boundary time before removing for undo
        boundary_time = tier.boundaries[boundary_idx]
        tier.remove_boundary(boundary_idx)
        self._push_undo('remove_boundary', tier_idx, boundary_time)

        # Clear the hover state since boundary is gone
        self._hovered_boundary_idx = None
        if tier_idx < len(self._tier_items):
            self._tier_items[tier_idx].set_hovered_boundary(None)

        # Clear selection since interval bounds changed
        self._clear_interval_selection()
        self.clear_selection()
        self.refresh()
        self.boundary_removed.emit(tier_idx, boundary_idx)
        return True

    def _push_undo(self, action_type: str, tier_idx: int, data):
        """Push an action to the undo stack."""
        # When boundaries change, text undo entries become invalid (interval indices shift)
        if action_type in ('add_boundary', 'remove_boundary'):
            self._clear_text_undo_for_tier(tier_idx)

        self._undo_stack.append((action_type, tier_idx, data))
        # Limit stack size
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

    def _clear_text_undo_for_tier(self, tier_idx: int):
        """Remove text undo entries for a tier (called when boundaries change)."""
        self._undo_stack = [
            entry for entry in self._undo_stack
            if not (entry[0] == 'set_text' and entry[1] == tier_idx)
        ]

    def undo(self) -> bool:
        """Undo the last action. Returns True if an action was undone."""
        if not self._undo_stack or self._annotations is None:
            return False

        action_type, tier_idx, data = self._undo_stack.pop()
        tier = self._annotations.get_tier(tier_idx)

        if action_type == 'add_boundary':
            # Undo add boundary = remove it
            boundary_time = data
            # Find the boundary index
            for i, b in enumerate(tier.boundaries):
                if abs(b - boundary_time) < 0.001:
                    tier.remove_boundary(i)
                    break

        elif action_type == 'remove_boundary':
            # Undo remove boundary = add it back
            boundary_time = data
            try:
                tier.add_boundary(boundary_time)
            except ValueError:
                pass  # Boundary might already exist

        elif action_type == 'set_text':
            # Undo text change = restore old text
            interval_idx, old_text = data
            try:
                tier.set_interval_text(interval_idx, old_text)
            except (IndexError, ValueError):
                pass  # Interval might not exist anymore

        # Clear selection and refresh
        self._clear_interval_selection()
        self.clear_selection()
        self._hide_text_editor()
        self.refresh()
        return True

    def play_interval_at_cursor(self):
        """Request playback of the interval at the cursor."""
        if self._annotations is None:
            return

        tier = self._annotations.active_tier
        if tier is None:
            return

        try:
            _, interval = tier.get_interval_at_time(self._cursor_time)
            self.interval_play_requested.emit(interval.start, interval.end)
        except ValueError:
            pass

    def _show_text_editor(self):
        """Show the inline text editor for the selected interval."""
        if self._selected_tier_idx is None or self._selected_interval_idx is None:
            return
        if self._annotations is None:
            return

        try:
            tier = self._annotations.get_tier(self._selected_tier_idx)
            interval = tier.get_interval(self._selected_interval_idx)
            current_text = tier.get_interval_text(self._selected_interval_idx)

            # Save original text for undo
            self._text_edit_original = current_text
            self._text_edit_tier_idx = self._selected_tier_idx
            self._text_edit_interval_idx = self._selected_interval_idx

            # Get the position in widget coordinates
            # Map interval center from view coords to widget coords
            view = self._plot.vb
            center_x = (interval.start + interval.end) / 2
            tier_y = (self._annotations.num_tiers - 1 - self._selected_tier_idx) * self.TIER_HEIGHT + self.TIER_HEIGHT / 2

            # Map to scene then to widget coordinates
            scene_pos = view.mapViewToScene(QPointF(center_x, tier_y))
            widget_pos = self.mapFromScene(scene_pos)

            # Position and show the editor
            editor_width = 150
            editor_height = 25
            self._text_editor.setGeometry(
                int(widget_pos.x() - editor_width / 2),
                int(widget_pos.y() - editor_height / 2),
                editor_width,
                editor_height
            )
            self._text_editor.setText(current_text)
            self._text_editor.show()
            self._text_editor.setFocus()
            self._text_editor.selectAll()

        except (IndexError, ValueError):
            pass

    def _hide_text_editor(self):
        """Hide the inline text editor and restore focus to main widget."""
        # Check if text changed and push to undo stack
        text_changed = False
        tier_idx = self._text_edit_tier_idx
        interval_idx = self._text_edit_interval_idx

        if (self._text_edit_original is not None and
            tier_idx is not None and
            interval_idx is not None and
            self._annotations is not None):
            try:
                tier = self._annotations.get_tier(tier_idx)
                current_text = tier.get_interval_text(interval_idx)
                if current_text != self._text_edit_original:
                    self._push_undo('set_text', tier_idx,
                                   (interval_idx, self._text_edit_original))
                    text_changed = True
            except (IndexError, ValueError):
                pass

        # Clear the tracking variables
        self._text_edit_original = None
        self._text_edit_tier_idx = None
        self._text_edit_interval_idx = None

        self._text_editor.hide()
        self.setFocus()  # Restore keyboard focus to the annotation editor

        # Emit signal after text edit is finalized (for global undo tracking)
        if text_changed and tier_idx is not None and interval_idx is not None:
            self.text_edit_finished.emit(tier_idx, interval_idx)

    def eventFilter(self, obj, event):
        """Handle events for child widgets."""
        # Handle text editor losing focus - finalize the edit
        if obj == self._text_editor and event.type() == QEvent.Type.FocusOut:
            if self._text_editor.isVisible():
                self._on_text_editor_return()  # Same as pressing Enter
            return False  # Don't block the event
        return super().eventFilter(obj, event)

    def _on_text_editor_enter(self):
        """Handle Enter key in text editor - confirm, close, and clear selection."""
        self._on_text_editor_return(clear_selection=True)

    def _on_text_editor_return(self, clear_selection: bool = False):
        """Handle text editor closing - confirm and close.

        Args:
            clear_selection: If True, also clear the interval selection.
                           Default is False to keep selection for play button clicks.
        """
        self._hide_text_editor()
        if clear_selection:
            self._clear_interval_selection()
        self.refresh()
        # Prevent the Enter key from also adding a boundary
        self._ignore_next_enter = True

    def _on_text_editor_changed(self, text: str):
        """Handle text changes in the editor - update interval immediately."""
        if self._selected_tier_idx is None or self._selected_interval_idx is None:
            return
        if self._annotations is None:
            return

        try:
            tier = self._annotations.get_tier(self._selected_tier_idx)
            tier.set_interval_text(self._selected_interval_idx, text)
            self.refresh()
            self.interval_text_changed.emit(
                self._selected_tier_idx, self._selected_interval_idx, text
            )
        except (IndexError, ValueError):
            pass

    def is_editing_text(self) -> bool:
        """Check if the text editor is currently visible/active."""
        return self._text_editor.isVisible()

    def add_char_to_selected_interval(self, char: str):
        """Add a character to the currently selected interval's text."""
        if self._selected_tier_idx is None or self._selected_interval_idx is None:
            return False
        if self._annotations is None:
            return False

        try:
            tier = self._annotations.get_tier(self._selected_tier_idx)
            current_text = tier.get_interval_text(self._selected_interval_idx)
            new_text = current_text + char
            tier.set_interval_text(self._selected_interval_idx, new_text)
            self.refresh()
            self.interval_text_changed.emit(
                self._selected_tier_idx, self._selected_interval_idx, new_text
            )
            return True
        except (IndexError, ValueError):
            return False

    def delete_char_from_selected_interval(self):
        """Delete the last character from the selected interval's text."""
        if self._selected_tier_idx is None or self._selected_interval_idx is None:
            return False
        if self._annotations is None:
            return False

        try:
            tier = self._annotations.get_tier(self._selected_tier_idx)
            current_text = tier.get_interval_text(self._selected_interval_idx)
            if current_text:
                new_text = current_text[:-1]
                tier.set_interval_text(self._selected_interval_idx, new_text)
                self.refresh()
                self.interval_text_changed.emit(
                    self._selected_tier_idx, self._selected_interval_idx, new_text
                )
            return True
        except (IndexError, ValueError):
            return False

    def confirm_selected_interval(self):
        """Confirm editing and deselect the current interval."""
        if self._selected_tier_idx is None or self._selected_interval_idx is None:
            return False

        self._hide_text_editor()
        self._clear_interval_selection()
        self.refresh()
        return True

    def deselect_interval(self):
        """Deselect the current interval and clear selection."""
        self._hide_text_editor()
        self._clear_interval_selection()
        self.clear_selection()
        self.refresh()

    def play_selected_interval(self):
        """Play the currently selected interval."""
        if self._selected_tier_idx is None or self._selected_interval_idx is None:
            return False
        if self._annotations is None:
            return False

        try:
            tier = self._annotations.get_tier(self._selected_tier_idx)
            interval = tier.get_interval(self._selected_interval_idx)
            self.interval_play_requested.emit(interval.start, interval.end)
            return True
        except (IndexError, ValueError):
            return False
