"""
Data collection points for spectrogram.

This module provides data models for collecting acoustic measurements
at specific time and frequency positions on the spectrogram.

Usage:
    from ozen.visualization.data_points import DataPoint, DataPointCollection

    # Create a collection
    points = DataPointCollection()

    # Add a point with acoustic values
    point = points.add_point(
        time=0.523,
        frequency=1450,
        acoustic_values={'Pitch': 125.3, 'F1': 720},
        annotation_intervals={'words': 'the', 'phones': 'DH AX'}
    )

    # Undo last action
    points.undo()

    # Export to TSV
    points.export_tsv('/path/to/output.tsv')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import csv


@dataclass
class DataPoint:
    """A single data collection point on the spectrogram.

    Attributes:
        id: Unique identifier for this point
        time: Time position in seconds
        frequency: Frequency position in Hz (Y position where user clicked)
        acoustic_values: Dict of acoustic measurements at this time
        annotation_intervals: Dict mapping tier name to interval text
    """
    id: int
    time: float
    frequency: float
    acoustic_values: dict = field(default_factory=dict)
    annotation_intervals: dict = field(default_factory=dict)


class DataPointCollection:
    """Collection of data points with undo support.

    This class manages a set of data collection points on the spectrogram.
    It provides methods for adding, removing, and moving points, with
    full undo support following the same pattern as the annotation editor.

    Signals are provided via callbacks to notify listeners of changes.
    """

    def __init__(self):
        self._points: list[DataPoint] = []
        self._next_id: int = 1
        self._undo_stack: list[tuple[str, dict]] = []  # (action_type, data)

        # Callbacks for change notifications
        self._on_point_added: Callable[[DataPoint], None] | None = None
        self._on_point_removed: Callable[[DataPoint], None] | None = None
        self._on_point_moved: Callable[[DataPoint], None] | None = None
        self._on_changed: Callable[[], None] | None = None

    @property
    def points(self) -> list[DataPoint]:
        """Get all data points."""
        return list(self._points)

    def set_callbacks(
        self,
        on_point_added: Callable[[DataPoint], None] | None = None,
        on_point_removed: Callable[[DataPoint], None] | None = None,
        on_point_moved: Callable[[DataPoint], None] | None = None,
        on_changed: Callable[[], None] | None = None
    ):
        """Set callbacks for change notifications."""
        self._on_point_added = on_point_added
        self._on_point_removed = on_point_removed
        self._on_point_moved = on_point_moved
        self._on_changed = on_changed

    def add_point(
        self,
        time: float,
        frequency: float,
        acoustic_values: dict | None = None,
        annotation_intervals: dict | None = None
    ) -> DataPoint:
        """Add a new data point.

        Args:
            time: Time position in seconds
            frequency: Frequency position in Hz
            acoustic_values: Dict of acoustic measurements
            annotation_intervals: Dict mapping tier name to interval text

        Returns:
            The newly created DataPoint
        """
        point = DataPoint(
            id=self._next_id,
            time=time,
            frequency=frequency,
            acoustic_values=acoustic_values or {},
            annotation_intervals=annotation_intervals or {}
        )
        self._next_id += 1
        self._points.append(point)

        # Push to undo stack
        self._undo_stack.append(('add', {'point': point}))

        # Notify listeners
        if self._on_point_added:
            self._on_point_added(point)
        if self._on_changed:
            self._on_changed()

        return point

    def remove_point(self, point_id: int) -> DataPoint | None:
        """Remove a data point by ID.

        Args:
            point_id: The ID of the point to remove

        Returns:
            The removed point, or None if not found
        """
        for i, point in enumerate(self._points):
            if point.id == point_id:
                removed = self._points.pop(i)

                # Push to undo stack
                self._undo_stack.append(('remove', {'point': removed, 'index': i}))

                # Notify listeners
                if self._on_point_removed:
                    self._on_point_removed(removed)
                if self._on_changed:
                    self._on_changed()

                return removed
        return None

    def move_point(self, point_id: int, new_time: float, new_frequency: float) -> bool:
        """Move a data point to a new position.

        Args:
            point_id: The ID of the point to move
            new_time: New time position in seconds
            new_frequency: New frequency position in Hz

        Returns:
            True if the point was found and moved, False otherwise
        """
        for point in self._points:
            if point.id == point_id:
                old_time = point.time
                old_frequency = point.frequency

                # Push to undo stack before modifying
                self._undo_stack.append(('move', {
                    'point_id': point_id,
                    'old_time': old_time,
                    'old_frequency': old_frequency,
                    'new_time': new_time,
                    'new_frequency': new_frequency
                }))

                point.time = new_time
                point.frequency = new_frequency

                # Notify listeners
                if self._on_point_moved:
                    self._on_point_moved(point)
                if self._on_changed:
                    self._on_changed()

                return True
        return False

    def update_point_data(
        self,
        point_id: int,
        acoustic_values: dict | None = None,
        annotation_intervals: dict | None = None
    ) -> bool:
        """Update the acoustic values and annotations for a point.

        Args:
            point_id: The ID of the point to update
            acoustic_values: New acoustic measurements
            annotation_intervals: New annotation intervals

        Returns:
            True if the point was found and updated, False otherwise
        """
        for point in self._points:
            if point.id == point_id:
                if acoustic_values is not None:
                    point.acoustic_values = acoustic_values
                if annotation_intervals is not None:
                    point.annotation_intervals = annotation_intervals
                return True
        return False

    def get_point_by_id(self, point_id: int) -> DataPoint | None:
        """Get a point by its ID."""
        for point in self._points:
            if point.id == point_id:
                return point
        return None

    def get_point_at_position(
        self,
        time: float,
        frequency: float,
        time_tolerance: float = 0.02,
        freq_tolerance: float = 100
    ) -> DataPoint | None:
        """Find a point near the given position.

        Args:
            time: Time position to search near
            frequency: Frequency position to search near
            time_tolerance: Maximum time difference in seconds
            freq_tolerance: Maximum frequency difference in Hz

        Returns:
            The closest point within tolerance, or None
        """
        best_point = None
        best_dist = float('inf')

        for point in self._points:
            time_diff = abs(point.time - time)
            freq_diff = abs(point.frequency - frequency)

            if time_diff <= time_tolerance and freq_diff <= freq_tolerance:
                # Use normalized distance
                dist = (time_diff / time_tolerance) ** 2 + (freq_diff / freq_tolerance) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_point = point

        return best_point

    def undo(self) -> bool:
        """Undo the last action.

        Returns:
            True if an action was undone, False if stack is empty
        """
        if not self._undo_stack:
            return False

        action_type, data = self._undo_stack.pop()

        if action_type == 'add':
            # Undo add by removing the point
            point = data['point']
            self._points = [p for p in self._points if p.id != point.id]
            if self._on_point_removed:
                self._on_point_removed(point)

        elif action_type == 'remove':
            # Undo remove by re-adding the point at original index
            point = data['point']
            index = data['index']
            self._points.insert(index, point)
            if self._on_point_added:
                self._on_point_added(point)

        elif action_type == 'move':
            # Undo move by restoring old position
            point_id = data['point_id']
            for point in self._points:
                if point.id == point_id:
                    point.time = data['old_time']
                    point.frequency = data['old_frequency']
                    if self._on_point_moved:
                        self._on_point_moved(point)
                    break

        if self._on_changed:
            self._on_changed()

        return True

    def clear(self):
        """Remove all points and clear undo stack."""
        self._points.clear()
        self._undo_stack.clear()
        self._next_id = 1
        if self._on_changed:
            self._on_changed()

    def export_tsv(
        self,
        file_path: str,
        tier_names: list[str] | None = None,
        annotation_provider: callable | None = None
    ):
        """Export points to a TSV file.

        Args:
            file_path: Path to the output file
            tier_names: List of tier names for column ordering (optional)
            annotation_provider: Optional callback(time) -> dict[str, str] to look up
                                 annotations at export time instead of using stored values
        """
        if not self._points:
            # Write empty file with header
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                f.write("time\tfrequency\n")
            return

        # Collect all acoustic measurement keys
        acoustic_keys = set()
        for point in self._points:
            acoustic_keys.update(point.acoustic_values.keys())
        acoustic_keys = sorted(acoustic_keys)

        # Use provided tier names or empty list
        if tier_names is None:
            tier_names = []

        # Write TSV
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')

            # Header
            header = ['time', 'frequency'] + list(acoustic_keys) + list(tier_names)
            writer.writerow(header)

            # Sort points by time
            sorted_points = sorted(self._points, key=lambda p: p.time)

            # Data rows
            for point in sorted_points:
                row = [
                    f"{point.time:.4f}",
                    f"{point.frequency:.1f}"
                ]

                # Acoustic values
                for key in acoustic_keys:
                    val = point.acoustic_values.get(key)
                    if val is not None:
                        # Strip units for numeric processing
                        if isinstance(val, str):
                            # Remove common units like "Hz", "dB"
                            val = val.replace(' Hz', '').replace(' dB', '').strip()
                            # Handle bandwidth notation like "720 (bw:100)"
                            if '(' in val:
                                val = val.split('(')[0].strip()
                        row.append(str(val))
                    else:
                        row.append('')

                # Annotation intervals - look up at export time if provider given
                if annotation_provider:
                    intervals = annotation_provider(point.time)
                    for tier in tier_names:
                        row.append(intervals.get(tier, ''))
                else:
                    for tier in tier_names:
                        row.append(point.annotation_intervals.get(tier, ''))

                writer.writerow(row)

    def import_tsv(self, file_path: str) -> int:
        """Import points from a TSV file.

        Args:
            file_path: Path to the input file

        Returns:
            Number of points imported (duplicates are skipped)
        """
        imported_count = 0

        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')

            for row in reader:
                try:
                    time = float(row.get('time', 0))
                    frequency = float(row.get('frequency', 0))

                    # Skip if a point already exists at this exact position
                    is_duplicate = any(
                        abs(p.time - time) < 1e-6 and abs(p.frequency - frequency) < 1e-6
                        for p in self._points
                    )
                    if is_duplicate:
                        continue

                    # Collect acoustic values (skip time, frequency, and annotation columns)
                    acoustic_values = {}
                    for key, val in row.items():
                        if key in ('time', 'frequency') or not val:
                            continue
                        # Try to parse as number
                        try:
                            acoustic_values[key] = float(val)
                        except ValueError:
                            # Keep as string (likely annotation)
                            pass

                    # Add point without triggering undo
                    point = DataPoint(
                        id=self._next_id,
                        time=time,
                        frequency=frequency,
                        acoustic_values=acoustic_values,
                        annotation_intervals={}
                    )
                    self._next_id += 1
                    self._points.append(point)
                    imported_count += 1

                    if self._on_point_added:
                        self._on_point_added(point)

                except (ValueError, KeyError):
                    continue  # Skip invalid rows

        if self._on_changed:
            self._on_changed()

        return imported_count
