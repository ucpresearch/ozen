"""Annotation tier data model."""

from dataclasses import dataclass, field
from typing import Optional
import bisect


@dataclass
class Interval:
    """An interval in an annotation tier."""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    text: str = ""  # Label/annotation text

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError(f"Interval start ({self.start}) must be <= end ({self.end})")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def contains(self, time: float) -> bool:
        """Check if time falls within this interval."""
        return self.start <= time < self.end

    def overlaps(self, other: 'Interval') -> bool:
        """Check if this interval overlaps with another."""
        return self.start < other.end and other.start < self.end


class Tier:
    """An annotation tier containing intervals defined by boundaries."""

    def __init__(self, name: str = "", start_time: float = 0.0, end_time: float = 0.0):
        self.name = name
        self._start_time = start_time
        self._end_time = end_time
        # Boundaries are stored as sorted list of times (excludes start and end)
        self._boundaries: list[float] = []
        # Labels for each interval (len = len(boundaries) + 1)
        self._labels: list[str] = [""]

    @property
    def start_time(self) -> float:
        return self._start_time

    @property
    def end_time(self) -> float:
        return self._end_time

    def set_time_range(self, start: float, end: float):
        """Set the time range for this tier."""
        self._start_time = start
        self._end_time = end
        # Remove any boundaries outside the new range
        self._boundaries = [b for b in self._boundaries if start < b < end]
        # Ensure labels match
        while len(self._labels) < len(self._boundaries) + 1:
            self._labels.append("")
        while len(self._labels) > len(self._boundaries) + 1:
            self._labels.pop()

    @property
    def num_intervals(self) -> int:
        """Number of intervals in this tier."""
        return len(self._boundaries) + 1

    @property
    def boundaries(self) -> list[float]:
        """Get list of boundary times (read-only copy)."""
        return self._boundaries.copy()

    def get_all_times(self) -> list[float]:
        """Get all times including start, boundaries, and end."""
        return [self._start_time] + self._boundaries + [self._end_time]

    def get_interval(self, index: int) -> Interval:
        """Get interval by index."""
        if index < 0 or index >= self.num_intervals:
            raise IndexError(f"Interval index {index} out of range (0-{self.num_intervals-1})")

        times = self.get_all_times()
        return Interval(
            start=times[index],
            end=times[index + 1],
            text=self._labels[index]
        )

    def get_intervals(self) -> list[Interval]:
        """Get all intervals."""
        return [self.get_interval(i) for i in range(self.num_intervals)]

    def get_interval_at_time(self, time: float) -> tuple[int, Interval]:
        """Get the interval containing the given time.

        Returns (index, interval) tuple.
        """
        if time < self._start_time or time > self._end_time:
            raise ValueError(f"Time {time} outside tier range ({self._start_time}-{self._end_time})")

        # Find which interval contains this time
        # bisect_right gives us the index where time would be inserted
        idx = bisect.bisect_right(self._boundaries, time)
        return idx, self.get_interval(idx)

    def add_boundary(self, time: float) -> int:
        """Add a boundary at the given time.

        Returns the index of the new boundary.
        Raises ValueError if boundary already exists or is outside range.
        """
        if time <= self._start_time or time >= self._end_time:
            raise ValueError(f"Boundary {time} must be within ({self._start_time}, {self._end_time})")

        if time in self._boundaries:
            raise ValueError(f"Boundary at {time} already exists")

        # Find insertion point
        idx = bisect.bisect_left(self._boundaries, time)
        self._boundaries.insert(idx, time)

        # Split the label at this boundary (new interval gets empty label)
        # The interval that was split keeps its label, new one is empty
        self._labels.insert(idx + 1, "")

        return idx

    def remove_boundary(self, index: int) -> float:
        """Remove boundary by index.

        Returns the time of the removed boundary.
        """
        if index < 0 or index >= len(self._boundaries):
            raise IndexError(f"Boundary index {index} out of range")

        time = self._boundaries.pop(index)
        # Merge labels: keep the first interval's label
        if index + 1 < len(self._labels):
            self._labels.pop(index + 1)

        return time

    def remove_boundary_at_time(self, time: float, tolerance: float = 0.001) -> bool:
        """Remove boundary closest to given time within tolerance.

        Returns True if a boundary was removed, False otherwise.
        """
        for i, b in enumerate(self._boundaries):
            if abs(b - time) <= tolerance:
                self.remove_boundary(i)
                return True
        return False

    def move_boundary(self, index: int, new_time: float) -> bool:
        """Move a boundary to a new time.

        Returns True if successful, False if move would violate ordering.
        """
        if index < 0 or index >= len(self._boundaries):
            raise IndexError(f"Boundary index {index} out of range")

        if new_time <= self._start_time or new_time >= self._end_time:
            return False

        # Check that new position doesn't violate ordering
        prev_time = self._boundaries[index - 1] if index > 0 else self._start_time
        next_time = self._boundaries[index + 1] if index < len(self._boundaries) - 1 else self._end_time

        if new_time <= prev_time or new_time >= next_time:
            return False

        self._boundaries[index] = new_time
        return True

    def find_nearest_boundary(self, time: float) -> tuple[int, float, float]:
        """Find the boundary nearest to the given time.

        Returns (index, boundary_time, distance) or (-1, 0, inf) if no boundaries.
        """
        if not self._boundaries:
            return (-1, 0.0, float('inf'))

        min_dist = float('inf')
        nearest_idx = -1
        nearest_time = 0.0

        for i, b in enumerate(self._boundaries):
            dist = abs(b - time)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
                nearest_time = b

        return (nearest_idx, nearest_time, min_dist)

    def set_interval_text(self, index: int, text: str):
        """Set the text for an interval."""
        if index < 0 or index >= len(self._labels):
            raise IndexError(f"Interval index {index} out of range")
        self._labels[index] = text

    def get_interval_text(self, index: int) -> str:
        """Get the text for an interval."""
        if index < 0 or index >= len(self._labels):
            raise IndexError(f"Interval index {index} out of range")
        return self._labels[index]

    def clear(self):
        """Remove all boundaries and labels."""
        self._boundaries.clear()
        self._labels = [""]


class AnnotationSet:
    """A collection of annotation tiers."""

    MAX_TIERS = 5

    def __init__(self, duration: float = 0.0):
        self._duration = duration
        self._tiers: list[Tier] = []
        self._active_tier_index: int = 0

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, value: float):
        self._duration = value
        # Update all tiers
        for tier in self._tiers:
            tier.set_time_range(0.0, value)

    @property
    def num_tiers(self) -> int:
        return len(self._tiers)

    @property
    def active_tier_index(self) -> int:
        return self._active_tier_index

    @active_tier_index.setter
    def active_tier_index(self, value: int):
        if 0 <= value < len(self._tiers):
            self._active_tier_index = value

    @property
    def active_tier(self) -> Optional[Tier]:
        if 0 <= self._active_tier_index < len(self._tiers):
            return self._tiers[self._active_tier_index]
        return None

    def add_tier(self, name: str = "") -> Tier:
        """Add a new tier."""
        if len(self._tiers) >= self.MAX_TIERS:
            raise ValueError(f"Maximum number of tiers ({self.MAX_TIERS}) reached")

        if not name:
            name = f"Tier {len(self._tiers) + 1}"

        tier = Tier(name=name, start_time=0.0, end_time=self._duration)
        self._tiers.append(tier)
        return tier

    def remove_tier(self, index: int) -> Tier:
        """Remove a tier by index."""
        if index < 0 or index >= len(self._tiers):
            raise IndexError(f"Tier index {index} out of range")

        tier = self._tiers.pop(index)

        # Adjust active tier index if needed
        if index < self._active_tier_index:
            # Removed tier before active, shift active index down
            self._active_tier_index -= 1
        elif self._active_tier_index >= len(self._tiers):
            # Active tier was removed or now out of range
            self._active_tier_index = max(0, len(self._tiers) - 1)

        return tier

    def get_tier(self, index: int) -> Tier:
        """Get tier by index."""
        if index < 0 or index >= len(self._tiers):
            raise IndexError(f"Tier index {index} out of range")
        return self._tiers[index]

    def get_tiers(self) -> list[Tier]:
        """Get all tiers."""
        return self._tiers.copy()

    def rename_tier(self, index: int, name: str):
        """Rename a tier."""
        self._tiers[index].name = name

    def move_tier(self, from_index: int, to_index: int):
        """Move a tier from one position to another."""
        if from_index < 0 or from_index >= len(self._tiers):
            raise IndexError(f"Source tier index {from_index} out of range")
        if to_index < 0 or to_index >= len(self._tiers):
            raise IndexError(f"Target tier index {to_index} out of range")

        tier = self._tiers.pop(from_index)
        self._tiers.insert(to_index, tier)

        # Update active tier index
        if self._active_tier_index == from_index:
            self._active_tier_index = to_index
        elif from_index < self._active_tier_index <= to_index:
            self._active_tier_index -= 1
        elif to_index <= self._active_tier_index < from_index:
            self._active_tier_index += 1

    def clear(self):
        """Remove all tiers."""
        self._tiers.clear()
        self._active_tier_index = 0
