"""Tests for annotation tier module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from waveannotator.annotation.tier import Tier, Interval, AnnotationSet


def test_interval_basic():
    """Test basic interval creation and properties."""
    interval = Interval(start=0.0, end=1.0, text="test")
    assert interval.start == 0.0
    assert interval.end == 1.0
    assert interval.text == "test"
    assert interval.duration == 1.0
    print("test_interval_basic PASSED")


def test_interval_contains():
    """Test interval contains method."""
    interval = Interval(start=1.0, end=2.0)
    assert interval.contains(1.0)
    assert interval.contains(1.5)
    assert not interval.contains(2.0)  # end is exclusive
    assert not interval.contains(0.5)
    print("test_interval_contains PASSED")


def test_interval_overlaps():
    """Test interval overlap detection."""
    a = Interval(start=1.0, end=3.0)
    b = Interval(start=2.0, end=4.0)
    c = Interval(start=4.0, end=5.0)

    assert a.overlaps(b)
    assert b.overlaps(a)
    assert not a.overlaps(c)
    assert not c.overlaps(a)
    print("test_interval_overlaps PASSED")


def test_tier_basic():
    """Test basic tier creation."""
    tier = Tier(name="test", start_time=0.0, end_time=10.0)
    assert tier.name == "test"
    assert tier.start_time == 0.0
    assert tier.end_time == 10.0
    assert tier.num_intervals == 1  # One interval initially
    print("test_tier_basic PASSED")


def test_tier_add_boundary():
    """Test adding boundaries to a tier."""
    tier = Tier(name="test", start_time=0.0, end_time=10.0)

    # Add first boundary
    tier.add_boundary(5.0)
    assert tier.num_intervals == 2
    assert tier.boundaries == [5.0]

    # Add another boundary
    tier.add_boundary(2.5)
    assert tier.num_intervals == 3
    assert tier.boundaries == [2.5, 5.0]

    # Try to add duplicate boundary
    try:
        tier.add_boundary(5.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Try to add boundary outside range
    try:
        tier.add_boundary(0.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("test_tier_add_boundary PASSED")


def test_tier_remove_boundary():
    """Test removing boundaries from a tier."""
    tier = Tier(name="test", start_time=0.0, end_time=10.0)
    tier.add_boundary(2.5)
    tier.add_boundary(5.0)
    tier.add_boundary(7.5)

    assert tier.num_intervals == 4

    # Remove middle boundary
    tier.remove_boundary(1)  # removes 5.0
    assert tier.num_intervals == 3
    assert tier.boundaries == [2.5, 7.5]

    print("test_tier_remove_boundary PASSED")


def test_tier_move_boundary():
    """Test moving boundaries."""
    tier = Tier(name="test", start_time=0.0, end_time=10.0)
    tier.add_boundary(2.5)
    tier.add_boundary(5.0)
    tier.add_boundary(7.5)

    # Move middle boundary
    result = tier.move_boundary(1, 6.0)
    assert result
    assert tier.boundaries == [2.5, 6.0, 7.5]

    # Try to move past next boundary (should fail)
    result = tier.move_boundary(1, 8.0)
    assert not result

    print("test_tier_move_boundary PASSED")


def test_tier_intervals():
    """Test getting intervals from tier."""
    tier = Tier(name="test", start_time=0.0, end_time=10.0)
    tier.add_boundary(5.0)
    tier.set_interval_text(0, "first")
    tier.set_interval_text(1, "second")

    intervals = tier.get_intervals()
    assert len(intervals) == 2

    assert intervals[0].start == 0.0
    assert intervals[0].end == 5.0
    assert intervals[0].text == "first"

    assert intervals[1].start == 5.0
    assert intervals[1].end == 10.0
    assert intervals[1].text == "second"

    print("test_tier_intervals PASSED")


def test_tier_get_interval_at_time():
    """Test finding interval at specific time."""
    tier = Tier(name="test", start_time=0.0, end_time=10.0)
    tier.add_boundary(3.0)
    tier.add_boundary(7.0)
    tier.set_interval_text(0, "first")
    tier.set_interval_text(1, "second")
    tier.set_interval_text(2, "third")

    idx, interval = tier.get_interval_at_time(1.0)
    assert idx == 0
    assert interval.text == "first"

    idx, interval = tier.get_interval_at_time(5.0)
    assert idx == 1
    assert interval.text == "second"

    idx, interval = tier.get_interval_at_time(9.0)
    assert idx == 2
    assert interval.text == "third"

    print("test_tier_get_interval_at_time PASSED")


def test_tier_find_nearest_boundary():
    """Test finding nearest boundary."""
    tier = Tier(name="test", start_time=0.0, end_time=10.0)
    tier.add_boundary(2.0)
    tier.add_boundary(5.0)
    tier.add_boundary(8.0)

    idx, time, dist = tier.find_nearest_boundary(4.5)
    assert idx == 1
    assert time == 5.0
    assert abs(dist - 0.5) < 0.001

    idx, time, dist = tier.find_nearest_boundary(1.0)
    assert idx == 0
    assert time == 2.0
    assert abs(dist - 1.0) < 0.001

    print("test_tier_find_nearest_boundary PASSED")


def test_annotation_set_basic():
    """Test basic annotation set operations."""
    annotations = AnnotationSet(duration=10.0)
    assert annotations.duration == 10.0
    assert annotations.num_tiers == 0

    # Add tiers
    tier1 = annotations.add_tier("words")
    tier2 = annotations.add_tier("phones")

    assert annotations.num_tiers == 2
    assert tier1.name == "words"
    assert tier2.name == "phones"

    # Active tier
    assert annotations.active_tier_index == 0
    annotations.active_tier_index = 1
    assert annotations.active_tier == tier2

    print("test_annotation_set_basic PASSED")


def test_annotation_set_max_tiers():
    """Test max tier limit."""
    annotations = AnnotationSet(duration=10.0)

    for i in range(AnnotationSet.MAX_TIERS):
        annotations.add_tier(f"Tier {i+1}")

    assert annotations.num_tiers == AnnotationSet.MAX_TIERS

    try:
        annotations.add_tier("Extra")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("test_annotation_set_max_tiers PASSED")


def test_annotation_set_remove_tier():
    """Test removing tiers."""
    annotations = AnnotationSet(duration=10.0)
    t1 = annotations.add_tier("A")
    t2 = annotations.add_tier("B")
    t3 = annotations.add_tier("C")

    annotations.active_tier_index = 1

    annotations.remove_tier(0)
    assert annotations.num_tiers == 2
    assert annotations.get_tier(0).name == "B"
    assert annotations.active_tier_index == 0  # Adjusted

    print("test_annotation_set_remove_tier PASSED")


if __name__ == "__main__":
    test_interval_basic()
    test_interval_contains()
    test_interval_overlaps()
    test_tier_basic()
    test_tier_add_boundary()
    test_tier_remove_boundary()
    test_tier_move_boundary()
    test_tier_intervals()
    test_tier_get_interval_at_time()
    test_tier_find_nearest_boundary()
    test_annotation_set_basic()
    test_annotation_set_max_tiers()
    test_annotation_set_remove_tier()
    print("\nAll tier tests passed!")
