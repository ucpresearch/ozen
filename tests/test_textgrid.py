"""Tests for TextGrid and TSV import/export."""

import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from waveannotator.annotation.tier import Tier, Interval, AnnotationSet
from waveannotator.annotation.textgrid import (
    read_textgrid, write_textgrid, read_tsv, write_tsv
)


def test_textgrid_long_format():
    """Test TextGrid long format round-trip."""
    # Create an AnnotationSet
    annotations = AnnotationSet(duration=5.0)
    tier1 = annotations.add_tier("words")
    tier1.add_boundary(1.0)
    tier1.add_boundary(2.5)
    tier1.add_boundary(4.0)
    tier1.set_interval_text(0, "hello")
    tier1.set_interval_text(1, "world")
    tier1.set_interval_text(2, "test")
    tier1.set_interval_text(3, "")

    tier2 = annotations.add_tier("phones")
    tier2.add_boundary(0.5)
    tier2.add_boundary(1.0)
    tier2.set_interval_text(0, "h")
    tier2.set_interval_text(1, "e")
    tier2.set_interval_text(2, "lo")

    # Write long format
    with tempfile.NamedTemporaryFile(mode='w', suffix='.TextGrid', delete=False) as f:
        path = f.name

    try:
        write_textgrid(annotations, path, short_format=False)

        # Read it back
        read_back = read_textgrid(path)

        assert read_back.num_tiers == 2
        assert read_back.duration == 5.0

        tier = read_back.get_tier(0)
        assert tier.name == "words"
        assert tier.num_intervals == 4

        intervals = tier.get_intervals()
        assert intervals[0].text == "hello"
        assert intervals[1].text == "world"
        assert intervals[2].text == "test"

        print("test_textgrid_long_format PASSED")
    finally:
        os.unlink(path)


def test_textgrid_short_format():
    """Test TextGrid short format round-trip."""
    annotations = AnnotationSet(duration=3.0)
    tier = annotations.add_tier("test")
    tier.add_boundary(1.5)
    tier.set_interval_text(0, "first")
    tier.set_interval_text(1, "second")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.TextGrid', delete=False) as f:
        path = f.name

    try:
        write_textgrid(annotations, path, short_format=True)
        read_back = read_textgrid(path)

        assert read_back.num_tiers == 1
        tier = read_back.get_tier(0)
        assert tier.num_intervals == 2

        intervals = tier.get_intervals()
        assert intervals[0].text == "first"
        assert intervals[1].text == "second"

        print("test_textgrid_short_format PASSED")
    finally:
        os.unlink(path)


def test_tsv_roundtrip():
    """Test TSV round-trip."""
    annotations = AnnotationSet(duration=5.0)
    tier1 = annotations.add_tier("words")
    tier1.add_boundary(1.0)
    tier1.add_boundary(2.5)
    tier1.set_interval_text(0, "hello")
    tier1.set_interval_text(1, "world")
    tier1.set_interval_text(2, "end")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        path = f.name

    try:
        write_tsv(annotations, path)
        read_back = read_tsv(path)

        assert read_back.num_tiers == 1
        tier = read_back.get_tier(0)
        assert tier.name == "words"
        assert tier.num_intervals == 3

        intervals = tier.get_intervals()
        assert intervals[0].text == "hello"
        assert intervals[1].text == "world"
        assert intervals[2].text == "end"

        print("test_tsv_roundtrip PASSED")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    test_textgrid_long_format()
    test_textgrid_short_format()
    test_tsv_roundtrip()
    print("\nAll tests passed!")
