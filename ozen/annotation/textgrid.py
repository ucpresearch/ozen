"""TextGrid and TSV import/export for annotation tiers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TextIO

from .tier import Tier, AnnotationSet


def _unescape_praat_string(s: str) -> str:
    """
    Unescape a Praat text string.

    In Praat TextGrid format, strings are quoted with double quotes,
    and a literal quote within the string is represented as two quotes ("").
    This function removes the outer quotes and unescapes inner quotes.
    """
    # Remove outer quotes if present
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    # Unescape doubled quotes
    return s.replace('""', '"')


def _escape_praat_string(s: str) -> str:
    """
    Escape a string for Praat TextGrid format.

    Escapes literal quotes as "" and wraps in outer quotes.
    """
    return '"' + s.replace('"', '""') + '"'


def read_textgrid(file_path: str | Path) -> AnnotationSet:
    """Read a Praat TextGrid file.

    Supports both short and long TextGrid formats.
    """
    file_path = Path(file_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Detect format by looking for long-format-specific patterns
    # Long format has lines like "item [1]:" and "xmin = " with equals signs
    is_long_format = bool(re.search(r'item\s*\[\d+\]:', content)) or \
                     bool(re.search(r'xmin\s*=', content))

    if is_long_format:
        return _read_textgrid_long(content)
    else:
        return _read_textgrid_short(content)


def _read_textgrid_long(content: str) -> AnnotationSet:
    """Parse long-format TextGrid."""
    lines = content.split('\n')

    # Extract header info
    xmin = xmax = 0.0
    for line in lines[:10]:
        if 'xmin' in line:
            match = re.search(r'[\d.]+', line)
            if match:
                xmin = float(match.group())
        elif 'xmax' in line:
            match = re.search(r'[\d.]+', line)
            if match:
                xmax = float(match.group())
            break

    annotations = AnnotationSet(duration=xmax)

    # Find tier sections
    tier_pattern = re.compile(
        r'item\s*\[\d+\]:\s*\n\s*class\s*=\s*"(IntervalTier|TextTier)"',
        re.MULTILINE
    )

    # Split into tier blocks
    tier_starts = [m.start() for m in tier_pattern.finditer(content)]
    tier_starts.append(len(content))

    for i in range(len(tier_starts) - 1):
        tier_block = content[tier_starts[i]:tier_starts[i + 1]]

        # Extract tier info
        class_match = re.search(r'class\s*=\s*"(\w+)"', tier_block)
        name_match = re.search(r'name\s*=\s*"((?:[^"]|"")*)"', tier_block)

        if not class_match:
            continue

        tier_class = class_match.group(1)
        # Unescape doubled quotes in tier name
        tier_name = name_match.group(1).replace('""', '"') if name_match else f"Tier {i+1}"

        if tier_class == "IntervalTier":
            tier = annotations.add_tier(tier_name)
            _parse_interval_tier_long(tier_block, tier)
        # Skip TextTiers for now (point tiers)

    return annotations


def _parse_interval_tier_long(block: str, tier: Tier):
    """Parse intervals from a long-format tier block."""
    # Find all intervals
    # The text pattern handles escaped quotes: "" represents a literal "
    interval_pattern = re.compile(
        r'intervals\s*\[\d+\]:\s*\n'
        r'\s*xmin\s*=\s*([\d.]+)\s*\n'
        r'\s*xmax\s*=\s*([\d.]+)\s*\n'
        r'\s*text\s*=\s*"((?:[^"]|"")*)"',
        re.MULTILINE
    )

    intervals = []
    for match in interval_pattern.finditer(block):
        xmin = float(match.group(1))
        xmax = float(match.group(2))
        # Unescape doubled quotes: "" -> "
        text = match.group(3).replace('""', '"')
        intervals.append((xmin, xmax, text))

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    # Add boundaries and labels
    tier._boundaries.clear()
    tier._labels.clear()

    for i, (xmin, xmax, text) in enumerate(intervals):
        if i == 0:
            tier._start_time = xmin
        if i > 0:
            tier._boundaries.append(xmin)
        tier._labels.append(text)

    if intervals:
        tier._end_time = intervals[-1][1]


def _read_textgrid_short(content: str) -> AnnotationSet:
    """Parse short-format TextGrid."""
    lines = [l.strip() for l in content.split('\n') if l.strip()]

    # Skip header lines until we find the duration
    idx = 0
    xmin = xmax = 0.0
    found_xmin = False

    while idx < len(lines):
        line = lines[idx]
        if re.match(r'^[\d.]+$', line):
            if not found_xmin:
                xmin = float(line)
                found_xmin = True
            else:
                xmax = float(line)
                break
        idx += 1

    idx += 1  # Skip to tier count or exists line

    # Skip "<exists>" if present
    while idx < len(lines) and not re.match(r'^\d+$', lines[idx]):
        idx += 1

    if idx >= len(lines):
        return AnnotationSet(duration=xmax)

    num_tiers = int(lines[idx])
    idx += 1

    annotations = AnnotationSet(duration=xmax)

    for _ in range(num_tiers):
        if idx >= len(lines):
            break

        # Tier type
        tier_type = _unescape_praat_string(lines[idx])
        idx += 1

        # Tier name
        tier_name = _unescape_praat_string(lines[idx]) if idx < len(lines) else ""
        idx += 1

        # Tier xmin, xmax
        tier_xmin = float(lines[idx]) if idx < len(lines) else 0
        idx += 1
        tier_xmax = float(lines[idx]) if idx < len(lines) else xmax
        idx += 1

        # Number of intervals
        num_intervals = int(lines[idx]) if idx < len(lines) else 0
        idx += 1

        if tier_type == "IntervalTier":
            tier = annotations.add_tier(tier_name)
            tier._start_time = tier_xmin
            tier._end_time = tier_xmax
            tier._boundaries.clear()
            tier._labels.clear()

            for j in range(num_intervals):
                if idx + 2 >= len(lines):
                    break
                int_xmin = float(lines[idx])
                idx += 1
                int_xmax = float(lines[idx])
                idx += 1
                text = _unescape_praat_string(lines[idx])
                idx += 1

                if j > 0:
                    tier._boundaries.append(int_xmin)
                tier._labels.append(text)
        else:
            # Skip TextTier points
            for _ in range(num_intervals):
                idx += 2  # time and text

    return annotations


def write_textgrid(annotations: AnnotationSet, file_path: str | Path, short_format: bool = False):
    """Write annotations to a Praat TextGrid file.

    Args:
        annotations: The annotation set to write
        file_path: Output file path
        short_format: If True, write short format; otherwise long format
    """
    file_path = Path(file_path)

    with open(file_path, 'w', encoding='utf-8') as f:
        if short_format:
            _write_textgrid_short(annotations, f)
        else:
            _write_textgrid_long(annotations, f)


def _write_textgrid_long(annotations: AnnotationSet, f: TextIO):
    """Write long-format TextGrid."""
    f.write('File type = "ooTextFile"\n')
    f.write('Object class = "TextGrid"\n')
    f.write('\n')
    f.write(f'xmin = 0\n')
    f.write(f'xmax = {annotations.duration}\n')
    f.write('tiers? <exists>\n')
    f.write(f'size = {annotations.num_tiers}\n')
    f.write('item []:\n')

    for i, tier in enumerate(annotations.get_tiers()):
        f.write(f'    item [{i + 1}]:\n')
        f.write(f'        class = "IntervalTier"\n')
        f.write(f'        name = {_escape_praat_string(tier.name)}\n')
        f.write(f'        xmin = {tier.start_time}\n')
        f.write(f'        xmax = {tier.end_time}\n')
        f.write(f'        intervals: size = {tier.num_intervals}\n')

        for j, interval in enumerate(tier.get_intervals()):
            f.write(f'        intervals [{j + 1}]:\n')
            f.write(f'            xmin = {interval.start}\n')
            f.write(f'            xmax = {interval.end}\n')
            f.write(f'            text = {_escape_praat_string(interval.text)}\n')


def _write_textgrid_short(annotations: AnnotationSet, f: TextIO):
    """Write short-format TextGrid."""
    f.write('File type = "ooTextFile"\n')
    f.write('Object class = "TextGrid"\n')
    f.write('\n')
    f.write(f'{0}\n')
    f.write(f'{annotations.duration}\n')
    f.write('<exists>\n')
    f.write(f'{annotations.num_tiers}\n')

    for tier in annotations.get_tiers():
        f.write('"IntervalTier"\n')
        f.write(f'{_escape_praat_string(tier.name)}\n')
        f.write(f'{tier.start_time}\n')
        f.write(f'{tier.end_time}\n')
        f.write(f'{tier.num_intervals}\n')

        for interval in tier.get_intervals():
            f.write(f'{interval.start}\n')
            f.write(f'{interval.end}\n')
            f.write(f'{_escape_praat_string(interval.text)}\n')


def read_tsv(file_path: str | Path) -> AnnotationSet:
    """Read annotations from a TSV file.

    Expected format:
    tier_name<tab>start<tab>end<tab>text

    Each row represents an interval. Tiers are created as needed.
    """
    file_path = Path(file_path)

    tiers_data: dict[str, list[tuple[float, float, str]]] = {}
    max_time = 0.0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                continue  # Skip malformed lines

            tier_name = parts[0]
            try:
                start = float(parts[1])
                end = float(parts[2])
            except ValueError:
                continue  # Skip lines with invalid numbers

            text = parts[3] if len(parts) > 3 else ""

            if tier_name not in tiers_data:
                tiers_data[tier_name] = []

            tiers_data[tier_name].append((start, end, text))
            max_time = max(max_time, end)

    # Create annotation set
    annotations = AnnotationSet(duration=max_time)

    for tier_name, intervals in tiers_data.items():
        if annotations.num_tiers >= AnnotationSet.MAX_TIERS:
            break

        tier = annotations.add_tier(tier_name)

        # Sort intervals by start time
        intervals.sort(key=lambda x: x[0])

        # Build tier from intervals
        tier._boundaries.clear()
        tier._labels.clear()

        for i, (start, end, text) in enumerate(intervals):
            if i == 0:
                tier._start_time = start
            if i > 0:
                tier._boundaries.append(start)
            tier._labels.append(text)
            tier._end_time = end

    return annotations


def write_tsv(annotations: AnnotationSet, file_path: str | Path, include_header: bool = True):
    """Write annotations to a TSV file.

    Format:
    tier_name<tab>start<tab>end<tab>text
    """
    file_path = Path(file_path)

    with open(file_path, 'w', encoding='utf-8') as f:
        if include_header:
            f.write('# tier\tstart\tend\ttext\n')

        for tier in annotations.get_tiers():
            for interval in tier.get_intervals():
                # Escape tabs and newlines in text
                text = interval.text.replace('\t', ' ').replace('\n', ' ')
                f.write(f'{tier.name}\t{interval.start:.6f}\t{interval.end:.6f}\t{text}\n')
