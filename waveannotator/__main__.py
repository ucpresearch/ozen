"""
Entry point for WaveAnnotator.

This module provides the command-line interface for launching WaveAnnotator.
It handles argument parsing and initializes the Qt application.

Usage:
    python -m waveannotator [audio_file] [textgrid_file] [--tiers tier1 tier2 ...]

Examples:
    # Open the application
    python -m waveannotator

    # Open with an audio file
    python -m waveannotator recording.wav

    # Open with audio and existing TextGrid
    python -m waveannotator recording.wav annotations.TextGrid

    # Open with audio and create new TextGrid with predefined tiers
    python -m waveannotator recording.wav --tiers words phones
"""

import sys
import argparse
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from .ui.main_window import MainWindow


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace with:
            - audio_file: Path to audio file (optional)
            - textgrid_file: Path to TextGrid file (optional)
            - tiers: List of tier names to create (optional)
    """
    parser = argparse.ArgumentParser(
        description="WaveAnnotator - Audio annotation tool"
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        help="Audio file to open (WAV, FLAC, OGG, MP3)"
    )
    parser.add_argument(
        "textgrid_file",
        nargs="?",
        help="TextGrid file to import (optional)"
    )
    parser.add_argument(
        "--tiers", "-t",
        nargs="+",
        help="Predefined tier names (e.g., --tiers words phones)"
    )
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the WaveAnnotator application.

    Creates the Qt application, shows the main window, and optionally
    loads files specified on the command line. File loading is deferred
    to after the event loop starts using QTimer to ensure the UI is
    fully initialized.

    Returns:
        Exit code from the Qt application (0 for success)
    """
    args = parse_args()

    # Initialize Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("WaveAnnotator")
    app.setOrganizationName("WaveAnnotator")

    window = MainWindow()
    window.show()

    # Deferred file loading function - called after event loop starts
    def load_files():
        """Load audio and annotation files specified on command line."""
        if args.audio_file:
            # Load the audio file first (this also initializes the spectrogram)
            window._load_audio_file(args.audio_file)

            # Handle TextGrid file if provided
            if args.textgrid_file:
                textgrid_path = Path(args.textgrid_file)
                if textgrid_path.suffix.lower() in ('.textgrid', '.txt'):
                    # setup_textgrid_from_path handles both existing files
                    # and prompting to create new ones
                    window.setup_textgrid_from_path(args.textgrid_file, args.tiers)

            # If no TextGrid but tier names specified, create empty tiers
            elif args.tiers:
                window._create_predefined_tiers(args.tiers)

    if args.audio_file:
        # Defer file loading until after the Qt event loop starts
        # This ensures the window is fully initialized and visible
        QTimer.singleShot(100, load_files)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
