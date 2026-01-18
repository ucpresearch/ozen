"""
Entry point for Ozen.

This module provides the command-line interface for launching Ozen.
It handles argument parsing and initializes the Qt application.

Usage:
    python -m ozen [audio_file] [textgrid_file] [options]

Options:
    --tiers, -t     Predefined tier names to create
    --config, -c    Path to custom config file (YAML or JSON)

Examples:
    # Open the application
    python -m ozen

    # Open with an audio file
    python -m ozen recording.wav

    # Open with audio and existing TextGrid
    python -m ozen recording.wav annotations.TextGrid

    # Open with audio and create new TextGrid with predefined tiers
    python -m ozen recording.wav --tiers words phones

    # Use a custom config file
    python -m ozen recording.wav --config myconfig.yaml
"""

import sys
import argparse
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from .ui.main_window import MainWindow
from . import config as config_module


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace with:
            - audio_file: Path to audio file (optional)
            - textgrid_file: Path to TextGrid file (optional)
            - tiers: List of tier names to create (optional)
            - config: Path to custom config file (optional)
    """
    parser = argparse.ArgumentParser(
        description="Ozen - Audio annotation tool"
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
    parser.add_argument(
        "--config", "-c",
        help="Path to custom config file (YAML or JSON)"
    )
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the Ozen application.

    Creates the Qt application, shows the main window, and optionally
    loads files specified on the command line. File loading is deferred
    to after the event loop starts using QTimer to ensure the UI is
    fully initialized.

    Returns:
        Exit code from the Qt application (0 for success)
    """
    args = parse_args()

    # Load custom config if specified (must happen before MainWindow is created)
    if args.config:
        config_module.reload_config(args.config)

    # Initialize Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Ozen")
    app.setOrganizationName("Ozen")

    window = MainWindow()

    # Set default tier names from CLI (takes priority over config)
    # This affects all future file opens, not just the initial one
    if args.tiers:
        window.set_default_tier_names(args.tiers)

    window.show()

    # Deferred file loading function - called after event loop starts
    def load_files():
        """Load audio and annotation files specified on command line."""
        if args.audio_file:
            # Load the audio file first (this also initializes the spectrogram)
            # _load_audio_file will use the default tiers set above (or from config)
            window._load_audio_file(args.audio_file)

            # Handle TextGrid file if provided - this overrides default tiers
            if args.textgrid_file:
                textgrid_path = Path(args.textgrid_file)
                if textgrid_path.suffix.lower() in ('.textgrid', '.txt'):
                    # setup_textgrid_from_path handles both existing files
                    # and prompting to create new ones
                    window.setup_textgrid_from_path(args.textgrid_file, args.tiers)

    if args.audio_file:
        # Defer file loading until after the Qt event loop starts
        # This ensures the window is fully initialized and visible
        QTimer.singleShot(100, load_files)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
