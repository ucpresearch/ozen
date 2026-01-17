"""Entry point for WaveAnnotator."""

import sys
import argparse
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from .ui.main_window import MainWindow


def parse_args():
    """Parse command line arguments."""
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
    """Main entry point."""
    args = parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("WaveAnnotator")
    app.setOrganizationName("WaveAnnotator")

    window = MainWindow()
    window.show()

    # Open files from command line arguments if provided
    def load_files():
        if args.audio_file:
            window._load_audio_file(args.audio_file)

            # Setup TextGrid if provided (existing or new)
            if args.textgrid_file:
                textgrid_path = Path(args.textgrid_file)
                if textgrid_path.suffix.lower() in ('.textgrid', '.txt'):
                    # Pass tier names for new file creation
                    window.setup_textgrid_from_path(args.textgrid_file, args.tiers)

            # Create predefined tiers if specified (and no TextGrid was provided)
            elif args.tiers:
                window._create_predefined_tiers(args.tiers)

    if args.audio_file:
        # Use QTimer to load after event loop starts
        QTimer.singleShot(100, load_files)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
