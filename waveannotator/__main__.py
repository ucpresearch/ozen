"""Entry point for WaveAnnotator."""

import sys

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from .ui.main_window import MainWindow


def main() -> int:
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("WaveAnnotator")
    app.setOrganizationName("WaveAnnotator")

    window = MainWindow()
    window.show()

    # Open file from command line argument if provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        # Use QTimer to load after event loop starts
        QTimer.singleShot(100, lambda: window._load_audio_file(file_path))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
