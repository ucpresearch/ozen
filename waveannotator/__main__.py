"""Entry point for WaveAnnotator."""

import sys

from PyQt6.QtWidgets import QApplication

from .ui.main_window import MainWindow


def main() -> int:
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("WaveAnnotator")
    app.setOrganizationName("WaveAnnotator")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
