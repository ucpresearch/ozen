"""Main application window."""

from pathlib import Path
from typing import Optional
import traceback

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QMetaObject, Q_ARG, Qt as QtCore
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QProgressDialog, QStatusBar, QToolBar,
    QCheckBox, QComboBox, QLabel, QSplitter, QGroupBox,
    QMessageBox, QPushButton
)

from ..audio.loader import load_audio, AudioData
from ..audio.player import AudioPlayer
from ..analysis.acoustic import extract_features, compute_spectrogram, AcousticFeatures
from ..visualization.waveform import WaveformWidget
from ..visualization.spectrogram import SpectrogramWidget


class FeatureExtractionThread(QThread):
    """Background thread for feature extraction."""

    progress = pyqtSignal(float)
    finished = pyqtSignal(object)  # AcousticFeatures or Exception
    error = pyqtSignal(str)

    # Formant presets
    FORMANT_PRESETS = {
        'male': {'max_formant': 5000, 'pitch_floor': 75, 'pitch_ceiling': 300},
        'female': {'max_formant': 5500, 'pitch_floor': 100, 'pitch_ceiling': 500},
        'child': {'max_formant': 8000, 'pitch_floor': 150, 'pitch_ceiling': 600},
    }

    def __init__(self, file_path: str, time_step: float = 0.01, preset: str = 'male'):
        super().__init__()
        self.file_path = file_path
        self.time_step = time_step
        self.preset = preset

    def run(self):
        try:
            params = self.FORMANT_PRESETS.get(self.preset, self.FORMANT_PRESETS['male'])
            features = extract_features(
                self.file_path,
                time_step=self.time_step,
                max_formant=params['max_formant'],
                pitch_floor=params['pitch_floor'],
                pitch_ceiling=params['pitch_ceiling'],
                progress_callback=lambda p: self.progress.emit(p)
            )
            self.finished.emit(features)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    """Main application window for WaveAnnotator."""

    # Signal for thread-safe playback finished notification
    _playback_finished_signal = pyqtSignal()

    def __init__(self):
        super().__init__()

        self._audio_data: AudioData | None = None
        self._features: AcousticFeatures | None = None
        self._player = AudioPlayer()
        self._current_file_path: str | None = None
        self._feature_thread: FeatureExtractionThread | None = None

        self._setup_ui()
        self._setup_menus()
        self._setup_toolbar()
        self._setup_shortcuts()
        self._connect_signals()

        # Timer for updating cursor during playback
        self._playback_timer = QTimer()
        self._playback_timer.setInterval(30)  # ~33 fps
        self._playback_timer.timeout.connect(self._update_playback_cursor)

        # Connect playback finished signal (for thread-safe callback)
        self._playback_finished_signal.connect(self._on_playback_finished)

    def _setup_ui(self):
        """Setup the main UI layout."""
        self.setWindowTitle("WaveAnnotator")
        self.setMinimumSize(1200, 800)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(5, 5, 5, 5)

        # Create splitter for waveform and spectrogram
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)

        # Waveform widget
        self._waveform = WaveformWidget()
        splitter.addWidget(self._waveform)

        # Spectrogram widget
        self._spectrogram = SpectrogramWidget()
        splitter.addWidget(self._spectrogram)

        # Set initial sizes (1:3 ratio)
        splitter.setSizes([200, 600])

        # Control panel
        controls = self._create_control_panel()
        layout.addWidget(controls)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready - Open an audio file to begin")

    def _create_control_panel(self) -> QWidget:
        """Create the control panel with track toggles and settings."""
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)

        # Spectrogram settings
        spec_group = QGroupBox("Spectrogram")
        spec_layout = QHBoxLayout(spec_group)

        # Wideband/Narrowband toggle (Narrowband default - shows harmonics better)
        self._bandwidth_combo = QComboBox()
        self._bandwidth_combo.addItems(['Narrowband', 'Wideband'])
        self._bandwidth_combo.setToolTip("Narrowband: better frequency resolution (shows harmonics)\nWideband: better time resolution")
        self._bandwidth_combo.currentTextChanged.connect(self._on_bandwidth_changed)
        spec_layout.addWidget(QLabel("Band:"))
        spec_layout.addWidget(self._bandwidth_combo)

        # Colormap
        self._colormap_combo = QComboBox()
        self._colormap_combo.addItems(['grayscale', 'inferno', 'viridis'])
        self._colormap_combo.currentTextChanged.connect(self._spectrogram.set_colormap)
        spec_layout.addWidget(QLabel("Color:"))
        spec_layout.addWidget(self._colormap_combo)

        layout.addWidget(spec_group)

        # Formant settings
        formant_group = QGroupBox("Formants")
        formant_layout = QHBoxLayout(formant_group)

        self._formant_preset_combo = QComboBox()
        self._formant_preset_combo.addItems(['Female', 'Male', 'Child'])
        self._formant_preset_combo.setToolTip(
            "Female: max 5500 Hz, pitch 100-500 Hz\nMale: max 5000 Hz, pitch 75-300 Hz\nChild: max 8000 Hz, pitch 150-600 Hz"
        )
        self._formant_preset_combo.currentTextChanged.connect(self._on_formant_preset_changed)
        formant_layout.addWidget(QLabel("Voice:"))
        formant_layout.addWidget(self._formant_preset_combo)

        layout.addWidget(formant_group)

        # Overlay toggles
        overlay_group = QGroupBox("Overlays")
        overlay_layout = QHBoxLayout(overlay_group)

        self._pitch_check = QCheckBox("Pitch")
        self._pitch_check.setChecked(True)
        self._pitch_check.toggled.connect(lambda v: self._spectrogram.set_track_visible('pitch', v))
        overlay_layout.addWidget(self._pitch_check)

        self._formants_check = QCheckBox("Formants")
        self._formants_check.setChecked(True)
        self._formants_check.toggled.connect(lambda v: self._spectrogram.set_track_visible('formants', v))
        overlay_layout.addWidget(self._formants_check)

        self._intensity_check = QCheckBox("Intensity")
        self._intensity_check.setChecked(False)
        self._intensity_check.toggled.connect(lambda v: self._spectrogram.set_track_visible('intensity', v))
        overlay_layout.addWidget(self._intensity_check)

        self._cog_check = QCheckBox("CoG")
        self._cog_check.setChecked(False)
        self._cog_check.toggled.connect(lambda v: self._spectrogram.set_track_visible('cog', v))
        overlay_layout.addWidget(self._cog_check)

        self._hnr_check = QCheckBox("HNR")
        self._hnr_check.setChecked(False)
        self._hnr_check.toggled.connect(lambda v: self._spectrogram.set_track_visible('hnr', v))
        overlay_layout.addWidget(self._hnr_check)

        layout.addWidget(overlay_group)

        # Extract features button
        self._extract_btn = QPushButton("Extract Features")
        self._extract_btn.setEnabled(False)
        self._extract_btn.clicked.connect(self._start_feature_extraction)
        layout.addWidget(self._extract_btn)

        # Spacer
        layout.addStretch()

        # Time display
        self._time_label = QLabel("Time: 0.000s")
        self._time_label.setMinimumWidth(120)
        layout.addWidget(self._time_label)

        return panel

    def _setup_menus(self):
        """Setup menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Audio...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_file_dialog)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        zoom_in = QAction("Zoom &In", self)
        zoom_in.setShortcut(QKeySequence.StandardKey.ZoomIn)
        zoom_in.triggered.connect(self._zoom_in)
        view_menu.addAction(zoom_in)

        zoom_out = QAction("Zoom &Out", self)
        zoom_out.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoom_out.triggered.connect(self._zoom_out)
        view_menu.addAction(zoom_out)

        fit_view = QAction("&Fit to Window", self)
        fit_view.setShortcut("Ctrl+0")
        fit_view.triggered.connect(self._fit_view)
        view_menu.addAction(fit_view)

    def _setup_toolbar(self):
        """Setup toolbar with prominent playback buttons."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(toolbar)

        # Open file button
        open_btn = QPushButton("Open")
        open_btn.clicked.connect(self._open_file_dialog)
        open_btn.setToolTip("Open audio file (Ctrl+O)")
        toolbar.addWidget(open_btn)

        toolbar.addSeparator()

        # Playback controls - larger, more prominent buttons
        self._play_btn = QPushButton("▶ Play")
        self._play_btn.clicked.connect(self._toggle_playback)
        self._play_btn.setEnabled(False)
        self._play_btn.setToolTip("Play/Pause (Space)")
        self._play_btn.setMinimumWidth(80)
        toolbar.addWidget(self._play_btn)

        self._stop_btn = QPushButton("■ Stop")
        self._stop_btn.clicked.connect(self._stop_playback)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setToolTip("Stop (Escape)")
        self._stop_btn.setMinimumWidth(70)
        toolbar.addWidget(self._stop_btn)

        toolbar.addSeparator()

        # Zoom controls
        zoom_in_btn = QPushButton("+ Zoom In")
        zoom_in_btn.clicked.connect(self._zoom_in)
        zoom_in_btn.setToolTip("Zoom in (Ctrl++)")
        toolbar.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("− Zoom Out")
        zoom_out_btn.clicked.connect(self._zoom_out)
        zoom_out_btn.setToolTip("Zoom out (Ctrl+-)")
        toolbar.addWidget(zoom_out_btn)

        fit_btn = QPushButton("Fit")
        fit_btn.clicked.connect(self._fit_view)
        fit_btn.setToolTip("Fit to window (Ctrl+0)")
        toolbar.addWidget(fit_btn)

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Space for play/pause
        pass  # Handled by keyPressEvent

    def _connect_signals(self):
        """Connect widget signals."""
        # Synchronize view ranges between waveform and spectrogram
        self._waveform.time_range_changed.connect(self._sync_spectrogram_range)
        self._spectrogram.time_range_changed.connect(self._sync_waveform_range)

        # Update cursor between views
        self._waveform.cursor_moved.connect(self._on_cursor_moved)
        self._spectrogram.cursor_moved.connect(self._on_cursor_moved)

        # Sync selection between views
        self._waveform.selection_changed.connect(self._sync_selection_to_spectrogram)
        self._spectrogram.selection_changed.connect(self._sync_selection_to_waveform)

        # Click on selection to play
        self._waveform.selection_clicked.connect(self._play_selection)
        self._spectrogram.selection_clicked.connect(self._play_selection)

        # Player callbacks
        self._player.set_position_callback(self._on_playback_position)
        # Use signal for thread-safe callback from audio thread
        self._player.set_finished_callback(lambda: self._playback_finished_signal.emit())

    def _sync_selection_to_spectrogram(self, start: float, end: float):
        """Sync selection from waveform to spectrogram."""
        self._spectrogram.blockSignals(True)
        self._spectrogram.set_selection(start, end)
        self._spectrogram.blockSignals(False)

    def _sync_selection_to_waveform(self, start: float, end: float):
        """Sync selection from spectrogram to waveform."""
        self._waveform.blockSignals(True)
        self._waveform.set_selection(start, end)
        self._waveform.blockSignals(False)

    def _sync_spectrogram_range(self, start: float, end: float):
        """Sync spectrogram view to waveform."""
        self._spectrogram.blockSignals(True)
        self._spectrogram.set_x_range(start, end)
        self._spectrogram.blockSignals(False)

    def _sync_waveform_range(self, start: float, end: float):
        """Sync waveform view to spectrogram."""
        self._waveform.blockSignals(True)
        self._waveform.setXRange(start, end, padding=0)
        self._waveform.blockSignals(False)

    def _on_cursor_moved(self, time: float):
        """Handle cursor movement."""
        self._time_label.setText(f"Time: {time:.3f}s")
        # Block signals to prevent recursion
        self._waveform.blockSignals(True)
        self._spectrogram.blockSignals(True)
        self._waveform.set_cursor_position(time)
        self._spectrogram.set_cursor_position(time)
        self._waveform.blockSignals(False)
        self._spectrogram.blockSignals(False)

    def _open_file_dialog(self):
        """Open file dialog to select audio file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "Audio Files (*.wav *.flac *.ogg *.mp3);;All Files (*)"
        )
        if file_path:
            self._load_audio_file(file_path)

    def _load_audio_file(self, file_path: str):
        """Load and analyze an audio file."""
        try:
            self._status_bar.showMessage(f"Loading {Path(file_path).name}...")

            # Load audio
            self._audio_data = load_audio(file_path)
            self._player.set_audio_data(self._audio_data)

            # Display waveform
            self._waveform.set_audio_data(self._audio_data)

            # Update window title with filename
            self.setWindowTitle(f"WaveAnnotator - {Path(file_path).name}")

            self._status_bar.showMessage("Computing spectrogram...")
            QTimer.singleShot(10, lambda: self._compute_analysis(file_path))

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load audio: {e}")
            self._status_bar.showMessage("Error loading file")

    def _on_bandwidth_changed(self, bandwidth: str):
        """Handle wideband/narrowband toggle."""
        if self._current_file_path:
            self._recompute_spectrogram()

    def _on_formant_preset_changed(self, preset: str):
        """Handle formant preset change."""
        # Store the preset for next feature extraction
        self._current_formant_preset = preset.lower()

        # Auto re-extract if features were already extracted and file is short
        if (self._features is not None and
            self._audio_data is not None and
            self._audio_data.duration <= 60.0 and
            (self._feature_thread is None or not self._feature_thread.isRunning())):
            self._status_bar.showMessage(
                f"Formant preset changed to {preset}. Re-extracting features..."
            )
            self._start_feature_extraction()
        else:
            self._status_bar.showMessage(
                f"Formant preset changed to {preset}. Click 'Extract Features' to apply."
            )

    def _recompute_spectrogram(self):
        """Recompute spectrogram with current settings."""
        if not self._current_file_path:
            return

        bandwidth = self._bandwidth_combo.currentText()
        if bandwidth == 'Wideband':
            window_length = 0.005  # 5ms - better time resolution
        else:  # Narrowband
            window_length = 0.025  # 25ms - better frequency resolution (shows harmonics)

        times, freqs, spec_db = compute_spectrogram(
            self._current_file_path,
            window_length=window_length,
            max_frequency=5000.0,
            dynamic_range=70.0
        )
        self._spectrogram.set_spectrogram(times, freqs, spec_db)

    def _compute_analysis(self, file_path: str):
        """Compute spectrogram (fast) - features extracted separately."""
        try:
            self._current_file_path = file_path
            self._current_formant_preset = 'female'  # Default

            # Compute spectrogram based on current bandwidth setting
            bandwidth = self._bandwidth_combo.currentText()
            if bandwidth == 'Wideband':
                window_length = 0.005  # Better time resolution
            else:
                window_length = 0.025  # Better frequency resolution

            times, freqs, spec_db = compute_spectrogram(
                file_path,
                window_length=window_length,
                max_frequency=5000.0,
                dynamic_range=70.0
            )
            self._spectrogram.set_spectrogram(times, freqs, spec_db)

            # Enable playback and feature extraction
            self._play_btn.setEnabled(True)
            self._stop_btn.setEnabled(True)
            self._extract_btn.setEnabled(True)

            # Auto-extract features for files under 60 seconds
            if self._audio_data.duration <= 60.0:
                self._status_bar.showMessage(
                    f"Loaded: {Path(file_path).name} - Auto-extracting features..."
                )
                # Start extraction automatically after a short delay
                QTimer.singleShot(100, self._start_feature_extraction)
            else:
                self._status_bar.showMessage(
                    f"Loaded: {Path(file_path).name} "
                    f"({self._audio_data.duration:.2f}s, {self._audio_data.sample_rate}Hz) "
                    f"- Click 'Extract Features' for overlays"
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {e}")
            self._status_bar.showMessage("Analysis error")

    def _start_feature_extraction(self):
        """Start feature extraction in background thread."""
        if not self._current_file_path:
            return

        self._extract_btn.setEnabled(False)
        self._extract_btn.setText("Extracting...")

        # Get current formant preset
        preset = getattr(self, '_current_formant_preset', 'male')
        self._status_bar.showMessage(
            f"Extracting features ({preset} preset)..."
        )

        # Use larger time step for faster extraction
        self._feature_thread = FeatureExtractionThread(
            self._current_file_path,
            time_step=0.01,
            preset=preset
        )
        self._feature_thread.progress.connect(self._on_feature_progress)
        self._feature_thread.finished.connect(self._on_features_ready)
        self._feature_thread.error.connect(self._on_feature_error)
        self._feature_thread.start()

    def _on_feature_progress(self, progress: float):
        """Update status with extraction progress."""
        self._status_bar.showMessage(f"Extracting features: {int(progress * 100)}%")

    def _on_features_ready(self, features: AcousticFeatures):
        """Handle completed feature extraction."""
        self._features = features
        self._spectrogram.set_features(features)
        self._extract_btn.setText("Features Ready")
        self._status_bar.showMessage(
            f"Features extracted: {len(features.times)} frames"
        )

    def _on_feature_error(self, error_msg: str):
        """Handle feature extraction error."""
        self._extract_btn.setEnabled(True)
        self._extract_btn.setText("Extract Features")
        QMessageBox.warning(self, "Feature Extraction Error", error_msg)
        self._status_bar.showMessage("Feature extraction failed")

    def _toggle_playback(self):
        """Toggle play/pause."""
        if self._audio_data is None:
            return

        if self._player.is_playing:
            self._player.pause()
            self._playback_timer.stop()
            self._play_btn.setText("▶ Play")
        else:
            # Check for selection first
            selection = self._waveform.get_selection()
            if selection:
                # Play selection
                self._player.play(selection[0], selection[1])
            else:
                # No selection - play from cursor to end of visible region
                cursor_time = self._waveform._cursor_time
                view_start, view_end = self._waveform.get_view_range()
                # Clamp to valid range
                start_time = max(0, min(cursor_time, self._audio_data.duration))
                end_time = min(view_end, self._audio_data.duration)
                if start_time < end_time:
                    self._player.play(start_time, end_time)
                else:
                    # Cursor at end, play from beginning of view
                    self._player.play(max(0, view_start), end_time)
            self._playback_timer.start()
            self._play_btn.setText("❚❚ Pause")

    def _stop_playback(self):
        """Stop playback."""
        self._player.stop()
        self._playback_timer.stop()
        self._play_btn.setText("▶ Play")

    def _play_selection(self):
        """Play the current selection."""
        selection = self._waveform.get_selection()
        if selection and self._audio_data:
            self._player.play(selection[0], selection[1])
            self._playback_timer.start()
            self._play_btn.setText("❚❚ Pause")

    def _update_playback_cursor(self):
        """Update cursor position during playback."""
        if self._player.is_playing:
            time = self._player.current_time
            self._waveform.set_cursor_position(time)
            self._spectrogram.set_cursor_position(time)
            self._time_label.setText(f"Time: {time:.3f}s")

    def _on_playback_position(self, time: float):
        """Handle position updates from player."""
        # This is called from audio thread, handled by timer instead
        pass

    def _on_playback_finished(self):
        """Handle playback finished."""
        self._playback_timer.stop()
        self._play_btn.setText("▶ Play")

    def _zoom_in(self):
        """Zoom in on time axis."""
        if self._audio_data is None:
            return
        start, end = self._waveform.get_view_range()
        center = (start + end) / 2
        width = (end - start) / 2  # Halve the width
        self._waveform.setXRange(center - width / 2, center + width / 2)

    def _zoom_out(self):
        """Zoom out on time axis."""
        if self._audio_data is None:
            return
        start, end = self._waveform.get_view_range()
        center = (start + end) / 2
        width = (end - start) * 2  # Double the width
        new_start = max(0, center - width / 2)
        new_end = min(self._audio_data.duration, center + width / 2)
        self._waveform.setXRange(new_start, new_end)

    def _fit_view(self):
        """Fit entire audio to view."""
        if self._audio_data is None:
            return
        self._waveform.setXRange(0, self._audio_data.duration)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Space:
            self._toggle_playback()
        elif event.key() == Qt.Key.Key_Escape:
            self._stop_playback()
        elif event.key() == Qt.Key.Key_Tab:
            # Play visible window
            if self._audio_data:
                start, end = self._waveform.get_view_range()
                start = max(0, start)
                end = min(self._audio_data.duration, end)
                self._player.play(start, end)
                self._playback_timer.start()
                self._play_btn.setText("❚❚ Pause")
        else:
            super().keyPressEvent(event)
