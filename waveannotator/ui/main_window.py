"""Main application window."""

from pathlib import Path
from typing import Optional
import traceback
import tempfile
import os

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QMetaObject, Q_ARG, Qt as QtCore, QEvent
from PyQt6.QtGui import QAction, QKeySequence, QKeyEvent
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QProgressDialog, QStatusBar, QToolBar,
    QCheckBox, QComboBox, QLabel, QSplitter, QGroupBox,
    QMessageBox, QPushButton, QInputDialog
)

from ..audio.loader import load_audio, AudioData
from ..audio.player import AudioPlayer
from ..analysis.acoustic import extract_features, compute_spectrogram, AcousticFeatures
from ..visualization.waveform import WaveformWidget
from ..visualization.spectrogram import SpectrogramWidget
from ..annotation import (
    AnnotationSet, AnnotationEditorWidget,
    read_textgrid, write_textgrid, read_tsv, write_tsv
)


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
        self._annotations: AnnotationSet | None = None

        # Save state tracking
        self._textgrid_path: str | None = None  # Current TextGrid save path
        self._is_dirty: bool = False  # True if unsaved changes exist

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

        # Auto-save timer (every 60 seconds)
        self._autosave_timer = QTimer()
        self._autosave_timer.setInterval(60000)  # 60 seconds
        self._autosave_timer.timeout.connect(self._auto_save)

        # Install event filter to catch key events for text input
        self.installEventFilter(self)

    def _setup_ui(self):
        """Setup the main UI layout."""
        self.setWindowTitle("WaveAnnotator")
        self.setMinimumSize(1200, 800)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(5, 5, 5, 5)

        # Create splitter for waveform, spectrogram, and annotations
        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)

        # Waveform widget
        self._waveform = WaveformWidget()
        splitter.addWidget(self._waveform)

        # Spectrogram widget
        self._spectrogram = SpectrogramWidget()
        splitter.addWidget(self._spectrogram)

        # Annotation editor widget
        self._annotation_editor = AnnotationEditorWidget()
        splitter.addWidget(self._annotation_editor)

        # Set initial sizes (waveform:spectrogram:annotations ratio)
        splitter.setSizes([150, 450, 200])

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

        # Annotation save/load
        import_textgrid = QAction("&Open TextGrid...", self)
        import_textgrid.triggered.connect(self._import_textgrid)
        file_menu.addAction(import_textgrid)

        save_textgrid = QAction("&Save", self)
        save_textgrid.setShortcut(QKeySequence.StandardKey.Save)
        save_textgrid.triggered.connect(self._save_textgrid)
        file_menu.addAction(save_textgrid)

        save_textgrid_as = QAction("Save &As...", self)
        save_textgrid_as.setShortcut("Ctrl+Shift+S")
        save_textgrid_as.triggered.connect(self._save_textgrid_as)
        file_menu.addAction(save_textgrid_as)

        file_menu.addSeparator()

        import_tsv = QAction("Import TS&V...", self)
        import_tsv.triggered.connect(self._import_tsv)
        file_menu.addAction(import_tsv)

        export_tsv = QAction("Export T&SV...", self)
        export_tsv.triggered.connect(self._export_tsv)
        file_menu.addAction(export_tsv)

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

        # Annotation menu
        annotation_menu = menubar.addMenu("&Annotation")

        add_tier = QAction("&Add Tier", self)
        add_tier.triggered.connect(self._add_tier)
        annotation_menu.addAction(add_tier)

        remove_tier = QAction("&Remove Tier", self)
        remove_tier.triggered.connect(self._remove_tier)
        annotation_menu.addAction(remove_tier)

        rename_tier = QAction("Re&name Tier...", self)
        rename_tier.triggered.connect(self._rename_tier)
        annotation_menu.addAction(rename_tier)

        annotation_menu.addSeparator()

        add_boundary = QAction("Add &Boundary (Enter)", self)
        add_boundary.triggered.connect(self._add_boundary_at_cursor)
        annotation_menu.addAction(add_boundary)

        remove_boundary = QAction("Remove Boun&dary (Delete)", self)
        remove_boundary.triggered.connect(self._remove_nearest_boundary)
        annotation_menu.addAction(remove_boundary)

        annotation_menu.addSeparator()

        play_interval = QAction("&Play Interval", self)
        play_interval.setShortcut("P")
        play_interval.triggered.connect(self._play_interval_at_cursor)
        annotation_menu.addAction(play_interval)

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

        # Navigation controls
        nav_left_btn = QPushButton("◀")
        nav_left_btn.clicked.connect(self._pan_left)
        nav_left_btn.setToolTip("Pan left (←)")
        nav_left_btn.setMaximumWidth(40)
        toolbar.addWidget(nav_left_btn)

        nav_right_btn = QPushButton("▶")
        nav_right_btn.clicked.connect(self._pan_right)
        nav_right_btn.setToolTip("Pan right (→)")
        nav_right_btn.setMaximumWidth(40)
        toolbar.addWidget(nav_right_btn)

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

        zoom_sel_btn = QPushButton("Zoom Sel")
        zoom_sel_btn.clicked.connect(self._zoom_to_selection)
        zoom_sel_btn.setToolTip("Zoom to selection (Z)")
        toolbar.addWidget(zoom_sel_btn)

        fit_btn = QPushButton("Fit All")
        fit_btn.clicked.connect(self._fit_view)
        fit_btn.setToolTip("Fit to window (Ctrl+0)")
        toolbar.addWidget(fit_btn)

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Space for play/pause
        pass  # Handled by keyPressEvent

    def _connect_signals(self):
        """Connect widget signals."""
        # Synchronize view ranges between all views
        self._waveform.time_range_changed.connect(self._sync_from_waveform)
        self._spectrogram.time_range_changed.connect(self._sync_from_spectrogram)
        self._annotation_editor.time_range_changed.connect(self._sync_from_annotations)

        # Update cursor between views
        self._waveform.cursor_moved.connect(self._on_cursor_moved)
        self._spectrogram.cursor_moved.connect(self._on_cursor_moved)
        self._annotation_editor.cursor_moved.connect(self._on_cursor_moved)

        # Sync selection between views
        self._waveform.selection_changed.connect(self._sync_selection_from_waveform)
        self._spectrogram.selection_changed.connect(self._sync_selection_from_spectrogram)
        self._annotation_editor.selection_changed.connect(self._sync_selection_from_annotations)

        # Click on selection to play
        self._waveform.selection_clicked.connect(self._play_selection)
        self._spectrogram.selection_clicked.connect(self._play_selection)
        self._annotation_editor.selection_clicked.connect(self._play_selection)

        # Annotation-specific signals
        self._annotation_editor.interval_play_requested.connect(self._play_interval)

        # Track annotation changes for dirty state
        self._annotation_editor.boundary_added.connect(self._mark_dirty)
        self._annotation_editor.boundary_removed.connect(self._mark_dirty)
        self._annotation_editor.boundary_moved.connect(self._mark_dirty)
        self._annotation_editor.interval_text_changed.connect(self._mark_dirty)

        # Player callbacks
        self._player.set_position_callback(self._on_playback_position)
        # Use signal for thread-safe callback from audio thread
        self._player.set_finished_callback(lambda: self._playback_finished_signal.emit())

    def _sync_selection_from_waveform(self, start: float, end: float):
        """Sync selection from waveform to other views."""
        self._spectrogram.blockSignals(True)
        self._annotation_editor.blockSignals(True)
        self._spectrogram.set_selection(start, end)
        self._annotation_editor.set_selection(start, end)
        self._spectrogram.blockSignals(False)
        self._annotation_editor.blockSignals(False)

    def _sync_selection_from_spectrogram(self, start: float, end: float):
        """Sync selection from spectrogram to other views."""
        self._waveform.blockSignals(True)
        self._annotation_editor.blockSignals(True)
        self._waveform.set_selection(start, end)
        self._annotation_editor.set_selection(start, end)
        self._waveform.blockSignals(False)
        self._annotation_editor.blockSignals(False)

    def _sync_selection_from_annotations(self, start: float, end: float):
        """Sync selection from annotations to other views."""
        self._waveform.blockSignals(True)
        self._spectrogram.blockSignals(True)
        self._waveform.set_selection(start, end)
        self._spectrogram.set_selection(start, end)
        self._waveform.blockSignals(False)
        self._spectrogram.blockSignals(False)

    def _sync_from_waveform(self, start: float, end: float):
        """Sync time range from waveform to other views."""
        self._spectrogram.blockSignals(True)
        self._annotation_editor.blockSignals(True)
        self._spectrogram.set_x_range(start, end)
        self._annotation_editor.set_x_range(start, end)
        self._spectrogram.blockSignals(False)
        self._annotation_editor.blockSignals(False)

    def _sync_from_spectrogram(self, start: float, end: float):
        """Sync time range from spectrogram to other views."""
        self._waveform.blockSignals(True)
        self._annotation_editor.blockSignals(True)
        self._waveform.setXRange(start, end, padding=0)
        self._annotation_editor.set_x_range(start, end)
        self._waveform.blockSignals(False)
        self._annotation_editor.blockSignals(False)

    def _sync_from_annotations(self, start: float, end: float):
        """Sync time range from annotations to other views."""
        self._waveform.blockSignals(True)
        self._spectrogram.blockSignals(True)
        self._waveform.setXRange(start, end, padding=0)
        self._spectrogram.set_x_range(start, end)
        self._waveform.blockSignals(False)
        self._spectrogram.blockSignals(False)

    def _on_cursor_moved(self, time: float):
        """Handle cursor movement."""
        self._time_label.setText(f"Time: {time:.3f}s")
        # Block signals to prevent recursion
        self._waveform.blockSignals(True)
        self._spectrogram.blockSignals(True)
        self._annotation_editor.blockSignals(True)
        self._waveform.set_cursor_position(time)
        self._spectrogram.set_cursor_position(time)
        self._annotation_editor.set_cursor_position(time)
        self._waveform.blockSignals(False)
        self._spectrogram.blockSignals(False)
        self._annotation_editor.blockSignals(False)

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

            # Reset state for new file
            self._textgrid_path = None
            self._is_dirty = False

            # Load audio
            self._audio_data = load_audio(file_path)
            self._player.set_audio_data(self._audio_data)

            # Display waveform
            self._waveform.set_audio_data(self._audio_data)

            # Initialize annotations with one default tier
            self._annotations = AnnotationSet(duration=self._audio_data.duration)
            self._annotations.add_tier("Annotation")
            self._annotation_editor.set_annotations(self._annotations)
            self._annotation_editor.set_x_range(0, self._audio_data.duration)

            # Update window title
            self._update_window_title()

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

    def _play_interval(self, start: float, end: float):
        """Play a specific interval (from annotation)."""
        if self._audio_data:
            self._player.play(start, end)
            self._playback_timer.start()
            self._play_btn.setText("❚❚ Pause")

    def _update_playback_cursor(self):
        """Update cursor position during playback."""
        if self._player.is_playing:
            time = self._player.current_time
            self._waveform.set_cursor_position(time)
            self._spectrogram.set_cursor_position(time)
            self._annotation_editor.set_cursor_position(time)
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

    def _pan_left(self):
        """Pan view to the left."""
        if self._audio_data is None:
            return
        start, end = self._waveform.get_view_range()
        width = end - start
        pan_amount = width * 0.25  # Pan by 25% of visible width
        new_start = max(0, start - pan_amount)
        new_end = new_start + width
        self._waveform.setXRange(new_start, new_end)

    def _pan_right(self):
        """Pan view to the right."""
        if self._audio_data is None:
            return
        start, end = self._waveform.get_view_range()
        width = end - start
        pan_amount = width * 0.25  # Pan by 25% of visible width
        new_end = min(self._audio_data.duration, end + pan_amount)
        new_start = new_end - width
        new_start = max(0, new_start)
        self._waveform.setXRange(new_start, new_end)

    def _zoom_to_selection(self):
        """Zoom to fit the current selection."""
        selection = self._waveform.get_selection()
        if selection is None:
            self._status_bar.showMessage("No selection to zoom to")
            return
        start, end = selection
        # Add small padding (5%)
        padding = (end - start) * 0.05
        self._waveform.setXRange(start - padding, end + padding)

    def _import_textgrid(self):
        """Import annotations from a TextGrid file (via dialog)."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import TextGrid",
            "",
            "TextGrid Files (*.TextGrid *.textgrid);;All Files (*)"
        )
        if file_path:
            self._import_textgrid_file(file_path)

    def _import_textgrid_file(self, file_path: str):
        """Import annotations from a TextGrid file path."""
        try:
            self._annotations = read_textgrid(file_path)
            if self._audio_data:
                self._annotations.duration = self._audio_data.duration
            self._annotation_editor.set_annotations(self._annotations)
            self._textgrid_path = file_path
            self._is_dirty = False
            self._update_window_title()
            self._autosave_timer.start()
            self._status_bar.showMessage(
                f"Opened {self._annotations.num_tiers} tier(s) from {Path(file_path).name}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Open Error", f"Failed to open TextGrid: {e}")

    def _create_predefined_tiers(self, tier_names: list[str]):
        """Create annotation tiers with predefined names."""
        if self._audio_data is None:
            return

        self._annotations = AnnotationSet(duration=self._audio_data.duration)
        for name in tier_names:
            self._annotations.add_tier(name)
        self._annotation_editor.set_annotations(self._annotations)
        self._is_dirty = True
        self._update_window_title()
        self._autosave_timer.start()
        self._status_bar.showMessage(
            f"Created {len(tier_names)} tier(s): {', '.join(tier_names)}"
        )

    def _setup_textgrid_path(self, file_path: str):
        """Setup a TextGrid path for saving (may be new or existing file)."""
        self._textgrid_path = file_path
        self._is_dirty = True
        self._update_window_title()
        self._autosave_timer.start()

    def setup_textgrid_from_path(self, file_path: str, tier_names: list[str] | None = None) -> bool:
        """Setup TextGrid from a path - load existing or create new with confirmation.

        Args:
            file_path: Path to TextGrid file (may or may not exist)
            tier_names: Tier names to create if file doesn't exist

        Returns:
            True if setup succeeded, False if user cancelled
        """
        path = Path(file_path)

        if path.exists():
            # Load existing file
            self._import_textgrid_file(file_path)
            return True
        else:
            # File doesn't exist - ask user if they want to create it
            tier_info = ""
            if tier_names:
                tier_info = f"\n\nTiers to create: {', '.join(tier_names)}"

            reply = QMessageBox.question(
                self,
                "Create New TextGrid",
                f"TextGrid file does not exist:\n{file_path}\n\n"
                f"Do you want to create it?{tier_info}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Create tiers (or default tier)
                if tier_names:
                    self._create_predefined_tiers(tier_names)
                # Set the path for saving
                self._textgrid_path = file_path
                self._update_window_title()
                self._status_bar.showMessage(
                    f"New TextGrid will be saved to {path.name}"
                )
                return True
            else:
                return False

    def _save_textgrid(self):
        """Save annotations to the current TextGrid path, or prompt for path."""
        if self._annotations is None or self._annotations.num_tiers == 0:
            QMessageBox.warning(self, "Save", "No annotations to save.")
            return

        if self._textgrid_path:
            # Save to existing path
            self._do_save_textgrid(self._textgrid_path)
        else:
            # No path set, use Save As
            self._save_textgrid_as()

    def _save_textgrid_as(self):
        """Save annotations to a new TextGrid file (always prompts)."""
        if self._annotations is None or self._annotations.num_tiers == 0:
            QMessageBox.warning(self, "Save", "No annotations to save.")
            return

        # Default filename based on audio file or existing textgrid path
        default_name = ""
        if self._textgrid_path:
            default_name = self._textgrid_path
        elif self._current_file_path:
            default_name = Path(self._current_file_path).stem + ".TextGrid"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save TextGrid As",
            default_name,
            "TextGrid Files (*.TextGrid);;All Files (*)"
        )
        if file_path:
            self._do_save_textgrid(file_path)
            self._textgrid_path = file_path
            self._update_window_title()

    def _do_save_textgrid(self, file_path: str):
        """Actually save the TextGrid to the given path."""
        try:
            write_textgrid(self._annotations, file_path)
            self._is_dirty = False
            self._update_window_title()
            self._status_bar.showMessage(f"Saved annotations to {Path(file_path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save TextGrid: {e}")

    def _import_tsv(self):
        """Import annotations from a TSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import TSV",
            "",
            "TSV Files (*.tsv *.txt);;All Files (*)"
        )
        if file_path:
            try:
                self._annotations = read_tsv(file_path)
                if self._audio_data:
                    self._annotations.duration = self._audio_data.duration
                self._annotation_editor.set_annotations(self._annotations)
                self._status_bar.showMessage(
                    f"Imported {self._annotations.num_tiers} tier(s) from {Path(file_path).name}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import TSV: {e}")

    def _export_tsv(self):
        """Export annotations to a TSV file."""
        if self._annotations is None or self._annotations.num_tiers == 0:
            QMessageBox.warning(self, "Export", "No annotations to export.")
            return

        default_name = ""
        if self._current_file_path:
            default_name = Path(self._current_file_path).stem + ".tsv"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export TSV",
            default_name,
            "TSV Files (*.tsv);;All Files (*)"
        )
        if file_path:
            try:
                write_tsv(self._annotations, file_path)
                self._status_bar.showMessage(f"Exported annotations to {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export TSV: {e}")

    def _add_tier(self):
        """Add a new annotation tier."""
        if self._annotations is None:
            if self._audio_data is None:
                QMessageBox.warning(self, "Add Tier", "Load an audio file first.")
                return
            self._annotations = AnnotationSet(duration=self._audio_data.duration)

        if self._annotations.num_tiers >= AnnotationSet.MAX_TIERS:
            QMessageBox.warning(
                self, "Add Tier",
                f"Maximum number of tiers ({AnnotationSet.MAX_TIERS}) reached."
            )
            return

        name, ok = QInputDialog.getText(
            self, "Add Tier", "Tier name:",
            text=f"Tier {self._annotations.num_tiers + 1}"
        )
        if ok and name:
            self._annotations.add_tier(name)
            self._annotation_editor.set_annotations(self._annotations)
            self._status_bar.showMessage(f"Added tier: {name}")

    def _remove_tier(self):
        """Remove the active annotation tier."""
        if self._annotations is None or self._annotations.num_tiers == 0:
            QMessageBox.warning(self, "Remove Tier", "No tiers to remove.")
            return

        tier = self._annotations.active_tier
        if tier is None:
            return

        reply = QMessageBox.question(
            self, "Remove Tier",
            f"Remove tier '{tier.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._annotations.remove_tier(self._annotations.active_tier_index)
            self._annotation_editor.set_annotations(self._annotations)
            self._status_bar.showMessage(f"Removed tier: {tier.name}")

    def _rename_tier(self):
        """Rename the active annotation tier."""
        if self._annotations is None or self._annotations.num_tiers == 0:
            QMessageBox.warning(self, "Rename Tier", "No tiers to rename.")
            return

        tier = self._annotations.active_tier
        if tier is None:
            return

        name, ok = QInputDialog.getText(
            self, "Rename Tier", "New name:", text=tier.name
        )
        if ok and name:
            old_name = tier.name
            self._annotations.rename_tier(self._annotations.active_tier_index, name)
            self._annotation_editor.refresh()
            self._status_bar.showMessage(f"Renamed tier: {old_name} → {name}")

    def _add_boundary_at_cursor(self):
        """Add a boundary at the current cursor position."""
        self._annotation_editor.add_boundary_at_cursor()

    def _remove_nearest_boundary(self):
        """Remove the boundary nearest to the cursor."""
        self._annotation_editor.remove_nearest_boundary()

    def _play_interval_at_cursor(self):
        """Play the interval at the current cursor position."""
        self._annotation_editor.play_interval_at_cursor()

    def _mark_dirty(self, *args):
        """Mark annotations as having unsaved changes."""
        if not self._is_dirty:
            self._is_dirty = True
            self._update_window_title()

    def _update_window_title(self):
        """Update window title to show file name and dirty state."""
        title = "WaveAnnotator"
        if self._current_file_path:
            title = f"WaveAnnotator - {Path(self._current_file_path).name}"
        if self._textgrid_path:
            title += f" [{Path(self._textgrid_path).name}]"
        if self._is_dirty:
            title += " *"
        self.setWindowTitle(title)

    def _auto_save(self):
        """Auto-save annotations to a backup file."""
        if self._annotations is None or self._annotations.num_tiers == 0:
            return
        if not self._is_dirty:
            return

        # Determine backup path
        if self._textgrid_path:
            backup_path = self._textgrid_path + ".autosave"
        elif self._current_file_path:
            backup_path = Path(self._current_file_path).stem + ".TextGrid.autosave"
            backup_path = str(Path(self._current_file_path).parent / backup_path)
        else:
            # Use temp directory
            backup_path = os.path.join(tempfile.gettempdir(), "waveannotator_autosave.TextGrid")

        try:
            write_textgrid(self._annotations, backup_path)
            self._status_bar.showMessage(f"Auto-saved to {Path(backup_path).name}", 3000)
        except Exception as e:
            self._status_bar.showMessage(f"Auto-save failed: {e}", 5000)

    def closeEvent(self, event):
        """Handle window close - prompt if unsaved changes."""
        if self._is_dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before closing?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Save:
                self._save_textgrid()
                # If still dirty (save was cancelled or failed), don't close
                if self._is_dirty:
                    event.ignore()
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            # Discard - just close

        # Stop playback and timers
        self._player.stop()
        self._playback_timer.stop()
        self._autosave_timer.stop()

        # Clean up autosave file if it exists and we saved successfully
        if not self._is_dirty:
            if self._textgrid_path:
                autosave_path = self._textgrid_path + ".autosave"
                if os.path.exists(autosave_path):
                    try:
                        os.remove(autosave_path)
                    except OSError:
                        pass

        event.accept()

    def eventFilter(self, obj, event):
        """Filter events to catch key presses for text input."""
        if event.type() == QEvent.Type.KeyPress:
            # Don't intercept if the text editor is active - let it handle input naturally
            if self._annotation_editor.is_editing_text():
                # Only handle Escape to close the editor
                if event.key() == Qt.Key.Key_Escape:
                    self._annotation_editor._hide_text_editor()
                    self._annotation_editor.deselect_interval()
                    return True
                # Let the QLineEdit handle all other keys
                return False

            # Check if there's a selected interval but editor not shown yet
            selected = self._annotation_editor.get_selected_interval()
            if selected is not None:
                key = event.key()

                if key == Qt.Key.Key_Escape:
                    self._annotation_editor.deselect_interval()
                    return True
                elif key == Qt.Key.Key_Space:
                    self._annotation_editor.play_selected_interval()
                    return True

        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        key = event.key()

        # Don't handle keys if text editor is active
        if self._annotation_editor.is_editing_text():
            super().keyPressEvent(event)
            return

        # No selected interval or text editor not active - handle global shortcuts
        if key == Qt.Key.Key_Space:
            self._toggle_playback()
        elif key == Qt.Key.Key_Escape:
            self._stop_playback()
        elif key == Qt.Key.Key_Tab:
            # Play visible window
            if self._audio_data:
                start, end = self._waveform.get_view_range()
                start = max(0, start)
                end = min(self._audio_data.duration, end)
                self._player.play(start, end)
                self._playback_timer.start()
                self._play_btn.setText("❚❚ Pause")
        elif key == Qt.Key.Key_Left:
            self._pan_left()
        elif key == Qt.Key.Key_Right:
            self._pan_right()
        elif key == Qt.Key.Key_Z:
            self._zoom_to_selection()
        elif key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            # Add boundary when no interval is selected
            self._add_boundary_at_cursor()
        elif key == Qt.Key.Key_Delete:
            # Remove boundary
            self._remove_nearest_boundary()
        else:
            super().keyPressEvent(event)
