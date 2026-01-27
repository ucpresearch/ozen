"""
Main application window for Ozen.

This module contains the MainWindow class, which is the central hub of
the application. It coordinates all the major components:

Components:
    - WaveformWidget: Displays the audio waveform
    - SpectrogramWidget: Displays spectrogram with acoustic overlays
    - AnnotationEditorWidget: Displays and edits annotation tiers
    - AudioPlayer: Handles audio playback
    - Control panel: Settings for spectrogram, formants, overlays

Responsibilities:
    - Layout management (splitter between views)
    - Menu and toolbar creation
    - Keyboard shortcut handling
    - Synchronization between views (time range, cursor, selection)
    - File I/O (audio, TextGrid, TSV)
    - Feature extraction (background thread)
    - Save/autosave state management
    - Playback control

Signal Connections:
    The main window connects signals between the three display widgets
    to keep them synchronized. When one widget changes its view range,
    cursor position, or selection, the change is propagated to the others.

Threading:
    Feature extraction runs in a background thread (FeatureExtractionThread)
    to keep the UI responsive. Progress is reported via Qt signals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import traceback
import tempfile
import os

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QMetaObject, Q_ARG, Qt as QtCore, QEvent
from PyQt6.QtGui import QAction, QKeySequence, QKeyEvent, QShortcut
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QProgressDialog, QStatusBar, QToolBar,
    QCheckBox, QComboBox, QLabel, QSplitter, QGroupBox,
    QMessageBox, QPushButton, QInputDialog
)

from ..audio.loader import load_audio, AudioData
from ..audio.player import AudioPlayer
from ..analysis.acoustic import extract_features, compute_spectrogram, AcousticFeatures
from ..analysis import (
    get_available_backends_display, get_current_backend_display, switch_backend
)
from ..visualization.waveform import WaveformWidget
from ..visualization.spectrogram import SpectrogramWidget
from ..annotation import (
    AnnotationSet, AnnotationEditorWidget,
    read_textgrid, write_textgrid, read_tsv, write_tsv
)
from ..config import config, reload_config


class SpectrogramComputeThread(QThread):
    """
    Background thread for spectrogram computation.

    Computes spectrogram in a separate thread to keep UI responsive
    during panning and zooming.

    Signals:
        finished(tuple): (times, freqs, spec_db, start_time, end_time)
        error(str): Error message if computation fails
    """

    finished = pyqtSignal(object)  # tuple of (times, freqs, spec_db, start, end)
    error = pyqtSignal(str)

    def __init__(self, file_path: str, window_length: float, max_frequency: float,
                 start_time: float, end_time: float):
        super().__init__()
        self.file_path = file_path
        self.window_length = window_length
        self.max_frequency = max_frequency
        self.start_time = start_time
        self.end_time = end_time

    def run(self):
        try:
            times, freqs, spec_db = compute_spectrogram(
                self.file_path,
                window_length=self.window_length,
                max_frequency=self.max_frequency,
                dynamic_range=70.0,
                start_time=self.start_time,
                end_time=self.end_time
            )
            self.finished.emit((times, freqs, spec_db, self.start_time, self.end_time))
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class FeatureExtractionThread(QThread):
    """
    Background thread for acoustic feature extraction.

    Runs the computationally expensive feature extraction in a separate
    thread to keep the UI responsive. Reports progress and results via
    Qt signals.

    Signals:
        progress(float): Extraction progress from 0.0 to 1.0
        finished(AcousticFeatures): Successful extraction result
        error(str): Error message if extraction fails

    Usage:
        thread = FeatureExtractionThread(file_path, preset='female')
        thread.progress.connect(update_progress_bar)
        thread.finished.connect(on_features_ready)
        thread.error.connect(on_extraction_error)
        thread.start()
    """

    progress = pyqtSignal(float)
    finished = pyqtSignal(object)  # AcousticFeatures or Exception
    error = pyqtSignal(str)

    def __init__(self, file_path: str, time_step: float = None, preset: str = 'male',
                 start_time: float = None, end_time: float = None):
        super().__init__()
        self.file_path = file_path
        # Use time_step from config if not specified
        self.time_step = time_step if time_step is not None else config['analysis']['time_step']
        self.preset = preset
        self.start_time = start_time
        self.end_time = end_time

    def run(self):
        try:
            # Get formant preset from config (for max_formant only)
            presets = config['formant_presets']
            params = presets.get(self.preset, presets['male'])
            # Use global pitch settings (like Praat) - works for all voice types
            analysis_cfg = config['analysis']
            features = extract_features(
                self.file_path,
                time_step=self.time_step,
                max_formant=params['max_formant'],
                pitch_floor=analysis_cfg['default_pitch_floor'],
                pitch_ceiling=analysis_cfg['default_pitch_ceiling'],
                start_time=self.start_time,
                end_time=self.end_time,
                progress_callback=lambda p: self.progress.emit(p)
            )
            self.finished.emit(features)
        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    """
    Main application window for Ozen.

    This is the central class that creates and coordinates all UI components.
    It manages the application state including:
    - Currently loaded audio file
    - Extracted acoustic features
    - Annotation data and save state
    - Playback state

    The window layout consists of a vertical splitter with three panels:
    1. Waveform display (top)
    2. Spectrogram with overlays (middle)
    3. Annotation editor (bottom)

    Below the panels is a control panel with settings and a status bar.

    Key methods:
        _load_audio_file(): Load audio and compute spectrogram
        _start_feature_extraction(): Begin background feature extraction
        _save_textgrid() / _save_textgrid_as(): Save annotations
        _toggle_playback(): Play/pause audio
        setup_textgrid_from_path(): Load or create TextGrid from path

    State tracking:
        _is_dirty: True if annotations have unsaved changes
        _textgrid_path: Current save path for annotations
        _autosave_timer: Timer for periodic autosave
    """

    # Signal for thread-safe playback finished notification
    # (audio callback runs on separate thread, needs signal to update UI)
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
        self._last_points_dir: str | None = None  # Last directory used for point export/import

        # Global undo stack: list of ('annotation' | 'data_point') entries
        # Records the order of undoable operations across both systems
        self._global_undo_stack: list[str] = []
        self._max_global_undo = 100
        self._is_undoing = False  # Flag to prevent re-pushing during undo

        # Default tier names (from config or CLI --tiers)
        # Used when opening new files without an explicit TextGrid
        self._default_tier_names: list[str] = config['annotation'].get('default_tiers', [])

        # Spectrogram computation state
        self._spectrogram_thread: SpectrogramComputeThread | None = None
        self._spectrogram_debounce_timer: QTimer | None = None
        self._pending_spectrogram_range: tuple[float, float] | None = None

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

        # Setup global keyboard shortcuts
        self._setup_shortcuts()

    def _setup_shortcuts(self):
        """Setup global keyboard shortcuts that work regardless of focus."""
        # Install event filter on application to catch Ctrl+Z/Cmd+Z globally
        from PyQt6.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self)

    def _global_undo(self):
        """Handle global undo across all systems."""
        self._is_undoing = True
        try:
            while self._global_undo_stack:
                source = self._global_undo_stack.pop()

                if source == 'data_point':
                    if self._spectrogram.undo_data_point():
                        self._status_bar.showMessage("Undo", 2000)
                        return
                elif source == 'annotation':
                    if self._annotation_editor.undo():
                        self._status_bar.showMessage("Undo", 2000)
                        return
                # If undo failed, continue to try next item in stack
        finally:
            self._is_undoing = False

    def _setup_ui(self):
        """Setup the main UI layout."""
        self.setWindowTitle("Ozen")
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

        # Max frequency
        self._max_freq_combo = QComboBox()
        self._max_freq_combo.addItems(['5000 Hz', '7500 Hz', '10000 Hz'])
        self._max_freq_combo.setToolTip("Maximum frequency displayed on spectrogram")
        self._max_freq_combo.currentTextChanged.connect(self._on_max_freq_changed)
        spec_layout.addWidget(QLabel("Max:"))
        spec_layout.addWidget(self._max_freq_combo)

        layout.addWidget(spec_group)

        # Formant settings
        formant_group = QGroupBox("Formants")
        formant_layout = QHBoxLayout(formant_group)

        self._formant_preset_combo = QComboBox()
        self._formant_preset_combo.addItems(['Female', 'Male', 'Child'])
        self._formant_preset_combo.setToolTip(
            "Formant analysis preset (affects max formant frequency):\n"
            "Female: max 5500 Hz\nMale: max 5000 Hz\nChild: max 8000 Hz"
        )
        self._formant_preset_combo.currentTextChanged.connect(self._on_formant_preset_changed)
        formant_layout.addWidget(QLabel("Voice:"))
        formant_layout.addWidget(self._formant_preset_combo)

        layout.addWidget(formant_group)

        # =====================================================================
        # Backend selector - allows runtime switching of acoustic analysis engine
        # =====================================================================
        # Available backends (varies by installation):
        #   - Praatfan (slow): Pure Python implementation (MIT license, portable)
        #   - Praatfan (fast): Rust implementation praatfan_rust (MIT, fast)
        #   - Praatfan (GPL): Rust implementation praatfan_gpl (GPL)
        #   - Praat: Original Praat via parselmouth bindings (GPL, reference impl)
        # The 'Praat' option is always listed last since it requires GPL compliance.
        # Switching backends triggers automatic re-extraction of features.
        backend_group = QGroupBox("Backend")
        backend_layout = QHBoxLayout(backend_group)
        self._backend_combo = QComboBox()
        self._backend_combo.addItems(get_available_backends_display())
        self._backend_combo.setCurrentText(get_current_backend_display())
        self._backend_combo.setToolTip(
            "Acoustic analysis backend:\n"
            "Praatfan (slow): Pure Python (MIT)\n"
            "Praatfan (fast): Rust praatfan_rust (MIT)\n"
            "Praatfan (GPL): Rust praatfan_gpl (GPL)\n"
            "Praat: Original Praat bindings (GPL)"
        )
        self._backend_combo.currentTextChanged.connect(self._on_backend_changed)
        backend_layout.addWidget(QLabel("Engine:"))
        backend_layout.addWidget(self._backend_combo)
        layout.addWidget(backend_group)

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

        self._tilt_check = QCheckBox("Tilt")
        self._tilt_check.setChecked(False)
        self._tilt_check.setToolTip("Spectral tilt (low vs high frequency energy)")
        self._tilt_check.toggled.connect(lambda v: self._spectrogram.set_track_visible('spectral_tilt', v))
        overlay_layout.addWidget(self._tilt_check)

        self._a1p0_check = QCheckBox("A1-P0")
        self._a1p0_check.setChecked(False)
        self._a1p0_check.setToolTip("A1-P0 nasal ratio: amplitude at F0 minus amplitude at ~250Hz (requires voicing)")
        self._a1p0_check.toggled.connect(lambda v: self._spectrogram.set_track_visible('a1p0', v))
        overlay_layout.addWidget(self._a1p0_check)

        self._nasal_murmur_check = QCheckBox("NMR")
        self._nasal_murmur_check.setChecked(False)
        self._nasal_murmur_check.setToolTip("Nasal murmur ratio: low-freq energy (0-500Hz) / total energy (0-5000Hz)")
        self._nasal_murmur_check.toggled.connect(lambda v: self._spectrogram.set_track_visible('nasal_murmur', v))
        overlay_layout.addWidget(self._nasal_murmur_check)

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

        import_points = QAction("Import &Point Information...", self)
        import_points.triggered.connect(self._import_points_tsv)
        file_menu.addAction(import_points)

        export_points = QAction("Export P&oint Information...", self)
        export_points.triggered.connect(self._export_points_tsv)
        file_menu.addAction(export_points)

        file_menu.addSeparator()

        load_config_action = QAction("Load &Config...", self)
        load_config_action.triggered.connect(self._load_config_dialog)
        file_menu.addAction(load_config_action)

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

        # Track annotation changes for global undo stack
        self._annotation_editor.boundary_added.connect(self._on_annotation_changed)
        self._annotation_editor.boundary_removed.connect(self._on_annotation_changed)
        self._annotation_editor.boundary_moved.connect(self._on_annotation_changed)
        self._annotation_editor.text_edit_finished.connect(self._on_annotation_changed)

        # Track text changes for dirty state only (not undo - that's handled by text_edit_finished)
        self._annotation_editor.interval_text_changed.connect(self._mark_dirty)

        # Data point integration
        self._spectrogram.point_added.connect(self._on_data_point_changed)
        self._spectrogram.point_removed.connect(self._on_data_point_changed)
        self._spectrogram.point_moved.connect(self._on_data_point_changed)

        # Player callbacks
        # Position updates are handled by _playback_timer polling current_time
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

        # Check if we need to compute/hide spectrogram based on view duration
        self._check_spectrogram_threshold(start, end)

    def _sync_from_spectrogram(self, start: float, end: float):
        """Sync time range from spectrogram to other views."""
        self._waveform.blockSignals(True)
        self._annotation_editor.blockSignals(True)
        self._waveform.setXRange(start, end, padding=0)
        self._annotation_editor.set_x_range(start, end)
        self._waveform.blockSignals(False)
        self._annotation_editor.blockSignals(False)

        # Check if we need to compute/hide spectrogram based on view duration
        self._check_spectrogram_threshold(start, end)

    def _sync_from_annotations(self, start: float, end: float):
        """Sync time range from annotations to other views."""
        self._waveform.blockSignals(True)
        self._spectrogram.blockSignals(True)
        self._waveform.setXRange(start, end, padding=0)
        self._spectrogram.set_x_range(start, end)
        self._waveform.blockSignals(False)
        self._spectrogram.blockSignals(False)

        # Check if we need to compute/hide spectrogram based on view duration
        self._check_spectrogram_threshold(start, end)

    def _check_spectrogram_threshold(self, start: float, end: float):
        """Check if spectrogram should be computed or hidden based on view duration.

        When the visible view is ≤ the threshold (default 60s), schedules
        spectrogram computation for that region (debounced). When the view
        is > threshold, shows a placeholder message instead.

        Uses caching with two checks:
        1. Coverage: if existing spectrogram covers >80% of the visible region
        2. Resolution: if user has zoomed in >2x, recompute at higher resolution

        Uses debouncing: waits 200ms after user stops panning before computing.
        """
        if not self._current_file_path or self._audio_data is None:
            return

        view_duration = end - start
        max_view_duration = config['analysis'].get('spectrogram_max_view_duration', 60.0)

        if view_duration <= max_view_duration:
            # View is within threshold - check if we need to compute spectrogram
            current_range = self._spectrogram.get_spectrogram_time_range()

            needs_computation = False
            if not self._spectrogram.has_spectrogram():
                needs_computation = True
            elif current_range is not None:
                spec_start, spec_end = current_range
                spec_duration = spec_end - spec_start

                # Check 1: Coverage - does current spectrogram cover visible region?
                overlap_start = max(start, spec_start)
                overlap_end = min(end, spec_end)
                overlap = max(0, overlap_end - overlap_start)
                coverage = overlap / view_duration if view_duration > 0 else 0

                # Check 2: Resolution - is user zoomed in significantly?
                # If spectrogram covers much more than visible, it's being stretched
                zoom_ratio = spec_duration / view_duration if view_duration > 0 else 1

                # Recompute if:
                # - Coverage is less than 80%, OR
                # - Zoomed in more than 2x (spectrogram would be pixelated)
                if coverage < 0.8 or zoom_ratio > 2.0:
                    needs_computation = True

            if needs_computation:
                # Schedule debounced computation
                self._schedule_spectrogram_computation(start, end)
        else:
            # View is too wide - show placeholder and cancel any pending computation
            self._cancel_spectrogram_computation()
            self._spectrogram.show_placeholder("Zoom in for spectrogram")

    def _schedule_spectrogram_computation(self, start: float, end: float):
        """Schedule spectrogram computation with debouncing.

        Waits 200ms after user stops panning before starting computation.
        This prevents computing spectrograms during rapid pan/zoom.
        """
        # Minimum duration for spectrogram (need enough samples for FFT window)
        min_duration = 0.1  # 100ms minimum

        view_duration = end - start
        if view_duration < min_duration:
            # View too small for meaningful spectrogram
            return

        # Add padding for smoother panning (20% on each side for more coverage)
        padding = view_duration * 0.2
        padded_start = max(0, start - padding)
        padded_end = min(self._audio_data.duration, end + padding)

        self._pending_spectrogram_range = (padded_start, padded_end)

        # Cancel existing timer
        if self._spectrogram_debounce_timer is not None:
            self._spectrogram_debounce_timer.stop()

        # Don't schedule if already computing the same range
        if (self._spectrogram_thread is not None and
            self._spectrogram_thread.isRunning() and
            self._spectrogram_thread.start_time == padded_start and
            self._spectrogram_thread.end_time == padded_end):
            return

        # Create debounce timer (200ms delay)
        self._spectrogram_debounce_timer = QTimer()
        self._spectrogram_debounce_timer.setSingleShot(True)
        self._spectrogram_debounce_timer.timeout.connect(self._start_spectrogram_computation)
        self._spectrogram_debounce_timer.start(200)

    def _cancel_spectrogram_computation(self):
        """Cancel any pending spectrogram computation."""
        if self._spectrogram_debounce_timer is not None:
            self._spectrogram_debounce_timer.stop()
            self._spectrogram_debounce_timer = None
        self._pending_spectrogram_range = None

    def _start_spectrogram_computation(self):
        """Start background spectrogram computation."""
        if self._pending_spectrogram_range is None:
            return

        start, end = self._pending_spectrogram_range

        # Don't start if already computing
        if self._spectrogram_thread is not None and self._spectrogram_thread.isRunning():
            # Re-schedule to run after current computation
            self._spectrogram_debounce_timer = QTimer()
            self._spectrogram_debounce_timer.setSingleShot(True)
            self._spectrogram_debounce_timer.timeout.connect(self._start_spectrogram_computation)
            self._spectrogram_debounce_timer.start(100)
            return

        # Get current settings
        bandwidth = self._bandwidth_combo.currentText()
        window_length = 0.005 if bandwidth == 'Wideband' else 0.025
        max_freq = self._get_max_frequency()

        self._status_bar.showMessage("Computing spectrogram...")

        # Start background thread
        self._spectrogram_thread = SpectrogramComputeThread(
            self._current_file_path,
            window_length=window_length,
            max_frequency=max_freq,
            start_time=start,
            end_time=end
        )
        self._spectrogram_thread.finished.connect(self._on_spectrogram_computed)
        self._spectrogram_thread.error.connect(self._on_spectrogram_error)
        self._spectrogram_thread.start()

    def _on_spectrogram_computed(self, result: tuple):
        """Handle completed spectrogram computation."""
        times, freqs, spec_db, start, end = result

        # Check if view has changed significantly since we started
        current_start, current_end = self._waveform.get_view_range()
        view_duration = current_end - current_start
        max_view_duration = config['analysis'].get('spectrogram_max_view_duration', 60.0)

        # Only apply if view is still within threshold
        if view_duration <= max_view_duration:
            self._spectrogram.set_spectrogram(times, freqs, spec_db)
            self._status_bar.showMessage(
                f"Spectrogram ready ({end - start:.1f}s)", 2000
            )

            # Also extract features for this region if not already running
            if self._feature_thread is None or not self._feature_thread.isRunning():
                self._start_feature_extraction_for_view(start, end)
        else:
            self._status_bar.showMessage("", 0)

        # Check if there's a new pending range that needs computation
        if self._pending_spectrogram_range is not None:
            pending_start, pending_end = self._pending_spectrogram_range
            if pending_start != start or pending_end != end:
                # New range requested, schedule computation
                self._schedule_spectrogram_computation(
                    current_start, current_end
                )

    def _on_spectrogram_error(self, error_msg: str):
        """Handle spectrogram computation error."""
        self._status_bar.showMessage("Spectrogram computation failed", 3000)
        print(f"Spectrogram error: {error_msg}")

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
            self._global_undo_stack.clear()
            self._features = None

            # Clear data points from previous file
            self._spectrogram.clear_data_points()

            # Load audio
            self._audio_data = load_audio(file_path)
            self._player.set_audio_data(self._audio_data)

            # Set audio duration and frequency range for spectrogram
            # This also sets the initial view to full audio and clears any previous spectrogram
            self._spectrogram.set_audio_duration(
                self._audio_data.duration,
                max_frequency=self._get_max_frequency()
            )

            # Display waveform (this also sets waveform view to full duration)
            self._waveform.set_audio_data(self._audio_data)

            # Initialize annotations with default tiers (from config/CLI or fallback)
            self._annotations = AnnotationSet(duration=self._audio_data.duration)
            tier_names = self._default_tier_names if self._default_tier_names else ["Annotation"]
            for name in tier_names:
                self._annotations.add_tier(name)
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

    def _on_max_freq_changed(self, freq_str: str):
        """Handle max frequency change."""
        if self._current_file_path:
            self._recompute_spectrogram()

    def _get_max_frequency(self) -> float:
        """Get the currently selected max frequency in Hz."""
        freq_str = self._max_freq_combo.currentText()
        # Parse "5000 Hz" -> 5000.0
        return float(freq_str.replace(' Hz', ''))

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

    def _on_backend_changed(self, backend: str):
        """Handle acoustic analysis backend change.

        Called when user selects a different backend from the dropdown.
        Switches the praatfan backend and automatically re-extracts features
        if a file is loaded and short enough for auto-extraction.

        Args:
            backend: Display name of the selected backend (e.g., 'praatfan', 'praat')
        """
        if switch_backend(backend):
            self._status_bar.showMessage(f"Switched to {backend} backend")
            # Re-extract features automatically if:
            # 1. Features were already extracted (user expects to see results)
            # 2. Audio file is loaded
            # 3. File is short enough (≤60s) for quick re-extraction
            # 4. No extraction is currently in progress
            if (self._features is not None and
                self._audio_data is not None and
                self._audio_data.duration <= 60.0 and
                (self._feature_thread is None or not self._feature_thread.isRunning())):
                self._status_bar.showMessage(
                    f"Backend changed to {backend}. Re-extracting features..."
                )
                self._start_feature_extraction()
        else:
            # Backend switch failed (backend not available)
            self._status_bar.showMessage(f"Failed to switch to {backend} backend")
            # Reset combo to show the actual current backend
            # Block signals to prevent recursive call
            self._backend_combo.blockSignals(True)
            self._backend_combo.setCurrentText(get_current_backend_display())
            self._backend_combo.blockSignals(False)

    def _recompute_spectrogram(self):
        """Recompute spectrogram with current settings (for visible view)."""
        if not self._current_file_path:
            return

        # Clear current spectrogram to force recomputation
        self._spectrogram.clear_spectrogram()

        # Get visible view range
        start, end = self._waveform.get_view_range()
        view_duration = end - start

        # Get max view duration threshold from config
        max_view_duration = config['analysis'].get('spectrogram_max_view_duration', 60.0)

        if view_duration > max_view_duration:
            self._spectrogram.show_placeholder("Zoom in for spectrogram")
        else:
            # Use async computation
            self._schedule_spectrogram_computation(start, end)

    def _compute_analysis(self, file_path: str):
        """Initialize analysis for loaded audio file - all computation is async."""
        try:
            self._current_file_path = file_path
            self._current_formant_preset = 'female'  # Default

            # Get max view duration threshold from config
            max_view_duration = config['analysis'].get('spectrogram_max_view_duration', 60.0)

            # Enable playback and feature extraction
            self._play_btn.setEnabled(True)
            self._stop_btn.setEnabled(True)
            self._extract_btn.setEnabled(True)
            self._extract_btn.setText("Extract Features")

            # Check if file is too long for immediate spectrogram
            if self._audio_data.duration > max_view_duration:
                # Show placeholder - spectrogram will be computed when zoomed in
                self._spectrogram.show_placeholder("Zoom in for spectrogram")
                self._status_bar.showMessage(
                    f"Loaded: {Path(file_path).name} "
                    f"({self._audio_data.duration:.2f}s) - Zoom to ≤{max_view_duration:.0f}s for spectrogram"
                )
            else:
                # File is short enough - compute spectrogram async for entire file
                self._status_bar.showMessage(
                    f"Loaded: {Path(file_path).name} - Computing spectrogram..."
                )
                # Set the view to show the whole file
                self._waveform.setXRange(0, self._audio_data.duration, padding=0)
                # Schedule async spectrogram computation (no debounce for initial load)
                self._pending_spectrogram_range = (0, self._audio_data.duration)
                self._start_spectrogram_computation()

                # Auto-extract features for files under 60 seconds
                QTimer.singleShot(100, self._start_feature_extraction)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {e}")
            self._status_bar.showMessage("Analysis error")

    def _start_feature_extraction(self):
        """Start feature extraction in background thread.

        For large files, extracts only the visible region.
        """
        if not self._current_file_path:
            return

        # For large files, extract only the visible region
        max_view_duration = config['analysis'].get('spectrogram_max_view_duration', 60.0)
        start_time = None
        end_time = None

        if self._audio_data and self._audio_data.duration > max_view_duration:
            start, end = self._waveform.get_view_range()
            view_duration = end - start
            if view_duration <= max_view_duration:
                start_time = max(0, start)
                end_time = min(self._audio_data.duration, end)

        self._start_feature_extraction_for_view(start_time, end_time)

    def _start_feature_extraction_for_view(self, start_time: float = None, end_time: float = None):
        """Start feature extraction for a specific time range.

        Args:
            start_time: Start time (None = beginning)
            end_time: End time (None = end of file)
        """
        if not self._current_file_path:
            return

        self._extract_btn.setEnabled(False)
        self._extract_btn.setText("Extracting...")

        # Get current formant preset
        preset = getattr(self, '_current_formant_preset', 'male')

        if start_time is not None and end_time is not None:
            self._status_bar.showMessage(
                f"Extracting features for {start_time:.1f}-{end_time:.1f}s ({preset} preset)..."
            )
        else:
            self._status_bar.showMessage(
                f"Extracting features ({preset} preset)..."
            )

        # Use larger time_step (0.02s) for faster extraction while maintaining
        # sufficient resolution for display (50 frames/sec)
        self._feature_thread = FeatureExtractionThread(
            self._current_file_path,
            time_step=0.02,
            preset=preset,
            start_time=start_time,
            end_time=end_time
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

    def _load_config_dialog(self):
        """Load a config file via dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Config",
            "",
            "Config Files (*.yaml *.yml *.json);;All Files (*)"
        )
        if file_path:
            try:
                reload_config(file_path)
                self._status_bar.showMessage(f"Loaded config from {Path(file_path).name}")
                QMessageBox.information(
                    self,
                    "Config Loaded",
                    f"Configuration loaded from:\n{file_path}\n\n"
                    "Note: Some settings (like colors) will take effect "
                    "when you reload the audio file or restart the application."
                )
            except Exception as e:
                QMessageBox.critical(self, "Config Error", f"Failed to load config: {e}")

    def _create_predefined_tiers(self, tier_names: list[str]):
        """Create annotation tiers with predefined names."""
        if self._audio_data is None:
            return

        self._annotations = AnnotationSet(duration=self._audio_data.duration)
        for name in tier_names:
            self._annotations.add_tier(name)
        self._annotation_editor.set_annotations(self._annotations)
        # Don't mark as dirty - empty tiers are not unsaved work
        self._update_window_title()
        self._autosave_timer.start()
        self._status_bar.showMessage(
            f"Created {len(tier_names)} tier(s): {', '.join(tier_names)}"
        )

    def set_default_tier_names(self, tier_names: list[str]):
        """Set the default tier names to use when opening new files.

        Args:
            tier_names: List of tier names. If empty, falls back to ["Annotation"].
        """
        self._default_tier_names = tier_names

    def _setup_textgrid_path(self, file_path: str):
        """Setup a TextGrid path for saving (may be new or existing file)."""
        self._textgrid_path = file_path
        # Don't mark as dirty - just setting up a path is not unsaved work
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

    def _get_annotation_intervals_at_time(self, time: float) -> dict[str, str]:
        """Get annotation intervals at a specific time.

        This is called by the spectrogram to populate data points with
        annotation context.

        Args:
            time: Time position in seconds

        Returns:
            Dict mapping tier name to interval text at that time
        """
        intervals = {}
        if self._annotations is None:
            return intervals

        for tier in self._annotations.get_tiers():
            try:
                _, interval = tier.get_interval_at_time(time)
                intervals[tier.name] = interval.text
            except (ValueError, IndexError):
                intervals[tier.name] = ""

        return intervals

    def _import_points_tsv(self):
        """Import data collection points from a TSV file."""
        # Use last directory or current file's directory
        start_dir = self._last_points_dir
        if not start_dir and self._current_file_path:
            start_dir = str(Path(self._current_file_path).parent)

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Point Information",
            start_dir or "",
            "TSV Files (*.tsv);;All Files (*)"
        )
        if file_path:
            try:
                self._last_points_dir = str(Path(file_path).parent)
                data_points = self._spectrogram.get_data_points()
                count = data_points.import_tsv(file_path)

                # Create visual items for imported points
                for point in data_points.points:
                    if point.id not in self._spectrogram._data_point_items:
                        self._spectrogram._create_data_point_item(point)

                self._status_bar.showMessage(
                    f"Imported {count} point(s) from {Path(file_path).name}"
                )
                self._mark_dirty()
            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import points: {e}")

    def _export_points_tsv(self):
        """Export data collection points to a TSV file."""
        data_points = self._spectrogram.get_data_points()
        if not data_points.points:
            QMessageBox.warning(self, "Export", "No data points to export.")
            return

        # Use last directory or current file's directory
        start_dir = self._last_points_dir
        if not start_dir and self._current_file_path:
            start_dir = str(Path(self._current_file_path).parent)

        default_name = ""
        if self._current_file_path:
            default_name = Path(self._current_file_path).stem + "_points.tsv"
            if start_dir:
                default_name = str(Path(start_dir) / default_name)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Point Information",
            default_name,
            "TSV Files (*.tsv);;All Files (*)"
        )
        if file_path:
            try:
                self._last_points_dir = str(Path(file_path).parent)

                # Get tier names for column ordering
                tier_names = None
                if self._annotations:
                    tier_names = [tier.name for tier in self._annotations.get_tiers()]

                # Export with annotation provider for current annotations
                data_points.export_tsv(
                    file_path,
                    tier_names,
                    annotation_provider=self._get_annotation_intervals_at_time
                )
                self._status_bar.showMessage(
                    f"Exported {len(data_points.points)} point(s) to {Path(file_path).name}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export points: {e}")

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

    def _on_annotation_changed(self, *args):
        """Track annotation changes for global undo ordering."""
        if self._is_undoing:
            return  # Don't track changes during undo
        self._global_undo_stack.append('annotation')
        if len(self._global_undo_stack) > self._max_global_undo:
            self._global_undo_stack.pop(0)
        self._mark_dirty()

    def _on_data_point_changed(self, *args):
        """Track data point changes for global undo ordering."""
        if self._is_undoing:
            return  # Don't track changes during undo
        self._global_undo_stack.append('data_point')
        if len(self._global_undo_stack) > self._max_global_undo:
            self._global_undo_stack.pop(0)
        self._mark_dirty()

    def _update_window_title(self):
        """Update window title to show file name and dirty state."""
        title = "Ozen"
        if self._current_file_path:
            title = f"Ozen - {Path(self._current_file_path).name}"
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
            backup_path = os.path.join(tempfile.gettempdir(), "ozen_autosave.TextGrid")

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
        """Filter events to catch key presses globally."""
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            modifiers = event.modifiers()

            # Check for Ctrl+Z (Windows/Linux) or Cmd+Z (Mac) for undo
            is_undo = (key == Qt.Key.Key_Z and
                      (modifiers == Qt.KeyboardModifier.ControlModifier or
                       modifiers == Qt.KeyboardModifier.MetaModifier))

            if is_undo:
                self._global_undo()
                return True  # Event handled

            # Don't intercept if the text editor is active - let it handle input naturally
            if self._annotation_editor.is_editing_text():
                # Only handle Escape to close the editor
                if key == Qt.Key.Key_Escape:
                    self._annotation_editor._hide_text_editor()
                    self._annotation_editor.deselect_interval()
                    return True
                # Let the QLineEdit handle all other keys
                return False

            # Check if there's a selected interval but editor not shown yet
            selected = self._annotation_editor.get_selected_interval()
            if selected is not None:
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
