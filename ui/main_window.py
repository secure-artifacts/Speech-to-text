"""
主窗口 v2.0

新增功能：
  - 全局快捷键切换录音（任意位置）
  - 自动输出转录文字到当前活动窗口
  - 中文标点自动后处理
  - 系统托盘（关闭后后台运行）
  - 托盘菜单（显示/切换录音/退出）
  - 快捷键设置显示
  - 自动输出切换按钮
  - 模型管理 & 设置按钮
"""

import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QListWidget, QListWidgetItem,
    QSplitter, QFrame, QComboBox, QProgressBar, QFileDialog,
    QMenuBar, QMenu, QStatusBar, QMessageBox, QSystemTrayIcon,
    QApplication,
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize,
)
from PyQt6.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush, QAction, QIcon,
    QTextCursor,
)

from core.recorder import AudioRecorder, SAMPLE_RATE
from core.transcriber import RealtimeTranscriber
from core.storage import TranscriptStorage
from core.config import load_config, save_config
from core.hotkey import HotkeyManager
from core.output_engine import OutputEngine
from core.punctuation import add_punctuation
from ui.floating_indicator import FloatingIndicator
from ui.styles import (
    MAIN_STYLE,
    STATUS_DOT_IDLE, STATUS_DOT_RECORDING,
    STATUS_DOT_PROCESSING, STATUS_DOT_ERROR,
    COLOR_ACCENT, COLOR_SUCCESS, COLOR_DANGER, COLOR_BG_SURFACE,
    COLOR_TEXT_MUTED, COLOR_BORDER, COLOR_TEXT_SECONDARY,
    COLOR_BG_ELEVATED, COLOR_TEXT_PRIMARY,
)


def _find_icon() -> QIcon:
    """查找应用图标（image.ico）。"""
    candidates = [
        Path(__file__).parent.parent / "image.ico",
        Path(sys.executable).parent / "image.ico",
        Path("image.ico"),
    ]
    for p in candidates:
        if p.exists():
            return QIcon(str(p))
    return QIcon()


# ─────────────────────────────────────────────────────────
# 波形可视化组件（与 v1 相同）
# ─────────────────────────────────────────────────────────

class WaveformWidget(QWidget):
    HISTORY_LEN = 80

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("waveformWidget")
        self.setMinimumHeight(60)
        self.setMaximumHeight(60)
        self._levels = [0.0] * self.HISTORY_LEN
        self._is_recording = False

    def update_level(self, level: float):
        self._levels.append(min(1.0, max(0.0, level)))
        if len(self._levels) > self.HISTORY_LEN:
            self._levels.pop(0)
        self.update()

    def set_recording(self, recording: bool):
        self._is_recording = recording
        if not recording:
            self._levels = [0.0] * self.HISTORY_LEN
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        center_y = h // 2
        painter.fillRect(0, 0, w, h, QColor(COLOR_BG_SURFACE))
        if not self._levels:
            return
        bar_w = max(2, w // self.HISTORY_LEN - 1)
        spacing = w // self.HISTORY_LEN
        accent = QColor(COLOR_ACCENT if not self._is_recording else COLOR_SUCCESS)
        for i, level in enumerate(self._levels):
            x = i * spacing
            bar_h = max(3, int(level * (h - 12)))
            alpha = int(80 + 175 * (i / self.HISTORY_LEN))
            color = QColor(accent)
            color.setAlpha(alpha)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(x, center_y - bar_h // 2, bar_w, bar_h, 1, 1)


# ─────────────────────────────────────────────────────────
# 模型加载线程
# ─────────────────────────────────────────────────────────

class ModelLoaderThread(QThread):
    loaded = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, transcriber: RealtimeTranscriber):
        super().__init__()
        self._transcriber = transcriber

    def run(self):
        try:
            self._transcriber._load_model()
            self.loaded.emit()
        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────────────────
# 主窗口
# ─────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    sig_transcript_received = pyqtSignal(str)
    sig_status_update = pyqtSignal(str, str)
    sig_level_update = pyqtSignal(float)
    sig_toggle_recording = pyqtSignal()   # 快捷键触发（跨线程安全）

    def __init__(self, start_hidden: bool = False):
        super().__init__()
        self._config = load_config()
        self._is_recording = False
        self._model_loaded = False
        self._record_start_time = None
        self._level_history = []
        self._level_lock = threading.Lock()
        self._quitting = False   # 真正退出（区别于最小化到托盘）

        # ── 核心模块 ──────────────────────────────────────
        self._recorder = AudioRecorder(
            on_audio_chunk=self._on_audio_chunk,
            on_error=self._on_recorder_error,
        )
        self._transcriber = RealtimeTranscriber(
            model_name=self._config.get("model", "small"),
            language=self._config.get("language"),
            on_final=self._on_transcript_final,
            on_error=self._on_transcriber_error,
        )
        self._storage = TranscriptStorage()
        self._hotkey_mgr = HotkeyManager()
        self._output_engine = OutputEngine()
        self._output_engine.set_auto_output(self._config.get("auto_output", False))

        # 屏幕底部悬浮指示器
        self._floating = FloatingIndicator(
            toggle_callback=self._toggle_recording
        )

        self._setup_ui()
        self._setup_tray()
        self._connect_signals()
        self._setup_menu()
        self._load_model()

        # 注册快捷键
        self._register_hotkey(self._config.get("hotkey", "ctrl+alt+r"))

        # 将本窗口句柄传给输出引擎（延迟，等窗口创建完成）
        QTimer.singleShot(500, self._init_output_engine)

        if start_hidden:
            self.hide()
        else:
            self.show()

    def _init_output_engine(self):
        """初始化输出引擎的窗口句柄。"""
        try:
            hwnd = int(self.winId())
            self._output_engine.set_our_window(hwnd)
        except Exception:
            pass

    # ─────────────────────────────────────────
    # UI 构建
    # ─────────────────────────────────────────

    def _setup_ui(self):
        self.setWindowTitle("Whisper 语音转文字")
        self.setWindowIcon(_find_icon())
        self.setMinimumSize(940, 660)
        self.resize(1120, 730)
        self.setStyleSheet(MAIN_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_title_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {COLOR_BORDER}; }}")
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([700, 320])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        root_layout.addWidget(splitter, 1)

        self._build_status_bar()

    def _build_title_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("titleBar")
        bar.setFixedHeight(64)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(24, 0, 24, 0)

        icon_label = QLabel("🎙")
        icon_label.setStyleSheet("font-size: 26px;")
        title = QLabel("语音转文字")
        title.setObjectName("appTitle")
        subtitle = QLabel("· Powered by OpenAI Whisper（本地运行）")
        subtitle.setObjectName("appSubtitle")

        layout.addWidget(icon_label)
        layout.addSpacing(8)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()

        # 快捷键显示
        self._hotkey_label = QLabel()
        self._hotkey_label.setStyleSheet(
            f"color: {COLOR_ACCENT}; font-size: 11px; font-weight: 600; "
            f"background: rgba(91,127,255,0.12); border: 1px solid rgba(91,127,255,0.3); "
            f"border-radius: 4px; padding: 3px 8px;"
        )
        self._update_hotkey_label()
        layout.addWidget(self._hotkey_label)
        layout.addSpacing(16)

        # 状态指示
        self._status_dot = QLabel()
        self._status_dot.setObjectName("statusDot")
        self._status_dot.setFixedSize(10, 10)
        self._status_dot.setStyleSheet(STATUS_DOT_IDLE)
        self._status_label = QLabel("正在加载模型…")
        self._status_label.setObjectName("statusLabel")
        layout.addWidget(self._status_dot)
        layout.addSpacing(6)
        layout.addWidget(self._status_label)

        return bar

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 18, 12, 18)
        layout.setSpacing(10)

        # ── 控制行 ────────────────────────────────────────
        control_row = QHBoxLayout()
        control_row.setSpacing(16)

        # 录音按钮
        self._record_btn = QPushButton("⏺")
        self._record_btn.setObjectName("recordButton")
        self._record_btn.setProperty("recording", "false")
        self._record_btn.setToolTip("点击或按快捷键开始录音")
        self._record_btn.setEnabled(False)
        self._record_btn.clicked.connect(self._toggle_recording)
        control_row.addWidget(self._record_btn)

        # 右侧信息列
        info_col = QVBoxLayout()
        info_col.setSpacing(6)
        self._record_status_label = QLabel("等待模型加载…")
        self._record_status_label.setStyleSheet(
            f"font-size: 13px; color: {COLOR_TEXT_SECONDARY};"
        )
        self._timer_label = QLabel("00:00")
        self._timer_label.setStyleSheet(
            f"font-size: 30px; font-weight: 700; color: {COLOR_TEXT_PRIMARY}; "
            f"font-family: 'Courier New', monospace;"
        )
        self._mic_label = QLabel(f"🎤 {AudioRecorder.get_default_input_device()}")
        self._mic_label.setObjectName("infoLabel")
        info_col.addWidget(self._record_status_label)
        info_col.addWidget(self._timer_label)
        info_col.addWidget(self._mic_label)
        info_col.addStretch()
        control_row.addLayout(info_col)
        control_row.addStretch()

        # 功能按钮组（右侧）
        btn_col = QVBoxLayout()
        btn_col.setSpacing(8)
        btn_col.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 自动输出切换按钮
        self._auto_output_btn = QPushButton("📤  自动输出：关")
        self._auto_output_btn.setObjectName("actionButton")
        self._auto_output_btn.setToolTip(
            "开启后，识别文字自动粘贴到当前活动窗口（Teams、Word 等）"
        )
        self._auto_output_btn.setCheckable(True)
        self._auto_output_btn.setChecked(self._config.get("auto_output", False))
        self._auto_output_btn.clicked.connect(self._toggle_auto_output)
        self._update_auto_output_btn()
        btn_col.addWidget(self._auto_output_btn)

        self._clear_btn = QPushButton("🗑  清空记录")
        self._clear_btn.setObjectName("actionButton")
        self._clear_btn.clicked.connect(self._clear_transcript)
        btn_col.addWidget(self._clear_btn)

        self._export_btn = QPushButton("💾  导出文本")
        self._export_btn.setObjectName("actionButton")
        self._export_btn.clicked.connect(self._export_transcript)
        btn_col.addWidget(self._export_btn)

        self._model_btn = QPushButton("🤖  模型管理")
        self._model_btn.setObjectName("actionButton")
        self._model_btn.clicked.connect(self._show_model_dialog)
        btn_col.addWidget(self._model_btn)

        self._settings_btn = QPushButton("⚙️  设置")
        self._settings_btn.setObjectName("actionButton")
        self._settings_btn.clicked.connect(self._show_settings)
        btn_col.addWidget(self._settings_btn)

        control_row.addLayout(btn_col)
        layout.addLayout(control_row)

        # 波形
        self._waveform = WaveformWidget()
        layout.addWidget(self._waveform)

        # 加载进度条
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setFormat("正在加载 Whisper 模型…")
        layout.addWidget(self._progress_bar)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        # 转录区标题
        text_header = QHBoxLayout()
        text_title = QLabel("转录内容")
        text_title.setObjectName("panelTitle")
        self._word_count_label = QLabel("0 条记录")
        self._word_count_label.setObjectName("countLabel")
        text_header.addWidget(text_title)
        text_header.addStretch()
        text_header.addWidget(self._word_count_label)
        layout.addLayout(text_header)

        # 转录文字区
        self._transcript_area = QTextEdit()
        self._transcript_area.setObjectName("transcriptArea")
        self._transcript_area.setReadOnly(False)
        self._transcript_area.setPlaceholderText(
            "点击录音按钮或按快捷键开始说话，文字将实时显示在这里…\n\n"
            "✅ 开启「自动输出」后，识别内容将自动输入到任意文本框。\n"
            "🔒 所有录音和文字仅存储在本地，不上传任何数据。"
        )
        layout.addWidget(self._transcript_area, 1)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("historyPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("历史记录")
        title.setObjectName("panelTitle")
        self._history_count = QLabel("今日 0 条")
        self._history_count.setObjectName("countLabel")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self._history_count)
        layout.addLayout(header)

        date_row = QHBoxLayout()
        date_lbl = QLabel("日期：")
        date_lbl.setObjectName("infoLabel")
        self._date_combo = QComboBox()
        self._date_combo.addItem(f"今天 ({datetime.now().strftime('%Y-%m-%d')})")
        self._date_combo.currentIndexChanged.connect(self._on_date_changed)
        date_row.addWidget(date_lbl)
        date_row.addWidget(self._date_combo, 1)
        layout.addLayout(date_row)

        self._history_list = QListWidget()
        self._history_list.setObjectName("historyList")
        self._history_list.itemClicked.connect(self._on_history_item_clicked)
        layout.addWidget(self._history_list, 1)

        privacy_label = QLabel("🔒 所有数据仅存储在本地")
        privacy_label.setObjectName("infoLabel")
        privacy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        privacy_label.setStyleSheet("color: #22c55e; font-size: 11px; padding: 6px;")
        layout.addWidget(privacy_label)

        return panel

    def _build_status_bar(self):
        sb = self.statusBar()
        sb.setStyleSheet(
            f"QStatusBar {{ background: #0d0f14; color: #4a5268; "
            f"border-top: 1px solid #252a38; font-size: 12px; padding: 0 12px; }}"
        )
        sb.showMessage("就绪  ·  数据路径：" + str(self._storage.data_dir))

    # ─────────────────────────────────────────
    # 系统托盘
    # ─────────────────────────────────────────

    def _setup_tray(self):
        """初始化系统托盘图标和菜单。"""
        icon = _find_icon()
        if icon.isNull():
            # 如果没有图标文件，用文字代替
            from PyQt6.QtGui import QPixmap, QPainter, QColor
            pm = QPixmap(32, 32)
            pm.fill(QColor(COLOR_ACCENT))
            icon = QIcon(pm)

        self._tray = QSystemTrayIcon(icon, self)
        self._tray.setToolTip("Whisper 语音转文字")

        tray_menu = QMenu()
        tray_menu.setStyleSheet(MAIN_STYLE)

        show_action = QAction("📋 显示主窗口", self)
        show_action.triggered.connect(self._show_main_window)
        tray_menu.addAction(show_action)

        tray_menu.addSeparator()

        self._tray_record_action = QAction("⏺ 开始录音", self)
        self._tray_record_action.triggered.connect(self._toggle_recording)
        tray_menu.addAction(self._tray_record_action)

        tray_menu.addSeparator()

        settings_action = QAction("⚙️ 设置", self)
        settings_action.triggered.connect(self._show_settings)
        tray_menu.addAction(settings_action)

        tray_menu.addSeparator()

        quit_action = QAction("✕ 退出", self)
        quit_action.triggered.connect(self._quit_app)
        tray_menu.addAction(quit_action)

        self._tray.setContextMenu(tray_menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._show_main_window()

    def _show_main_window(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _quit_app(self):
        """真正退出应用。"""
        self._quitting = True
        if self._is_recording:
            self._stop_recording()
        self._transcriber.stop_worker()
        self._hotkey_mgr.unregister()
        self._tray.hide()
        QApplication.quit()

    # ─────────────────────────────────────────
    # 快捷键
    # ─────────────────────────────────────────

    def _register_hotkey(self, hotkey_str: str):
        """注册全局快捷键。"""
        ok = self._hotkey_mgr.register(
            hotkey_str,
            lambda: self.sig_toggle_recording.emit()
        )
        if ok:
            self._config["hotkey"] = hotkey_str
            self._update_hotkey_label()
        else:
            self.statusBar().showMessage(
                f"⚠️ 快捷键 '{hotkey_str}' 注册失败，可能已被其他应用占用", 5000
            )

    def _update_hotkey_label(self):
        hk = self._config.get("hotkey", "ctrl+alt+r").upper().replace("+", " + ")
        self._hotkey_label.setText(f"快捷键：{hk}")

    # ─────────────────────────────────────────
    # 模型加载
    # ─────────────────────────────────────────

    def _load_model(self):
        self._loader_thread = ModelLoaderThread(self._transcriber)
        self._loader_thread.loaded.connect(self._on_model_loaded)
        self._loader_thread.error.connect(self._on_model_error)
        self._loader_thread.start()

    def _on_model_loaded(self):
        self._model_loaded = True
        self._record_btn.setEnabled(True)
        self._progress_bar.setVisible(False)
        self._record_status_label.setText("就绪，点击或按快捷键开始录音")
        device_info = self._transcriber.device_display
        self._update_status(f"模型加载完成 ✓  ·  {device_info}", "idle")
        self.statusBar().showMessage(
            f"就绪  ·  {device_info}  ·  数据路径：{self._storage.data_dir}"
        )
        # 把设备信息显示在悬浮指示器上
        self._floating.set_device_label(device_info)
        self._transcriber.start_worker()
        self._load_history()

    def _on_model_error(self, err_msg: str):
        self._progress_bar.setVisible(False)
        self._record_status_label.setText("模型加载失败！")
        self._update_status(f"错误：{err_msg}", "error")
        QMessageBox.critical(
            self, "模型加载失败",
            f"无法加载模型：\n{err_msg}\n\n"
            "请点击「模型管理」按钮下载模型，或检查网络连接。"
        )

    # ─────────────────────────────────────────
    # 录音控制
    # ─────────────────────────────────────────

    def _toggle_recording(self):
        if not self._model_loaded:
            return
        if self._is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        self._is_recording = True
        self._record_start_time = time.time()

        self._record_btn.setProperty("recording", "true")
        self._record_btn.style().unpolish(self._record_btn)
        self._record_btn.style().polish(self._record_btn)
        self._record_btn.setText("⏹")
        self._record_btn.setToolTip("点击或按快捷键停止录音")
        self._record_status_label.setText("🔴 正在录音…")
        self._waveform.set_recording(True)
        self._update_status("录音中…", "recording")
        self._tray_record_action.setText("⏹ 停止录音")

        # 托盘提示（首次）
        if self._config.get("show_tray_tip", True):
            self._tray.showMessage(
                "正在录音",
                f"快捷键 {self._config.get('hotkey','').upper()} 可停止录音",
                QSystemTrayIcon.MessageIcon.Information, 2000,
            )

        self._elapsed_timer.start()
        self._level_timer.start()
        self._transcriber.reset_buffer()
        self._recorder.start()
        # 显示悬浮指示器
        self._floating.start_recording()

    def _stop_recording(self):
        self._is_recording = False
        self._recorder.stop()
        self._elapsed_timer.stop()
        self._level_timer.stop()
        self._transcriber.flush()

        self._record_btn.setProperty("recording", "false")
        self._record_btn.style().unpolish(self._record_btn)
        self._record_btn.style().polish(self._record_btn)
        self._record_btn.setText("⏺")
        self._record_btn.setToolTip("点击或按快捷键开始录音")
        self._record_status_label.setText("就绪，点击或按快捷键开始录音")
        self._timer_label.setText("00:00")
        self._waveform.set_recording(False)
        self._update_status("识别中…", "processing")
        self._tray_record_action.setText("⏺ 开始录音")
        # 隐藏悬浮指示器
        self._floating.stop_recording()

    # ─────────────────────────────────────────
    # 音频回调
    # ─────────────────────────────────────────

    def _on_audio_chunk(self, chunk: np.ndarray):
        self._transcriber.feed_audio(chunk)
        energy = float(np.sqrt(np.mean(chunk ** 2)))
        level = min(1.0, energy * 8)
        with self._level_lock:
            self._level_history.append(level)

    def _flush_level(self):
        with self._level_lock:
            if self._level_history:
                avg = sum(self._level_history) / len(self._level_history)
                self._level_history.clear()
                self.sig_level_update.emit(avg)

    def _on_recorder_error(self, err: Exception):
        self.sig_status_update.emit(f"录音错误：{err}", "error")

    # ─────────────────────────────────────────
    # 转录回调（转录线程 → 主线程）
    # ─────────────────────────────────────────

    def _on_transcript_final(self, raw_text: str):
        """Whisper 返回结果 → 标点处理 → 存储 → UI + 输出。"""
        if not raw_text.strip():
            return

        # 标点后处理
        processed = add_punctuation(raw_text)

        # 本地存储
        self._storage.save_entry(processed)

        # 通过信号更新 UI（线程安全）
        self.sig_transcript_received.emit(processed)

    def _on_transcriber_error(self, err: Exception):
        self.sig_status_update.emit(f"识别错误：{err}", "error")

    # ─────────────────────────────────────────
    # UI 更新槽（主线程）
    # ─────────────────────────────────────────

    def _append_transcript(self, text: str):
        """将识别文字追加到文字区域（不含时间戳）。"""
        # 文字区域追加（不含时间戳，纯文字）
        self._transcript_area.insertPlainText(text + "\n")
        self._transcript_area.ensureCursorVisible()

        # 自动输出到活动窗口
        self._output_engine.output(text)

        # 更新状态
        self._update_status("识别完成", "idle")
        count = self._storage.get_total_count()
        self._word_count_label.setText(f"{count} 条记录")
        self._load_history()

    def _update_status(self, message: str, state: str = "idle"):
        dot_styles = {
            "idle": STATUS_DOT_IDLE,
            "recording": STATUS_DOT_RECORDING,
            "processing": STATUS_DOT_PROCESSING,
            "error": STATUS_DOT_ERROR,
        }
        self._status_dot.setStyleSheet(dot_styles.get(state, STATUS_DOT_IDLE))
        self._status_label.setText(message)

    def _update_timer(self):
        if self._record_start_time:
            elapsed = int(time.time() - self._record_start_time)
            m, s = divmod(elapsed, 60)
            self._timer_label.setText(f"{m:02d}:{s:02d}")

    # ─────────────────────────────────────────
    # 自动输出按钮
    # ─────────────────────────────────────────

    def _toggle_auto_output(self):
        enabled = self._auto_output_btn.isChecked()
        self._output_engine.set_auto_output(enabled)
        self._config["auto_output"] = enabled
        save_config(self._config)
        self._update_auto_output_btn()

    def _update_auto_output_btn(self):
        enabled = self._config.get("auto_output", False)
        if enabled:
            self._auto_output_btn.setText("📤  自动输出：开")
            self._auto_output_btn.setStyleSheet(
                f"QPushButton {{ background: rgba(34,197,94,0.15); "
                f"color: #22c55e; border: 1px solid #22c55e; "
                f"border-radius: 8px; padding: 8px 14px; font-size: 13px; }}"
                f"QPushButton:hover {{ background: rgba(34,197,94,0.25); }}"
            )
        else:
            self._auto_output_btn.setText("📤  自动输出：关")
            self._auto_output_btn.setStyleSheet("")  # 使用默认样式

    # ─────────────────────────────────────────
    # 历史记录
    # ─────────────────────────────────────────

    def _load_history(self):
        records = self._storage.get_today_records()
        self._history_list.clear()
        for rec in reversed(records[-50:]):
            ts = datetime.fromisoformat(rec["timestamp"])
            text = rec["text"]
            display = f"{ts.strftime('%H:%M:%S')}\n{text[:60]}{'…' if len(text) > 60 else ''}"
            item = QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, text)
            self._history_list.addItem(item)
        self._history_count.setText(f"今日 {len(records)} 条")

        all_dates = self._storage.get_all_dates()
        current = [self._date_combo.itemText(i) for i in range(self._date_combo.count())]
        for d in all_dates:
            if d not in current:
                self._date_combo.addItem(d)

    def _on_date_changed(self, index: int):
        if index == 0:
            self._load_history()
            return
        date_str = self._date_combo.currentText()
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            records = self._storage.get_records_for_date(date)
            self._history_list.clear()
            for rec in reversed(records[-50:]):
                ts = datetime.fromisoformat(rec["timestamp"])
                display = f"{ts.strftime('%H:%M:%S')}\n{rec['text'][:60]}"
                item = QListWidgetItem(display)
                item.setData(Qt.ItemDataRole.UserRole, rec["text"])
                self._history_list.addItem(item)
        except ValueError:
            pass

    def _on_history_item_clicked(self, item: QListWidgetItem):
        text = item.data(Qt.ItemDataRole.UserRole)
        if text:
            QApplication.clipboard().setText(text)
            self.statusBar().showMessage(f"已复制：{text[:40]}…", 3000)

    # ─────────────────────────────────────────
    # 对话框
    # ─────────────────────────────────────────

    def _show_settings(self):
        from ui.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self._config, self)
        dlg.hotkey_changed.connect(self._register_hotkey)
        dlg.auto_output_changed.connect(self._on_auto_output_changed)
        dlg.language_changed.connect(self._on_language_changed)
        if dlg.exec():
            self._config = dlg.get_config()
            save_config(self._config)
            self._update_auto_output_btn()
            self._auto_output_btn.setChecked(self._config.get("auto_output", False))

    def _on_auto_output_changed(self, enabled: bool):
        self._output_engine.set_auto_output(enabled)
        self._config["auto_output"] = enabled
        self._update_auto_output_btn()

    def _on_language_changed(self, lang):
        self._transcriber.language = lang
        self._config["language"] = lang

    def _show_model_dialog(self):
        from ui.model_download_dialog import ModelDownloadDialog
        dlg = ModelDownloadDialog(self._config.get("model", "small"), self)
        dlg.model_changed.connect(self._on_model_changed)
        dlg.exec()

    def _on_model_changed(self, model_name: str):
        self._config["model"] = model_name
        save_config(self._config)
        self._transcriber.model_name = model_name

    def _clear_transcript(self):
        reply = QMessageBox.question(
            self, "确认清空",
            "确定清空当前文字区域？\n（历史文件不会被删除）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._transcript_area.clear()
            self._word_count_label.setText("0 条记录")

    def _export_transcript(self):
        default = str(
            Path.home() / "Desktop" /
            f"转录记录_{datetime.now().strftime('%Y%m%d')}.txt"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "导出转录记录", default,
            "文本文件 (*.txt);;所有文件 (*)"
        )
        if path:
            try:
                import shutil
                txt_path = self._storage.export_txt()
                shutil.copy(txt_path, path)
                self.statusBar().showMessage(f"已导出：{path}", 5000)
            except Exception as e:
                QMessageBox.warning(self, "导出失败", str(e))

    def _open_data_dir(self):
        import subprocess
        subprocess.Popen(f'explorer "{self._storage.data_dir}"')

    # ─────────────────────────────────────────
    # 菜单
    # ─────────────────────────────────────────

    def _setup_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件")
        file_menu.addAction("导出今日记录 (Ctrl+S)", self._export_transcript).setShortcut("Ctrl+S")
        file_menu.addSeparator()
        file_menu.addAction("打开数据文件夹", self._open_data_dir)
        file_menu.addSeparator()
        file_menu.addAction("退出 (Ctrl+Q)", self._quit_app).setShortcut("Ctrl+Q")

        settings_menu = menubar.addMenu("设置")
        settings_menu.addAction("打开设置…", self._show_settings)
        settings_menu.addSeparator()

        lang_menu = settings_menu.addMenu("识别语言")
        for name, code in [
            ("自动检测",      None),
            ("中文（简体）",  "zh-Hans"),
            ("中文（繁体）",  "zh-Hant"),
            ("英文",          "en"),
            ("日文",          "ja"),
            ("韩文",          "ko"),
        ]:
            lang_menu.addAction(name, lambda c=code: self._on_language_changed(c))

        about_menu = menubar.addMenu("关于")
        about_menu.addAction("关于", self._show_about)

    def _show_about(self):
        QMessageBox.about(
            self, "关于 Whisper 语音转文字",
            "<h3>Whisper 语音转文字 v2.0</h3>"
            "<p>基于 OpenAI Whisper 的本地实时语音识别应用</p>"
            "<p><b>功能：</b><br>"
            "• 全局快捷键切换录音<br>"
            "• 自动输出到任意窗口（Teams、Word 等）<br>"
            "• 中文标点自动添加<br>"
            "• 完全本地运行，录音不上传</p>"
        )

    # ─────────────────────────────────────────
    # 信号连接 & 计时器
    # ─────────────────────────────────────────

    def _connect_signals(self):
        self.sig_transcript_received.connect(self._append_transcript)
        self.sig_status_update.connect(self._update_status)
        self.sig_level_update.connect(self._waveform.update_level)
        self.sig_level_update.connect(self._floating.update_level)  # 同步给悬浮指示器
        self.sig_toggle_recording.connect(self._toggle_recording)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._update_timer)

        self._level_timer = QTimer(self)
        self._level_timer.setInterval(50)
        self._level_timer.timeout.connect(self._flush_level)

    # ─────────────────────────────────────────
    # 关闭行为（最小化到托盘）
    # ─────────────────────────────────────────

    def closeEvent(self, event):
        if self._quitting:
            if self._is_recording:
                self._stop_recording()
            self._transcriber.stop_worker()
            self._hotkey_mgr.unregister()
            event.accept()
        else:
            # 最小化到托盘
            event.ignore()
            self.hide()
            if self._config.get("show_tray_tip", True):
                self._tray.showMessage(
                    "Whisper 语音转文字",
                    f"已最小化到托盘。快捷键 "
                    f"{self._config.get('hotkey','').upper()} 可随时开始录音。",
                    QSystemTrayIcon.MessageIcon.Information, 3000,
                )
                self._config["show_tray_tip"] = False
                save_config(self._config)
