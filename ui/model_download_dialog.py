"""
模型下载管理对话框

显示所有可用的 Whisper 模型，支持：
  - 查看下载状态（已下载/未下载）
  - 一键下载，显示实时进度
  - 切换当前使用的模型
  - FFmpeg 安装状态检查
"""

import os
import shutil
import threading
from pathlib import Path

import requests
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QFrame, QScrollArea, QWidget, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

from ui.styles import (
    MAIN_STYLE, COLOR_ACCENT, COLOR_SUCCESS, COLOR_BG_SURFACE,
    COLOR_BG_ELEVATED, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY,
    COLOR_BORDER, COLOR_TEXT_MUTED, COLOR_DANGER,
)

# ─── Whisper 模型信息 ──────────────────────────────────────

MODELS = [
    {
        "name": "tiny",
        "display": "Tiny",
        "size_mb": 39,
        "speed": "⚡ 最快",
        "accuracy": "★★☆☆☆",
        "desc": "适合快速测试",
        "url": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
        "filename": "tiny.pt",
    },
    {
        "name": "base",
        "display": "Base",
        "size_mb": 74,
        "speed": "⚡ 快",
        "accuracy": "★★★☆☆",
        "desc": "速度与精度均衡",
        "url": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
        "filename": "base.pt",
    },
    {
        "name": "small",
        "display": "Small ⭐",
        "size_mb": 244,
        "speed": "🔄 中等",
        "accuracy": "★★★★☆",
        "desc": "推荐 — 中文识别效果好",
        "url": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
        "filename": "small.pt",
    },
    {
        "name": "medium",
        "display": "Medium",
        "size_mb": 769,
        "speed": "🐢 慢",
        "accuracy": "★★★★★",
        "desc": "高精度，需要较长时间",
        "url": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
        "filename": "medium.pt",
    },
    {
        "name": "large-v3",
        "display": "Large v3",
        "size_mb": 1550,
        "speed": "🐌 最慢",
        "accuracy": "★★★★★+",
        "desc": "最高精度，需要 GPU 或较长时间",
        "url": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
        "filename": "large-v3.pt",
    },
]

WHISPER_CACHE_DIR = Path.home() / ".cache" / "whisper"


def get_model_path(filename: str) -> Path:
    return WHISPER_CACHE_DIR / filename


def is_model_downloaded(filename: str) -> bool:
    p = get_model_path(filename)
    return p.exists() and p.stat().st_size > 1_000_000


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


# ─── 下载工作线程 ──────────────────────────────────────────

class DownloadThread(QThread):
    progress = pyqtSignal(int, int)   # (downloaded_bytes, total_bytes)
    finished = pyqtSignal(bool, str)  # (success, message)

    def __init__(self, url: str, dest: Path):
        super().__init__()
        self._url = url
        self._dest = dest
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        WHISPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = self._dest.with_suffix(".tmp")
        try:
            resp = requests.get(self._url, stream=True, timeout=30)
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0

            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if self._cancelled:
                        f.close()
                        tmp_path.unlink(missing_ok=True)
                        self.finished.emit(False, "已取消")
                        return
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        self.progress.emit(downloaded, total)

            # 使用 os.replace() 代替 Path.rename()：
            # Windows 上 rename 要求目标不存在（WinError 183），
            # os.replace() 会原子性覆盖已有文件，两个平台均兼容。
            os.replace(tmp_path, self._dest)
            self.finished.emit(True, "下载完成")
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            self.finished.emit(False, str(e))


# ─── 单个模型行组件 ───────────────────────────────────────

class ModelRow(QWidget):
    """显示一个模型的状态和操作按钮。"""

    use_model = pyqtSignal(str)  # model name

    def __init__(self, model_info: dict, current_model: str, parent=None):
        super().__init__(parent)
        self._info = model_info
        self._thread = None
        self._setup_ui(current_model)

    def _setup_ui(self, current_model: str):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)

        # 已下载标志
        is_downloaded = is_model_downloaded(self._info["filename"])
        is_current = self._info["name"] == current_model

        # 状态指示
        status_dot = QLabel("✅" if is_downloaded else "○")
        status_dot.setFixedWidth(24)
        status_dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status_dot)
        self._status_dot = status_dot

        # 模型名称
        name_lbl = QLabel(self._info["display"])
        name_lbl.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: "
            f"{'#5b7fff' if is_current else COLOR_TEXT_PRIMARY};"
        )
        name_lbl.setFixedWidth(100)
        layout.addWidget(name_lbl)

        # 大小 + 速度
        info_col = QVBoxLayout()
        info_col.setSpacing(2)
        size_lbl = QLabel(f"{self._info['size_mb']} MB  {self._info['speed']}")
        size_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 12px;")
        desc_lbl = QLabel(self._info["desc"])
        desc_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        info_col.addWidget(size_lbl)
        info_col.addWidget(desc_lbl)
        layout.addLayout(info_col)
        layout.addStretch()

        # 精度
        acc_lbl = QLabel(self._info["accuracy"])
        acc_lbl.setStyleSheet(f"color: {COLOR_ACCENT}; font-size: 12px;")
        layout.addWidget(acc_lbl)

        # 进度条（下载时显示）
        self._progress = QProgressBar()
        self._progress.setFixedWidth(160)
        self._progress.setMaximumHeight(16)
        self._progress.setVisible(False)
        self._progress.setFormat("%p%")
        layout.addWidget(self._progress)

        # 操作按钮
        self._btn = QPushButton()
        self._btn.setFixedWidth(90)
        self._btn.setObjectName("actionButton")
        self._update_button_state()
        self._btn.clicked.connect(self._on_btn_clicked)
        layout.addWidget(self._btn)

        self.setStyleSheet(
            f"ModelRow {{ background: {COLOR_BG_ELEVATED}; border: 1px solid "
            f"{'#5b7fff' if is_current else COLOR_BORDER}; border-radius: 8px; }}"
        )

    def _update_button_state(self):
        is_downloaded = is_model_downloaded(self._info["filename"])
        if is_downloaded:
            self._btn.setText("使用此模型")
            self._btn.setStyleSheet(
                f"QPushButton {{ background: rgba(91,127,255,0.15); "
                f"color: {COLOR_ACCENT}; border: 1px solid {COLOR_ACCENT}; "
                f"border-radius: 6px; padding: 6px; font-size: 12px; }}"
                f"QPushButton:hover {{ background: rgba(91,127,255,0.3); }}"
            )
        else:
            self._btn.setText("⬇ 下载")
            self._btn.setStyleSheet(
                f"QPushButton {{ background: {COLOR_BG_ELEVATED}; "
                f"color: {COLOR_TEXT_SECONDARY}; border: 1px solid {COLOR_BORDER}; "
                f"border-radius: 6px; padding: 6px; font-size: 12px; }}"
                f"QPushButton:hover {{ background: {COLOR_BG_ELEVATED}; color: white; }}"
            )

    def _on_btn_clicked(self):
        if is_model_downloaded(self._info["filename"]):
            self.use_model.emit(self._info["name"])
        else:
            self._start_download()

    def _start_download(self):
        dest = get_model_path(self._info["filename"])

        # 如果文件已完整存在，直接提示使用，无需重新下载
        if is_model_downloaded(self._info["filename"]):
            reply = QMessageBox.question(
                self.window(), "模型已存在",
                f"模型 {self._info['display']} 已下载。\n是否直接切换使用此模型？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.use_model.emit(self._info["name"])
            return

        # 清理上次下载失败留下的 .tmp 残留文件
        tmp_path = dest.with_suffix(".tmp")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

        self._btn.setText("取消")
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)

        self._thread = DownloadThread(self._info["url"], dest)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_finished)
        self._thread.start()

        # 按钮变为取消
        self._btn.clicked.disconnect()
        self._btn.clicked.connect(self._cancel_download)

    def _cancel_download(self):
        if self._thread:
            self._thread.cancel()

    def _on_progress(self, downloaded: int, total: int):
        if total > 0:
            pct = int(downloaded * 100 / total)
            self._progress.setValue(pct)
            mb = downloaded / 1_048_576
            self._progress.setFormat(f"{mb:.1f} MB / {total/1_048_576:.0f} MB")
        else:
            self._progress.setRange(0, 0)

    def _on_finished(self, success: bool, message: str):
        self._progress.setVisible(False)
        self._thread = None
        self._btn.clicked.disconnect()
        self._btn.clicked.connect(self._on_btn_clicked)

        if success:
            self._status_dot.setText("✅")
        else:
            QMessageBox.warning(self.window(), "下载失败", message)

        self._update_button_state()


# ─── 主对话框 ─────────────────────────────────────────────

class ModelDownloadDialog(QDialog):
    """Whisper 模型下载管理对话框。"""

    model_changed = pyqtSignal(str)  # 当用户选择新模型时

    def __init__(self, current_model: str = "small", parent=None):
        super().__init__(parent)
        self._current_model = current_model
        self.setWindowTitle("模型管理")
        self.setMinimumSize(680, 520)
        self.setStyleSheet(MAIN_STYLE)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # 标题
        title = QLabel("🤖  Whisper 模型管理")
        title.setStyleSheet(
            f"font-size: 18px; font-weight: 700; color: {COLOR_TEXT_PRIMARY};"
        )
        layout.addWidget(title)

        subtitle = QLabel(
            "模型文件存储在本地，下载后可完全离线使用。"
            "不同模型在速度和精度之间有所取舍。"
        )
        subtitle.setStyleSheet(
            f"color: {COLOR_TEXT_SECONDARY}; font-size: 13px;"
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # FFmpeg 状态
        ffmpeg_ok = check_ffmpeg()
        ffmpeg_row = QHBoxLayout()
        ffmpeg_icon = QLabel("✅" if ffmpeg_ok else "⚠️")
        ffmpeg_text = QLabel(
            "FFmpeg 已安装 — 支持所有音频格式" if ffmpeg_ok
            else "未检测到 FFmpeg — 请安装后重启应用（winget install ffmpeg）"
        )
        ffmpeg_text.setStyleSheet(
            f"color: {'#22c55e' if ffmpeg_ok else '#f59e0b'}; font-size: 12px;"
        )
        ffmpeg_row.addWidget(ffmpeg_icon)
        ffmpeg_row.addWidget(ffmpeg_text)
        ffmpeg_row.addStretch()
        layout.addLayout(ffmpeg_row)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {COLOR_BORDER};")
        layout.addWidget(sep)

        # 模型列表（可滚动）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        model_layout = QVBoxLayout(container)
        model_layout.setSpacing(8)
        model_layout.setContentsMargins(0, 0, 0, 0)

        for info in MODELS:
            row = ModelRow(info, self._current_model, container)
            row.use_model.connect(self._on_use_model)
            model_layout.addWidget(row)

        model_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

        # 底部关闭按钮
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("关闭")
        close_btn.setObjectName("actionButton")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _on_use_model(self, model_name: str):
        self._current_model = model_name
        self.model_changed.emit(model_name)
        QMessageBox.information(
            self, "模型已切换",
            f"已切换到 {model_name} 模型。\n重启录音后生效。"
        )
