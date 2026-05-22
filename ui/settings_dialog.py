"""
设置对话框

包含：
  - 全局快捷键设置（自定义键盘组合）
  - 自动输出模式开关
  - 开机自启动开关
  - 识别语言选择
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QCheckBox, QLineEdit, QGroupBox,
    QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QFont

from ui.styles import (
    MAIN_STYLE, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY,
    COLOR_BG_ELEVATED, COLOR_BORDER, COLOR_ACCENT, COLOR_TEXT_MUTED,
)
from core.autostart import enable_autostart, disable_autostart, is_autostart_enabled


# ─── 快捷键输入框 ──────────────────────────────────────────

class HotkeyEdit(QLineEdit):
    """
    自定义快捷键输入框。
    
    点击后进入捕获模式，按下组合键即自动填充。
    格式：ctrl+alt+r, ctrl+shift+s 等
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlaceholderText("点击此处，然后按下快捷键…")
        self._capturing = False
        self.setStyleSheet(
            f"QLineEdit {{ background: {COLOR_BG_ELEVATED}; "
            f"color: {COLOR_TEXT_PRIMARY}; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 6px; padding: 8px 12px; font-size: 13px; }}"
            f"QLineEdit:focus {{ border-color: {COLOR_ACCENT}; }}"
        )

    def mousePressEvent(self, event):
        self._capturing = True
        self.setPlaceholderText("请按下快捷键组合…")
        self.setStyleSheet(
            self.styleSheet().replace(COLOR_BORDER, COLOR_ACCENT)
        )
        self.setFocus()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if not self._capturing:
            super().keyPressEvent(event)
            return

        key = event.key()
        modifiers = event.modifiers()

        # 忽略单独的修饰键
        if key in (
            Qt.Key.Key_Control, Qt.Key.Key_Shift,
            Qt.Key.Key_Alt, Qt.Key.Key_Meta,
        ):
            return

        # 按 Escape 取消
        if key == Qt.Key.Key_Escape:
            self._capturing = False
            self.setPlaceholderText("点击此处，然后按下快捷键…")
            return

        parts = []
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            parts.append("ctrl")
        if modifiers & Qt.KeyboardModifier.AltModifier:
            parts.append("alt")
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            parts.append("shift")

        key_text = QKeySequence(key).toString().lower()
        if key_text and key_text not in ("ctrl", "alt", "shift", "meta"):
            parts.append(key_text)

        if parts:
            self.setText("+".join(parts))

        self._capturing = False
        self.setPlaceholderText("点击此处，然后按下快捷键…")


# ─── 设置对话框 ───────────────────────────────────────────

class SettingsDialog(QDialog):
    """应用设置对话框。"""

    # 发出信号，通知主窗口配置变更
    hotkey_changed = pyqtSignal(str)
    auto_output_changed = pyqtSignal(bool)
    language_changed = pyqtSignal(object)  # str or None

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = dict(config)
        self.setWindowTitle("设置")
        self.setMinimumWidth(480)
        self.setStyleSheet(MAIN_STYLE)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # 标题
        title = QLabel("⚙️  应用设置")
        title.setStyleSheet(
            f"font-size: 17px; font-weight: 700; color: {COLOR_TEXT_PRIMARY};"
        )
        layout.addWidget(title)

        # ── 快捷键设置 ───────────────────────────────────
        hotkey_group = self._make_group("🎹  全局快捷键")
        hk_layout = QVBoxLayout()

        hk_desc = QLabel(
            "设置用于开启/关闭录音的全局快捷键。\n"
            "在任何窗口按下此键即可立即切换录音状态。"
        )
        hk_desc.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 12px;")
        hk_desc.setWordWrap(True)
        hk_layout.addWidget(hk_desc)

        self._hotkey_edit = HotkeyEdit()
        self._hotkey_edit.setText(self._config.get("hotkey", "ctrl+alt+r"))
        hk_layout.addWidget(self._hotkey_edit)

        reset_btn = QPushButton("恢复默认 (Ctrl+Alt+R)")
        reset_btn.setObjectName("actionButton")
        reset_btn.clicked.connect(lambda: self._hotkey_edit.setText("ctrl+alt+r"))
        hk_layout.addWidget(reset_btn)

        hotkey_group.layout().addLayout(hk_layout)
        layout.addWidget(hotkey_group)

        # ── 自动输出 ─────────────────────────────────────
        output_group = self._make_group("📤  自动输出")
        out_layout = QVBoxLayout()

        self._auto_output_cb = QCheckBox("开启自动输出到当前活动窗口")
        self._auto_output_cb.setStyleSheet(
            f"color: {COLOR_TEXT_PRIMARY}; font-size: 13px;"
        )
        self._auto_output_cb.setChecked(self._config.get("auto_output", False))
        out_layout.addWidget(self._auto_output_cb)

        out_desc = QLabel(
            "开启后，转录文字将自动粘贴到当前聚焦的输入框（如 Teams、Word 等）。\n"
            "关闭时，文字仅显示在应用内。"
        )
        out_desc.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        out_desc.setWordWrap(True)
        out_layout.addWidget(out_desc)

        output_group.layout().addLayout(out_layout)
        layout.addWidget(output_group)

        # ── 语言设置 ─────────────────────────────────────
        lang_group = self._make_group("🌍  识别语言")
        lang_layout = QHBoxLayout()

        lang_lbl = QLabel("语言：")
        lang_lbl.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY};")
        self._lang_combo = QComboBox()
        self._lang_combo.addItem("自动检测", None)
        self._lang_combo.addItem("中文（简体）", "zh-Hans")
        self._lang_combo.addItem("中文（繁体）", "zh-Hant")
        self._lang_combo.addItem("英文", "en")
        self._lang_combo.addItem("日文", "ja")
        self._lang_combo.addItem("韩文", "ko")
        self._lang_combo.addItem("法文", "fr")
        self._lang_combo.addItem("德文", "de")
        self._lang_combo.addItem("西班牙文", "es")

        # 设置当前语言
        current_lang = self._config.get("language", None)
        for i in range(self._lang_combo.count()):
            if self._lang_combo.itemData(i) == current_lang:
                self._lang_combo.setCurrentIndex(i)
                break

        lang_layout.addWidget(lang_lbl)
        lang_layout.addWidget(self._lang_combo, 1)
        lang_group.layout().addLayout(lang_layout)
        layout.addWidget(lang_group)

        # ── 开机启动 ─────────────────────────────────────
        system_group = self._make_group("🚀  系统设置")
        sys_layout = QVBoxLayout()

        self._autostart_cb = QCheckBox("开机后自动启动（在后台运行，不显示窗口）")
        self._autostart_cb.setStyleSheet(
            f"color: {COLOR_TEXT_PRIMARY}; font-size: 13px;"
        )
        self._autostart_cb.setChecked(is_autostart_enabled())

        autostart_desc = QLabel(
            "启用后，开机时自动在后台启动。\n"
            "通过快捷键或系统托盘图标使用应用。"
        )
        autostart_desc.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        autostart_desc.setWordWrap(True)

        sys_layout.addWidget(self._autostart_cb)
        sys_layout.addWidget(autostart_desc)
        system_group.layout().addLayout(sys_layout)
        layout.addWidget(system_group)

        # ── 按钮行 ───────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {COLOR_BORDER};")
        layout.addWidget(sep)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.setObjectName("actionButton")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton("保存设置")
        save_btn.setStyleSheet(
            f"QPushButton {{ background: {COLOR_ACCENT}; color: white; "
            f"border: none; border-radius: 8px; padding: 8px 20px; "
            f"font-size: 13px; font-weight: 600; }}"
            f"QPushButton:hover {{ background: #7a96ff; }}"
        )
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

    def _make_group(self, title: str) -> QGroupBox:
        group = QGroupBox(title)
        group.setStyleSheet(
            f"QGroupBox {{ color: {COLOR_TEXT_SECONDARY}; font-size: 12px; "
            f"font-weight: 600; border: 1px solid {COLOR_BORDER}; "
            f"border-radius: 8px; margin-top: 8px; padding: 8px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px; "
            f"padding: 0 4px; }}"
        )
        group.setLayout(QVBoxLayout())
        return group

    def _save(self):
        new_hotkey = self._hotkey_edit.text().strip()
        if not new_hotkey:
            QMessageBox.warning(self, "快捷键无效", "请设置一个有效的快捷键。")
            return

        # ⚠️ 必须在修改 self._config 之前保存旧值，否则比较永远相等
        old_hotkey = self._config.get("hotkey", "")

        # 更新配置
        self._config["hotkey"] = new_hotkey
        self._config["auto_output"] = self._auto_output_cb.isChecked()
        self._config["language"] = self._lang_combo.currentData()

        # 开机启动
        if self._autostart_cb.isChecked():
            enable_autostart()
        else:
            disable_autostart()
        self._config["autostart"] = self._autostart_cb.isChecked()

        # 发出信号：只要快捷键有值就始终注册（确保生效）
        self.hotkey_changed.emit(new_hotkey)
        self.auto_output_changed.emit(self._config["auto_output"])
        self.language_changed.emit(self._config["language"])

        self.accept()

    def get_config(self) -> dict:
        return self._config
