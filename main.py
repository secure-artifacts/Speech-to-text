"""
Whisper 语音转文字 - 程序入口 v2.0

支持 --hidden 参数静默启动（开机自启时使用）。

macOS 注意：
  PyInstaller 在 macOS 上使用 multiprocessing 时，Python 会用 spawn
  方式创建子进程，导致每个子进程都重新执行本文件。
  必须在最顶部调用 multiprocessing.freeze_support()，
  并将所有业务代码放在 if __name__ == "__main__" 守卫内。
"""

import sys
import os
import multiprocessing

# ❶ 必须在最顶部、任何其他导入之前调用
#    这是 PyInstaller macOS 多进程的必要修复（防止 Dock 重复出现 App 图标）
multiprocessing.freeze_support()

# ❷ macOS 专用：阻止 torch / numpy 内部启动子进程（每个子进程 = 新 Dock 图标）
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

os.environ.setdefault("PYTHONWARNINGS", "ignore")


def main():
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont, QIcon
    from PyQt6.QtCore import Qt

    # 高 DPI 支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("WhisperSTT")
    app.setApplicationDisplayName("语音转文字")
    app.setOrganizationName("Local")

    # 关闭最后一个窗口不退出（托盘运行）
    app.setQuitOnLastWindowClosed(False)

    # 设置全局字体（跨平台）
    if sys.platform == "darwin":
        # macOS 使用系统字体
        font = QFont(".AppleSystemUIFont", 13)
    else:
        font = QFont("Microsoft YaHei UI", 10)
        font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    app.setFont(font)

    # 设置应用图标
    from pathlib import Path
    for icon_name in ("image.icns", "image.ico"):
        icon_path = Path(__file__).parent / icon_name
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
            break

    # --hidden 参数：开机自启时不显示窗口
    start_hidden = "--hidden" in sys.argv

    from ui.main_window import MainWindow
    window = MainWindow(start_hidden=start_hidden)

    sys.exit(app.exec())


# ❷ 所有业务代码必须在此守卫内
#    子进程会 import 本模块，但不会执行 main()，避免重复启动 UI
if __name__ == "__main__":
    main()
