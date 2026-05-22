"""
Whisper 转录模块 - 实时语音识别

使用 OpenAI Whisper 在本地进行语音识别，
支持中文、英文及多语言自动检测。

实时转录原理：
  - 维护一个滑动音频缓冲区（约 30 秒）
  - 每隔 VAD_INTERVAL 秒检测一次语音活动
  - 检测到停顿时，对累积的音频段进行识别
  - 将结果通过回调返回给 UI 线程
"""

import sys
import os
import threading
import time
import numpy as np

# ── PyInstaller 打包路径修复 ────────────────────────────────
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _mei = sys._MEIPASS
    if _mei not in sys.path:
        sys.path.insert(0, _mei)
    _whisper_assets = os.path.join(_mei, "whisper", "assets")
    if os.path.isdir(_whisper_assets):
        os.environ["WHISPER_ASSETS_DIR"] = _whisper_assets

import torch
import whisper


# ── 设备自动选择 ─────────────────────────────────────────────

def _select_device() -> str:
    """
    自动选择最优计算设备：
      1. NVIDIA GPU (CUDA)
      2. Apple Silicon (MPS)
      3. CPU（使用全部核心）
    """
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    # CPU 模式：使用全部核心，提升并行度
    cores = os.cpu_count() or 4
    torch.set_num_threads(cores)
    return "cpu"


def _device_display_name(device: str) -> str:
    """返回用于 UI 显示的设备名称。"""
    if device == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
            return f"GPU ⚡ {name[:20]}"
        except Exception:
            return "GPU ⚡ CUDA"
    if device == "mps":
        return "GPU ⚡ Apple Silicon"
    cores = os.cpu_count() or 4
    return f"CPU ({cores} 核)"


# 音频参数（需与 recorder.py 保持一致）
SAMPLE_RATE = 16000

# 实时转录参数
VAD_CHUNK_SECONDS = 1.5      # 每次积累多少秒后尝试识别
MAX_BUFFER_SECONDS = 30      # 最大音频缓冲区长度（秒）
SILENCE_THRESHOLD = 0.01     # 静音能量阈值
SILENCE_DURATION = 1.2       # 静音多少秒后认为一段话结束（秒）
MIN_SPEECH_SECONDS = 0.3     # 最少有多少秒有效语音才触发识别

# ── 中文简/繁体处理 ──────────────────────────────────────────
# Whisper 只有 "zh" 语言码，通过 initial_prompt 引导输出简/繁体字形。
_LANGUAGE_ALIAS = {
    "zh-Hans": "zh",    # 简体 → Whisper 内部码
    "zh-Hant": "zh",    # 繁体 → Whisper 内部码
}

_INITIAL_PROMPTS = {
    # 简体：用简体字的提示词，引导模型进入简体输出模式
    "zh-Hans": "以下是用简体中文输入的内容：",
    # 繁体：用繁体字的提示词，引导模型进入繁体输出模式
    "zh-Hant": "以下是以繁體中文輸入的內容：",
}


class RealtimeTranscriber:
    """
    实时语音转录器。
    
    接收来自 AudioRecorder 的音频块，
    自动检测语音活动（VAD），
    在检测到停顿时调用 Whisper 进行识别，
    通过回调将结果实时返回给调用方。
    """

    def __init__(self, model_name: str = "small",
                 language: str = None,
                 on_partial: callable = None,
                 on_final: callable = None,
                 on_model_loaded: callable = None,
                 on_error: callable = None):
        """
        初始化转录器。

        Args:
            model_name: Whisper 模型名称（tiny/base/small/medium/large）
            language: 语言代码（如 "zh" 中文，None 为自动检测）
            on_partial: 中间结果回调（识别中的文字）
            on_final: 最终结果回调（识别完成的一段文字）
            on_model_loaded: 模型加载完成回调
            on_error: 错误回调
        """
        self.model_name = model_name
        self.language = language
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_model_loaded = on_model_loaded
        self._on_error = on_error

        self._model = None
        self._model_lock = threading.Lock()
        self._is_loaded = False
        self._device = "cpu"          # 实际使用的设备
        self._device_display = "CPU"  # UI 显示名称

        # 音频缓冲区（存储连续语音段）
        self._audio_buffer = np.array([], dtype=np.float32)
        self._buffer_lock = threading.Lock()

        # VAD 状态
        self._silence_frames = 0
        self._speech_frames = 0
        self._is_speaking = False
        self._pending_chunks = np.array([], dtype=np.float32)
        self._chunk_lock = threading.Lock()

        # 转录工作线程
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._transcribe_event = threading.Event()

    # ─────────────────────────────────────────
    # 模型管理
    # ─────────────────────────────────────────

    def load_model_async(self):
        """在后台线程中加载 Whisper 模型（避免阻塞 UI）。"""
        t = threading.Thread(target=self._load_model, daemon=True)
        t.start()

    def _load_model(self):
        """加载 Whisper 模型（自动选择 GPU/CPU）。"""
        try:
            device = _select_device()
            with self._model_lock:
                self._model = whisper.load_model(
                    self.model_name,
                    device=device,
                )
                self._is_loaded = True
                self._device = device
                self._device_display = _device_display_name(device)
            if self._on_model_loaded:
                self._on_model_loaded()
        except Exception as e:
            if self._on_error:
                self._on_error(e)

    @property
    def device_display(self) -> str:
        """返回当前使用设备的显示名称（如 'GPU ⚡ RTX 3080'）。"""
        return self._device_display

    @property
    def is_loaded(self):
        return self._is_loaded

    # ─────────────────────────────────────────
    # 转录工作线程
    # ─────────────────────────────────────────

    def start_worker(self):
        """启动后台转录工作线程。"""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True
        )
        self._worker_thread.start()

    def stop_worker(self):
        """停止后台转录工作线程。"""
        self._stop_event.set()
        self._transcribe_event.set()  # 唤醒工作线程使其退出

    def _worker_loop(self):
        """后台工作线程：等待事件触发，执行转录。"""
        while not self._stop_event.is_set():
            triggered = self._transcribe_event.wait(timeout=0.5)
            if triggered:
                self._transcribe_event.clear()
                if not self._stop_event.is_set():
                    self._do_transcribe()

    def _do_transcribe(self):
        """执行 Whisper 识别。"""
        with self._buffer_lock:
            if len(self._audio_buffer) < int(MIN_SPEECH_SECONDS * SAMPLE_RATE):
                return
            audio_to_transcribe = self._audio_buffer.copy()
            self._audio_buffer = np.array([], dtype=np.float32)

        if not self._is_loaded:
            return

        try:
            with self._model_lock:
                use_fp16 = (self._device == "cuda")  # GPU 用 FP16 更快

                # 处理中文简/繁体：映射到 Whisper 内部语言码，并加提示词引导字形
                lang = self.language
                whisper_lang = _LANGUAGE_ALIAS.get(lang, lang)
                initial_prompt = _INITIAL_PROMPTS.get(lang, None)

                options = dict(
                    language=whisper_lang,
                    fp16=use_fp16,
                    task="transcribe",
                    beam_size=1,
                    best_of=1,
                    temperature=0,
                    condition_on_previous_text=False,
                )
                if initial_prompt:
                    options["initial_prompt"] = initial_prompt

                result = self._model.transcribe(audio_to_transcribe, **options)

            text = result.get("text", "").strip()
            if text and self._on_final:
                self._on_final(text)

        except Exception as e:
            if self._on_error:
                self._on_error(e)

    # ─────────────────────────────────────────
    # 音频输入（由 AudioRecorder 回调驱动）
    # ─────────────────────────────────────────

    def feed_audio(self, chunk: np.ndarray):
        """
        接收来自录音模块的音频块，进行 VAD 并决定是否触发转录。
        
        此方法在录音回调线程中调用，必须快速返回。
        """
        if not self._is_loaded:
            return

        energy = float(np.sqrt(np.mean(chunk ** 2)))
        is_speech = energy > SILENCE_THRESHOLD

        chunk_samples = len(chunk)
        silence_threshold_frames = int(SILENCE_DURATION * SAMPLE_RATE / chunk_samples)

        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
            self._is_speaking = True

            with self._buffer_lock:
                self._audio_buffer = np.concatenate([self._audio_buffer, chunk])
                # 防止缓冲区过大
                max_samples = MAX_BUFFER_SECONDS * SAMPLE_RATE
                if len(self._audio_buffer) > max_samples:
                    self._audio_buffer = self._audio_buffer[-max_samples:]
        else:
            if self._is_speaking:
                self._silence_frames += 1
                # 将静音也加入缓冲（保证连续性）
                with self._buffer_lock:
                    self._audio_buffer = np.concatenate([self._audio_buffer, chunk])

                # 静音持续足够长 → 触发转录
                if self._silence_frames >= silence_threshold_frames:
                    self._is_speaking = False
                    self._silence_frames = 0
                    self._speech_frames = 0
                    self._transcribe_event.set()

    def flush(self):
        """强制转录当前缓冲区中剩余的音频（停止录音时调用）。"""
        with self._buffer_lock:
            has_audio = len(self._audio_buffer) > int(MIN_SPEECH_SECONDS * SAMPLE_RATE)

        if has_audio:
            self._transcribe_event.set()

    def reset_buffer(self):
        """清空音频缓冲区。"""
        with self._buffer_lock:
            self._audio_buffer = np.array([], dtype=np.float32)
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0
