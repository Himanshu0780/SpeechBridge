"""
utils/helpers.py
-----------------
Common utility functions used across the project.
"""

import os
import time
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_audio_bytes(audio_bytes: bytes, suffix: str = ".wav",
                     output_dir: Optional[str] = None) -> str:
    """
    Save audio bytes to a temporary file.
    Returns the file path.
    """
    if output_dir:
        ensure_dir(output_dir)
        ts = int(time.time() * 1000)
        path = os.path.join(output_dir, f"audio_{ts}{suffix}")
        with open(path, "wb") as f:
            f.write(audio_bytes)
        return path
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name


def audio_to_base64(audio_bytes: bytes) -> str:
    """Convert audio bytes to base64 string for HTML embedding."""
    import base64
    return base64.b64encode(audio_bytes).decode("utf-8")


def hash_text(text: str) -> str:
    """Return a short hash of text for caching/deduplication."""
    return hashlib.md5(text.encode()).hexdigest()[:8]


def chunk_text(text: str, max_length: int = 400) -> list:
    """
    Split text into chunks at sentence boundaries.
    Used to handle long texts in translation and TTS.
    """
    import re
    if len(text) <= max_length:
        return [text]

    # Split at sentence boundaries
    sentences = re.split(r'(?<=[.!?।॥\n])\s*', text)
    chunks = []
    current = ""

    for sent in sentences:
        if not sent.strip():
            continue
        if len(current) + len(sent) + 1 <= max_length:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            current = sent

    if current:
        chunks.append(current)

    return chunks if chunks else [text]


def normalize_audio_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 audio to int16 for WAV file compatibility."""
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 audio to float32."""
    return audio.astype(np.float32) / 32768.0


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def truncate_text(text: str, max_len: int = 100, suffix: str = "...") -> str:
    """Truncate text to max_len characters."""
    if len(text) <= max_len:
        return text
    return text[:max_len - len(suffix)] + suffix


def estimate_reading_time(text: str, wpm: int = 200) -> float:
    """Estimate time to read text in seconds."""
    words = len(text.split())
    return (words / wpm) * 60


def get_file_info(file_path: str) -> dict:
    """Get file metadata."""
    path = Path(file_path)
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {
        "exists": True,
        "name": path.name,
        "size_bytes": stat.st_size,
        "size_kb": round(stat.st_size / 1024, 1),
        "extension": path.suffix.lower(),
        "modified": time.ctime(stat.st_mtime),
    }


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
        if self.label:
            logger.debug(f"{self.label}: {self.elapsed:.3f}s")

    def __str__(self):
        return f"{self.elapsed:.3f}s"


class RollingAverage:
    """Compute a rolling average over a sliding window."""

    def __init__(self, window: int = 10):
        self.window = window
        self._values = []

    def update(self, value: float) -> float:
        self._values.append(value)
        if len(self._values) > self.window:
            self._values.pop(0)
        return self.average

    @property
    def average(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    def reset(self):
        self._values = []
