"""
app/audio_stream.py
--------------------
Real-time microphone audio streaming and chunked processing.
Captures continuous audio from laptop microphone using sounddevice.

Architecture:
- Continuous ring buffer captures audio from mic
- Sliding window (2s chunks with 0.5s overlap) fed to ASR pipeline
- Thread-safe queue for communication between capture and processing threads
"""

import numpy as np
import threading
import queue
import time
import logging
from typing import Callable, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Audio capture settings
SAMPLE_RATE = 16000       # 16kHz for Whisper
CHANNELS = 1              # Mono
CHUNK_DURATION = 2.0      # seconds per processing chunk
OVERLAP_DURATION = 0.5    # seconds of overlap between chunks
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
OVERLAP_SAMPLES = int(SAMPLE_RATE * OVERLAP_DURATION)
SILENCE_THRESHOLD = 0.01  # RMS below this = silence


@dataclass
class AudioChunk:
    """Container for an audio chunk with metadata."""
    audio: np.ndarray
    timestamp: float
    chunk_id: int
    is_silence: bool = False


class MicrophoneStream:
    """
    Continuous microphone capture with chunked output.

    Usage:
        stream = MicrophoneStream()
        stream.start(callback=my_callback)
        # ... later ...
        stream.stop()
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_duration: float = CHUNK_DURATION,
        overlap_duration: float = OVERLAP_DURATION,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.overlap_samples = int(sample_rate * overlap_duration)
        self.device = device

        self._stream = None
        self._is_running = False
        self._audio_buffer = np.array([], dtype=np.float32)
        self._chunk_id = 0
        self._lock = threading.Lock()
        self._audio_queue: queue.Queue = queue.Queue(maxsize=10)
        self._callback: Optional[Callable] = None
        self._processing_thread: Optional[threading.Thread] = None
        self.audio_levels: List[float] = []  # RMS levels for waveform display

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time_info, status):
        """Called by sounddevice for each audio block (runs in audio thread)."""
        if status:
            logger.warning(f"Audio stream status: {status}")

        # Convert to mono float32
        audio_data = indata[:, 0].astype(np.float32)

        # Track audio level for waveform display
        rms = float(np.sqrt(np.mean(audio_data ** 2)))
        self.audio_levels.append(rms)
        if len(self.audio_levels) > 50:
            self.audio_levels.pop(0)

        # Append to buffer (thread-safe)
        with self._lock:
            self._audio_buffer = np.concatenate([self._audio_buffer, audio_data])

            # When we have enough samples for a chunk, extract and queue it
            while len(self._audio_buffer) >= self.chunk_samples:
                chunk_audio = self._audio_buffer[:self.chunk_samples].copy()
                # Keep overlap for next chunk
                self._audio_buffer = self._audio_buffer[
                    self.chunk_samples - self.overlap_samples:
                ]

                chunk = AudioChunk(
                    audio=chunk_audio,
                    timestamp=time.time(),
                    chunk_id=self._chunk_id,
                    is_silence=self._is_silence(chunk_audio),
                )
                self._chunk_id += 1

                try:
                    self._audio_queue.put_nowait(chunk)
                except queue.Full:
                    # Drop oldest chunk if queue is full (real-time priority)
                    try:
                        self._audio_queue.get_nowait()
                        self._audio_queue.put_nowait(chunk)
                    except queue.Empty:
                        pass

    def _is_silence(self, audio: np.ndarray) -> bool:
        """Detect if audio chunk is mostly silence."""
        rms = np.sqrt(np.mean(audio ** 2))
        return rms < SILENCE_THRESHOLD

    def _processing_loop(self):
        """Background thread that feeds audio chunks to the callback."""
        logger.info("Audio processing loop started")
        while self._is_running:
            try:
                chunk = self._audio_queue.get(timeout=1.0)
                if self._callback and not chunk.is_silence:
                    try:
                        self._callback(chunk)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing loop error: {e}")

        logger.info("Audio processing loop stopped")

    def start(self, callback: Optional[Callable] = None) -> bool:
        """
        Start microphone capture.

        Args:
            callback: Called with AudioChunk for each non-silent chunk

        Returns:
            True if started successfully
        """
        if self._is_running:
            logger.warning("Stream already running")
            return True

        try:
            import sounddevice as sd

            self._callback = callback
            self._is_running = True
            self._audio_buffer = np.array([], dtype=np.float32)
            self.audio_levels = []

            # Start background processing thread
            self._processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True,
                name="AudioProcessing",
            )
            self._processing_thread.start()

            # Start sounddevice stream
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype="float32",
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                device=self.device,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info(f"Microphone stream started at {self.sample_rate}Hz")
            return True

        except Exception as e:
            self._is_running = False
            logger.error(f"Failed to start microphone: {e}")
            raise

    def stop(self):
        """Stop microphone capture cleanly."""
        self._is_running = False

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            except Exception as e:
                logger.warning(f"Error stopping stream: {e}")

        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)

        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("Microphone stream stopped")

    @property
    def is_running(self) -> bool:
        return self._is_running

    def get_current_level(self) -> float:
        """Get current audio input level (0-1 range)."""
        if self.audio_levels:
            return min(1.0, self.audio_levels[-1] * 10)
        return 0.0

    def get_level_history(self, n: int = 30) -> List[float]:
        """Get recent audio level history for waveform display."""
        levels = self.audio_levels[-n:] if len(self.audio_levels) > n else self.audio_levels
        return [min(1.0, l * 10) for l in levels]

    @staticmethod
    def list_devices() -> list:
        """List available audio input devices."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            input_devices = []
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    input_devices.append({
                        "id": i,
                        "name": d['name'],
                        "channels": d['max_input_channels'],
                        "sample_rate": d['default_samplerate'],
                    })
            return input_devices
        except Exception as e:
            logger.error(f"Could not list devices: {e}")
            return []
