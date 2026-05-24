"""
noise_reduction/noise_filter.py
--------------------------------
Audio noise reduction and cleaning pipeline using librosa and noisereduce.
Optimized for Indian speech audio captured on laptop microphones.
"""

import numpy as np
import librosa
import soundfile as sf
import io
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Target sample rate for all audio processing
TARGET_SR = 16000


class NoiseFilter:
    """
    Handles noise reduction, normalization, and audio cleaning.
    Designed to improve ASR accuracy for Indian language speech.
    """

    def __init__(self, target_sr: int = TARGET_SR):
        self.target_sr = target_sr

    def load_audio(self, audio_input) -> Tuple[np.ndarray, int]:
        """
        Load audio from file path, bytes, or numpy array.
        Resamples to target_sr automatically.
        """
        try:
            if isinstance(audio_input, np.ndarray):
                return audio_input, self.target_sr

            elif isinstance(audio_input, (str,)):
                # File path
                audio, sr = librosa.load(audio_input, sr=self.target_sr, mono=True)
                return audio, sr

            elif isinstance(audio_input, (bytes, bytearray)):
                # Raw bytes (e.g., from microphone stream)
                audio_buffer = io.BytesIO(audio_input)
                try:
                    audio, sr = sf.read(audio_buffer)
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)  # Convert stereo to mono
                    if sr != self.target_sr:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                    return audio.astype(np.float32), self.target_sr
                except Exception:
                    # Fallback: treat as raw PCM float32
                    audio = np.frombuffer(audio_input, dtype=np.float32)
                    return audio, self.target_sr

            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise

    def reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply spectral noise reduction using noisereduce library.
        Uses the first 0.5 seconds as noise profile if available.
        """
        try:
            import noisereduce as nr
            # Use first 500ms as noise sample if audio is long enough
            noise_sample = audio[:sr // 2] if len(audio) > sr // 2 else audio
            reduced = nr.reduce_noise(
                y=audio,
                y_noise=noise_sample,
                sr=sr,
                stationary=False,       # Non-stationary noise for real-world audio
                prop_decrease=0.75,     # Reduce noise by 75%
            )
            return reduced.astype(np.float32)
        except ImportError:
            logger.warning("noisereduce not available, skipping noise reduction")
            return audio
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}, returning original audio")
            return audio

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Peak normalize audio to -1.0 dB to maximize ASR input quality.
        Prevents clipping while maximizing signal strength.
        """
        peak = np.max(np.abs(audio))
        if peak > 0:
            # Normalize to 0.95 to leave headroom
            audio = audio / peak * 0.95
        return audio.astype(np.float32)

    def trim_silence(self, audio: np.ndarray, sr: int,
                     top_db: int = 30) -> np.ndarray:
        """
        Trim leading and trailing silence from audio.
        Uses librosa's energy-based VAD.
        """
        try:
            trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            # Return original if trimming removed too much (< 0.2s remaining)
            if len(trimmed) < sr * 0.2:
                return audio
            return trimmed
        except Exception as e:
            logger.warning(f"Silence trimming failed: {e}")
            return audio

    def apply_preemphasis(self, audio: np.ndarray,
                          coeff: float = 0.97) -> np.ndarray:
        """
        Apply pre-emphasis filter to boost high frequencies.
        Improves ASR performance on consonant-heavy Indian languages.
        """
        return np.append(audio[0], audio[1:] - coeff * audio[:-1]).astype(np.float32)

    def process(self, audio_input,
                apply_noise_reduction: bool = True,
                apply_normalization: bool = True,
                apply_trim: bool = True,
                apply_preemphasis: bool = False) -> Tuple[np.ndarray, int]:
        """
        Full audio processing pipeline.

        Args:
            audio_input: File path, bytes, or numpy array
            apply_noise_reduction: Whether to run spectral noise reduction
            apply_normalization: Whether to normalize audio levels
            apply_trim: Whether to trim leading/trailing silence
            apply_preemphasis: Whether to apply pre-emphasis filter

        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        # Step 1: Load audio
        audio, sr = self.load_audio(audio_input)

        logger.debug(f"Loaded audio: {len(audio)/sr:.2f}s @ {sr}Hz")

        # Step 2: Trim silence (before noise reduction for efficiency)
        if apply_trim and len(audio) > sr * 0.5:
            audio = self.trim_silence(audio, sr)

        # Step 3: Noise reduction
        if apply_noise_reduction and len(audio) > sr * 0.3:
            audio = self.reduce_noise(audio, sr)

        # Step 4: Pre-emphasis (optional, helps with Indian accents)
        if apply_preemphasis:
            audio = self.apply_preemphasis(audio)

        # Step 5: Normalize
        if apply_normalization:
            audio = self.normalize_audio(audio)

        logger.debug(f"Processed audio: {len(audio)/sr:.2f}s")
        return audio, sr

    def audio_to_bytes(self, audio: np.ndarray, sr: int,
                       format: str = "wav") -> bytes:
        """Convert numpy audio array back to bytes for playback or storage."""
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format=format)
        buffer.seek(0)
        return buffer.read()

    def get_audio_stats(self, audio: np.ndarray, sr: int) -> dict:
        """
        Compute audio statistics useful for debugging and UI display.
        """
        duration = len(audio) / sr
        rms_energy = float(np.sqrt(np.mean(audio ** 2)))
        peak_amplitude = float(np.max(np.abs(audio)))

        # Simple SNR estimate
        signal_power = np.mean(audio ** 2)
        noise_floor = np.percentile(np.abs(audio), 10) ** 2
        snr_db = 10 * np.log10(signal_power / (noise_floor + 1e-10)) if noise_floor > 0 else 0

        return {
            "duration_seconds": round(duration, 2),
            "sample_rate": sr,
            "rms_energy": round(rms_energy, 4),
            "peak_amplitude": round(peak_amplitude, 4),
            "snr_db": round(float(snr_db), 1),
            "num_samples": len(audio),
        }
