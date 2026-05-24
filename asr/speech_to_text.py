"""
asr/speech_to_text.py
----------------------
Automatic Speech Recognition (ASR) module using OpenAI Whisper.
Optimized for Indian languages and accents on CPU hardware.

Whisper "small" model provides excellent multilingual support including
all major Indian languages while remaining feasible on laptop CPUs.
"""

import numpy as np
import logging
import time
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Mapping from Whisper's detected language codes to our display names
WHISPER_LANG_MAP = {
    "hi": "Hindi",
    "en": "English",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "or": "Odia",
    "as": "Assamese",
}

# Indian language hints for Whisper (helps with detection accuracy)
INDIAN_LANGUAGE_CODES = list(WHISPER_LANG_MAP.keys())


class SpeechToText:
    """
    Whisper-based ASR engine optimized for Indian languages.

    Uses whisper-small model which supports 99 languages including
    all major Indian languages. Runs on CPU in ~2-4s for 5s audio clips.
    """

    def __init__(self, model_size: str = "small", device: str = "cpu"):
        """
        Initialize Whisper ASR model.

        Args:
            model_size: "tiny", "base", "small", "medium"
                       "small" recommended for accuracy/speed balance on CPU
            device: "cpu" or "cuda"
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Whisper model with error handling."""
        try:
            import whisper
            logger.info(f"Loading Whisper {self.model_size} model...")
            start_time = time.time()
            self.model = whisper.load_model(self.model_size, device=self.device)
            elapsed = time.time() - start_time
            logger.info(f"Whisper model loaded in {elapsed:.1f}s")
        except ImportError:
            logger.error("openai-whisper not installed. Run: pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper.

        Args:
            audio: Float32 numpy array of audio samples
            sample_rate: Sample rate of audio (Whisper expects 16000)
            language: Optional language hint (ISO 639-1 code, e.g., "hi" for Hindi)
            task: "transcribe" or "translate" (translate always outputs English)

        Returns:
            Dict with keys: text, language, language_name, confidence, segments, duration
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if len(audio) == 0:
            return self._empty_result()

        try:
            import whisper

            # Ensure audio is float32 and at 16kHz
            audio = audio.astype(np.float32)
            if sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

            # Pad or trim to Whisper's expected chunk size (30s max)
            audio = whisper.pad_or_trim(audio)

            start_time = time.time()

            # Build decode options
            decode_options = {
                "task": task,
                "fp16": False,  # CPU doesn't support fp16
                "beam_size": 3,  # Balance speed/accuracy
                "best_of": 1,
                "temperature": 0.0,  # Greedy decoding for speed
            }

            if language:
                decode_options["language"] = language

            # Run transcription
            result = self.model.transcribe(audio, **decode_options)

            elapsed = time.time() - start_time
            logger.debug(f"ASR completed in {elapsed:.2f}s")

            detected_lang = result.get("language", "en")
            lang_name = WHISPER_LANG_MAP.get(detected_lang, detected_lang.upper())

            # Extract segment-level info
            segments = []
            for seg in result.get("segments", []):
                segments.append({
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "").strip(),
                })

            return {
                "text": result["text"].strip(),
                "language": detected_lang,
                "language_name": lang_name,
                "segments": segments,
                "processing_time": round(elapsed, 2),
                "duration": len(audio) / 16000,
                "success": True,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                "text": "",
                "language": "unknown",
                "language_name": "Unknown",
                "segments": [],
                "processing_time": 0,
                "duration": 0,
                "success": False,
                "error": str(e),
            }

    def transcribe_file(self, file_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe an audio file directly (Whisper handles loading internally).

        Args:
            file_path: Path to audio file (.wav, .mp3, .m4a, etc.)
            language: Optional language hint

        Returns:
            Same dict structure as transcribe()
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            import whisper
            start_time = time.time()

            options = {
                "task": "transcribe",
                "fp16": False,
                "beam_size": 3,
                "temperature": 0.0,
            }
            if language:
                options["language"] = language

            result = self.model.transcribe(str(file_path), **options)
            elapsed = time.time() - start_time

            detected_lang = result.get("language", "en")
            lang_name = WHISPER_LANG_MAP.get(detected_lang, detected_lang.upper())

            segments = [
                {
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "").strip(),
                }
                for seg in result.get("segments", [])
            ]

            return {
                "text": result["text"].strip(),
                "language": detected_lang,
                "language_name": lang_name,
                "segments": segments,
                "processing_time": round(elapsed, 2),
                "duration": segments[-1]["end"] if segments else 0,
                "success": True,
                "error": None,
            }

        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return {
                "text": "",
                "language": "unknown",
                "language_name": "Unknown",
                "segments": [],
                "processing_time": 0,
                "duration": 0,
                "success": False,
                "error": str(e),
            }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for silent/too-short audio."""
        return {
            "text": "",
            "language": "unknown",
            "language_name": "Unknown",
            "segments": [],
            "processing_time": 0,
            "duration": 0,
            "success": True,
            "error": "Audio too short or silent",
        }

    def get_supported_languages(self) -> Dict[str, str]:
        """Return supported Indian languages."""
        return WHISPER_LANG_MAP

    @property
    def model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "supported_languages": len(WHISPER_LANG_MAP),
            "max_audio_duration": 30,  # seconds
        }
