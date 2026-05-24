"""
pipeline.py
------------
Main orchestration pipeline for the Indian Multilingual Speech Translation System.

Ties together:
- Noise Reduction
- ASR (Whisper)
- Language Detection
- Emotion Detection
- Translation (Helsinki-NLP / Google)
- Text-to-Speech (gTTS)

Can be used standalone or via the Streamlit UI.
"""

import numpy as np
import logging
import time
from typing import Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Complete result from the translation pipeline."""
    # Input metadata
    input_mode: str           # "microphone" or "file"
    timestamp: float = 0.0

    # ASR output
    transcript: str = ""
    asr_language: str = ""
    asr_language_name: str = ""
    asr_processing_time: float = 0.0
    asr_confidence: float = 0.0

    # Language detection
    detected_language: str = ""
    detected_language_code: str = ""
    lang_confidence: float = 0.0

    # Emotion
    emotion: str = "neutral"
    emotion_emoji: str = "😐"
    emotion_display: str = "Neutral"
    emotion_color: str = "#90EE90"
    emotion_confidence: float = 0.0

    # Translation
    translated_text: str = ""
    target_language: str = ""
    target_language_code: str = ""
    translation_method: str = ""
    translation_time: float = 0.0

    # TTS
    audio_bytes: Optional[bytes] = None
    audio_file_path: Optional[str] = None

    # Overall
    total_processing_time: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop("audio_bytes", None)  # Don't serialize bytes
        return d


class TranslationPipeline:
    """
    End-to-end Indian language speech translation pipeline.

    Designed for:
    - CPU-only inference
    - Lazy model loading (only loads models when first needed)
    - Both real-time microphone and file upload modes
    """

    def __init__(
        self,
        asr_model_size: str = "small",
        enable_noise_reduction: bool = True,
        enable_emotion: bool = True,
        enable_tts: bool = True,
        tts_output_dir: str = "data/tts_output",
    ):
        self.asr_model_size = asr_model_size
        self.enable_noise_reduction = enable_noise_reduction
        self.enable_emotion = enable_emotion
        self.enable_tts = enable_tts
        self.tts_output_dir = tts_output_dir

        # Lazy-loaded components
        self._noise_filter = None
        self._asr = None
        self._lang_detector = None
        self._emotion_detector = None
        self._translator = None
        self._tts = None

        logger.info("Pipeline initialized (models will load on first use)")

    # ─── Lazy Loaders ────────────────────────────────────────────────────────

    @property
    def noise_filter(self):
        if self._noise_filter is None:
            from noise_reduction.noise_filter import NoiseFilter
            self._noise_filter = NoiseFilter()
        return self._noise_filter

    @property
    def asr(self):
        if self._asr is None:
            from asr.speech_to_text import SpeechToText
            self._asr = SpeechToText(model_size=self.asr_model_size)
        return self._asr

    @property
    def lang_detector(self):
        if self._lang_detector is None:
            from language_detection.detector import LanguageDetector
            self._lang_detector = LanguageDetector()
        return self._lang_detector

    @property
    def emotion_detector(self):
        if self._emotion_detector is None:
            from emotion.inference import EmotionDetector
            self._emotion_detector = EmotionDetector()
        return self._emotion_detector

    @property
    def translator(self):
        if self._translator is None:
            from nmt.translator import Translator
            self._translator = Translator()
        return self._translator

    @property
    def tts(self):
        if self._tts is None:
            from tts.text_to_speech import TextToSpeech
            self._tts = TextToSpeech(output_dir=self.tts_output_dir)
        return self._tts

    # ─── Core Pipeline ───────────────────────────────────────────────────────

    def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_language: str = "English",
        source_language_hint: Optional[str] = None,
        generate_tts: bool = True,
        input_mode: str = "microphone",
    ) -> TranslationResult:
        """
        Process audio through the full pipeline.

        Args:
            audio: Float32 numpy array of audio samples
            sample_rate: Sample rate of audio
            target_language: Target language for translation (display name)
            source_language_hint: Optional hint for ASR language detection
            generate_tts: Whether to synthesize translated speech
            input_mode: "microphone" or "file"

        Returns:
            TranslationResult with all outputs
        """
        result = TranslationResult(
            input_mode=input_mode,
            timestamp=time.time(),
            target_language=target_language,
        )
        overall_start = time.time()

        try:
            # ── Step 1: Noise Reduction ─────────────────────────────────────
            if self.enable_noise_reduction:
                try:
                    audio, sample_rate = self.noise_filter.process(
                        audio,
                        apply_noise_reduction=True,
                        apply_normalization=True,
                        apply_trim=True,
                    )
                except Exception as e:
                    logger.warning(f"Noise reduction failed: {e}, using raw audio")

            # ── Step 2: ASR ─────────────────────────────────────────────────
            asr_result = self.asr.transcribe(
                audio,
                sample_rate=sample_rate,
                language=source_language_hint,
            )

            result.transcript = asr_result.get("text", "")
            result.asr_language = asr_result.get("language", "en")
            result.asr_language_name = asr_result.get("language_name", "English")
            result.asr_processing_time = asr_result.get("processing_time", 0)

            if not result.transcript:
                result.error = "No speech detected"
                result.success = False
                return result

            # ── Step 3: Language Detection ───────────────────────────────────
            lang_result = self.lang_detector.detect(result.transcript)
            result.detected_language = lang_result.get("language", result.asr_language_name)
            result.detected_language_code = lang_result.get("language_code", "en")
            result.lang_confidence = lang_result.get("confidence", 0.5)

            # Use ASR language if detection is uncertain
            if result.lang_confidence < 0.6:
                result.detected_language = result.asr_language_name
                result.detected_language_code = result.asr_language

            # ── Step 4: Emotion Detection ─────────────────────────────────
            if self.enable_emotion:
                try:
                    emotion_result = self.emotion_detector.detect(result.transcript)
                    result.emotion = emotion_result.get("emotion", "neutral")
                    result.emotion_emoji = emotion_result.get("emoji", "😐")
                    result.emotion_display = emotion_result.get("display_name", "Neutral")
                    result.emotion_color = emotion_result.get("color", "#90EE90")
                    result.emotion_confidence = emotion_result.get("confidence", 0.5)
                except Exception as e:
                    logger.warning(f"Emotion detection failed: {e}")

            # ── Step 5: Translation ──────────────────────────────────────────
            trans_start = time.time()
            try:
                trans_result = self.translator.translate(
                    text=result.transcript,
                    source_language=result.detected_language_code,
                    target_language=target_language,
                )
                result.translated_text = trans_result.get("translated_text", result.transcript)
                result.translation_method = trans_result.get("method", "unknown")
                result.translation_time = time.time() - trans_start
                result.target_language_code = trans_result.get("target_language", "en")
            except Exception as e:
                logger.error(f"Translation failed: {e}")
                result.translated_text = result.transcript
                result.error = f"Translation failed: {e}"

            # ── Step 6: Text-to-Speech ───────────────────────────────────────
            if generate_tts and self.enable_tts and result.translated_text:
                try:
                    tts_result = self.tts.synthesize(
                        text=result.translated_text,
                        language=target_language,
                        save_to_file=True,
                    )
                    if tts_result["success"]:
                        result.audio_bytes = tts_result.get("audio_bytes")
                        result.audio_file_path = tts_result.get("file_path")
                except Exception as e:
                    logger.warning(f"TTS failed: {e}")

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            result.success = False
            result.error = str(e)

        result.total_processing_time = round(time.time() - overall_start, 2)
        return result

    def process_file(
        self,
        file_path: str,
        target_language: str = "English",
        source_language_hint: Optional[str] = None,
        generate_tts: bool = True,
    ) -> TranslationResult:
        """
        Process an audio file through the pipeline.

        Args:
            file_path: Path to audio file (.wav, .mp3)
            target_language: Target translation language
            source_language_hint: Optional language hint
            generate_tts: Whether to generate TTS output

        Returns:
            TranslationResult
        """
        result = TranslationResult(
            input_mode="file",
            timestamp=time.time(),
            target_language=target_language,
        )
        overall_start = time.time()

        try:
            # Load and clean audio
            audio, sr = self.noise_filter.load_audio(file_path)

            if self.enable_noise_reduction:
                audio, sr = self.noise_filter.process(
                    audio,
                    apply_noise_reduction=True,
                    apply_normalization=True,
                    apply_trim=True,
                )

            # Run ASR on the full audio
            asr_result = self.asr.transcribe(
                audio,
                sample_rate=sr,
                language=source_language_hint,
            )

            result.transcript = asr_result.get("text", "")
            result.asr_language = asr_result.get("language", "en")
            result.asr_language_name = asr_result.get("language_name", "English")
            result.asr_processing_time = asr_result.get("processing_time", 0)

            if not result.transcript:
                result.error = "No speech detected in file"
                result.success = False
                result.total_processing_time = time.time() - overall_start
                return result

            # Run remaining pipeline steps
            return self.process_audio(
                audio=audio,
                sample_rate=sr,
                target_language=target_language,
                source_language_hint=source_language_hint,
                generate_tts=generate_tts,
                input_mode="file",
            )

        except Exception as e:
            logger.error(f"File processing error: {e}", exc_info=True)
            result.success = False
            result.error = str(e)
            result.total_processing_time = time.time() - overall_start
            return result

    def warmup(self):
        """
        Pre-load all models to avoid cold-start delays during use.
        Call this during app initialization.
        """
        logger.info("Warming up pipeline components...")
        try:
            _ = self.noise_filter
            logger.info("✓ Noise filter ready")
        except Exception as e:
            logger.warning(f"✗ Noise filter: {e}")

        try:
            _ = self.asr
            logger.info("✓ ASR model ready")
        except Exception as e:
            logger.warning(f"✗ ASR: {e}")

        try:
            _ = self.lang_detector
            logger.info("✓ Language detector ready")
        except Exception as e:
            logger.warning(f"✗ Language detector: {e}")

        try:
            if self.enable_emotion:
                _ = self.emotion_detector
                logger.info("✓ Emotion detector ready")
        except Exception as e:
            logger.warning(f"✗ Emotion detector: {e}")

        try:
            _ = self.translator
            logger.info("✓ Translator ready")
        except Exception as e:
            logger.warning(f"✗ Translator: {e}")

        try:
            if self.enable_tts:
                _ = self.tts
                logger.info("✓ TTS ready")
        except Exception as e:
            logger.warning(f"✗ TTS: {e}")

        logger.info("Pipeline warmup complete")
