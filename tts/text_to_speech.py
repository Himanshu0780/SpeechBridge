"""
tts/text_to_speech.py
----------------------
Text-to-Speech synthesis using gTTS (Google Text-to-Speech).
Supports all major Indian languages.
"""

import logging
import io
import os
import time
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# gTTS language code mapping for Indian languages
GTTS_LANG_CODES = {
    "Hindi":     "hi",
    "English":   "en",
    "Tamil":     "ta",
    "Telugu":    "te",
    "Bengali":   "bn",
    "Marathi":   "mr",
    "Gujarati":  "gu",
    "Kannada":   "kn",
    "Malayalam": "ml",
    "Punjabi":   "pa",
    "Urdu":      "ur",
    # ISO codes (passthrough)
    "hi": "hi", "en": "en", "ta": "ta", "te": "te",
    "bn": "bn", "mr": "mr", "gu": "gu", "kn": "kn",
    "ml": "ml", "pa": "pa", "ur": "ur",
}

# Languages where gTTS TLD can improve accent quality
GTTS_TLD_MAP = {
    "en": "co.in",  # Indian English accent
}

# Output directory for generated audio files
OUTPUT_DIR = Path("data/tts_output")


class TextToSpeech:
    """
    Google Text-to-Speech wrapper for Indian languages.

    Features:
    - Supports all major Indian scripts
    - Saves to MP3 file and returns bytes
    - Generates unique filenames per session
    - Handles TTS failures gracefully
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._verify_gtts()

    def _verify_gtts(self):
        """Verify gTTS is installed."""
        try:
            from gtts import gTTS
            logger.info("gTTS available")
        except ImportError:
            logger.error("gTTS not installed. Run: pip install gtts")

    def synthesize(
        self,
        text: str,
        language: str = "en",
        slow: bool = False,
        save_to_file: bool = True,
    ) -> Dict:
        """
        Convert text to speech audio.

        Args:
            text: Text to synthesize
            language: Language name (e.g., "Hindi") or ISO code (e.g., "hi")
            slow: Whether to use slower speech rate
            save_to_file: Whether to save MP3 to disk

        Returns:
            Dict with keys: audio_bytes, file_path, language, duration_estimate,
                           success, error
        """
        if not text or len(text.strip()) == 0:
            return self._empty_result(language)

        # Normalize language code
        lang_code = GTTS_LANG_CODES.get(language, "en")
        tld = GTTS_TLD_MAP.get(lang_code, "com")

        try:
            from gtts import gTTS

            # Split long texts to avoid TTS timeout
            chunks = self._split_text(text, max_chars=500)
            audio_parts = []

            for chunk in chunks:
                if not chunk.strip():
                    continue
                tts = gTTS(text=chunk, lang=lang_code, slow=slow, tld=tld)
                chunk_buffer = io.BytesIO()
                tts.write_to_fp(chunk_buffer)
                audio_parts.append(chunk_buffer.getvalue())

            if not audio_parts:
                return self._empty_result(language)

            # Combine audio parts
            combined_audio = b"".join(audio_parts)

            # Save to file if requested
            file_path = None
            if save_to_file:
                timestamp = int(time.time() * 1000)
                filename = f"tts_{lang_code}_{timestamp}.mp3"
                file_path = self.output_dir / filename
                with open(file_path, "wb") as f:
                    f.write(combined_audio)
                logger.debug(f"TTS saved to: {file_path}")

            # Estimate duration: ~150 words/min average
            word_count = len(text.split())
            duration_estimate = (word_count / 150) * 60  # seconds

            return {
                "audio_bytes": combined_audio,
                "file_path": str(file_path) if file_path else None,
                "language": language,
                "lang_code": lang_code,
                "text_length": len(text),
                "word_count": word_count,
                "duration_estimate": round(duration_estimate, 1),
                "success": True,
                "error": None,
            }

        except Exception as e:
            logger.error(f"TTS error: {e}")
            return {
                "audio_bytes": None,
                "file_path": None,
                "language": language,
                "lang_code": lang_code,
                "text_length": len(text),
                "word_count": 0,
                "duration_estimate": 0,
                "success": False,
                "error": str(e),
            }

    def synthesize_to_bytes(self, text: str, language: str = "en") -> Optional[bytes]:
        """
        Quick synthesis returning only audio bytes.
        Returns None if synthesis fails.
        """
        result = self.synthesize(text, language, save_to_file=False)
        return result.get("audio_bytes") if result["success"] else None

    def _split_text(self, text: str, max_chars: int = 500) -> list:
        """
        Split long text into chunks respecting sentence boundaries.
        gTTS has a practical limit on text length.
        """
        if len(text) <= max_chars:
            return [text]

        # Split by sentence endings
        import re
        sentences = re.split(r'(?<=[.!?।])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # Handle very long sentences by splitting at word boundaries
                if len(sentence) > max_chars:
                    words = sentence.split()
                    sub_chunk = ""
                    for word in words:
                        if len(sub_chunk) + len(word) + 1 <= max_chars:
                            sub_chunk += (" " if sub_chunk else "") + word
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            sub_chunk = word
                    if sub_chunk:
                        chunks.append(sub_chunk)
                    current_chunk = ""
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _empty_result(self, language: str) -> Dict:
        return {
            "audio_bytes": None,
            "file_path": None,
            "language": language,
            "lang_code": GTTS_LANG_CODES.get(language, "en"),
            "text_length": 0,
            "word_count": 0,
            "duration_estimate": 0,
            "success": False,
            "error": "Empty input text",
        }

    def cleanup_old_files(self, max_files: int = 20):
        """Remove old TTS files to prevent disk space buildup."""
        try:
            files = sorted(
                self.output_dir.glob("tts_*.mp3"),
                key=lambda f: f.stat().st_mtime
            )
            for old_file in files[:-max_files]:
                old_file.unlink()
                logger.debug(f"Deleted old TTS file: {old_file}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    @staticmethod
    def get_supported_languages() -> Dict[str, str]:
        """Return supported language names and their gTTS codes."""
        return {k: v for k, v in GTTS_LANG_CODES.items() if len(k) > 2}
