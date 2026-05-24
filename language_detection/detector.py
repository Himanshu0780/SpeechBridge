"""
language_detection/detector.py
--------------------------------
Language detection for Indian languages using langdetect + heuristic rules.

langdetect works well for major Indian languages when the text is reasonably long.
For shorter texts, we add script-based heuristics for better accuracy.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Unicode script ranges for Indian languages
SCRIPT_RANGES = {
    "Hindi":     (0x0900, 0x097F),   # Devanagari
    "Marathi":   (0x0900, 0x097F),   # Devanagari (same script as Hindi)
    "Bengali":   (0x0980, 0x09FF),   # Bengali
    "Punjabi":   (0x0A00, 0x0A7F),   # Gurmukhi
    "Gujarati":  (0x0A80, 0x0AFF),   # Gujarati
    "Odia":      (0x0B00, 0x0B7F),   # Odia
    "Tamil":     (0x0B80, 0x0BFF),   # Tamil
    "Telugu":    (0x0C00, 0x0C7F),   # Telugu
    "Kannada":   (0x0C80, 0x0CFF),   # Kannada
    "Malayalam": (0x0D00, 0x0D7F),   # Malayalam
}

# langdetect language code to Indian language display name
LANGDETECT_MAP = {
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
}

# Reverse map: display name → ISO code
NAME_TO_CODE = {v: k for k, v in LANGDETECT_MAP.items()}

# All supported language names
SUPPORTED_LANGUAGES = list(set(LANGDETECT_MAP.values()))


class LanguageDetector:
    """
    Multi-strategy language detector optimized for Indian languages.

    Strategy:
    1. Script-based detection (most reliable for non-Latin scripts)
    2. langdetect statistical model (works well for longer texts)
    3. Fallback to "English" for Latin script text
    """

    def __init__(self):
        self._init_langdetect()

    def _init_langdetect(self):
        """Initialize langdetect with a fixed seed for reproducibility."""
        try:
            from langdetect import DetectorFactory
            DetectorFactory.seed = 42
            logger.info("langdetect initialized")
        except ImportError:
            logger.warning("langdetect not installed. Script-based detection only.")

    def detect_by_script(self, text: str) -> Optional[str]:
        """
        Detect language using Unicode script ranges.
        Highly accurate for Indian language scripts (non-Latin).

        Returns language name or None if script-based detection fails.
        """
        if not text or len(text.strip()) == 0:
            return None

        script_counts: Dict[str, int] = {}

        for char in text:
            code_point = ord(char)
            for lang, (start, end) in SCRIPT_RANGES.items():
                if start <= code_point <= end:
                    script_counts[lang] = script_counts.get(lang, 0) + 1

        if not script_counts:
            return None

        # Handle Devanagari ambiguity: Hindi vs Marathi
        # Both use Devanagari; use langdetect for disambiguation
        detected = max(script_counts, key=script_counts.get)

        # Require at least 30% of chars to be in detected script
        total_chars = len([c for c in text if not c.isspace()])
        if total_chars > 0 and script_counts[detected] / total_chars < 0.3:
            return None

        return detected

    def detect_by_langdetect(self, text: str) -> Tuple[Optional[str], float]:
        """
        Use langdetect statistical model for detection.

        Returns (language_name, confidence) or (None, 0.0)
        """
        try:
            from langdetect import detect_langs
            results = detect_langs(text)
            if results:
                top = results[0]
                lang_code = top.lang
                confidence = top.prob
                lang_name = LANGDETECT_MAP.get(lang_code, lang_code.upper())
                return lang_name, round(confidence, 3)
        except Exception as e:
            logger.debug(f"langdetect failed: {e}")
        return None, 0.0

    def detect(self, text: str) -> Dict:
        """
        Detect language using combined strategy.

        Args:
            text: Input text string

        Returns:
            Dict with keys: language, language_code, confidence, method
        """
        if not text or len(text.strip()) < 2:
            return self._default_result()

        # Strategy 1: Script-based (most reliable for non-Latin)
        script_lang = self.detect_by_script(text)
        if script_lang and script_lang not in ("Hindi", "Marathi"):
            # High confidence for unambiguous scripts
            return {
                "language": script_lang,
                "language_code": NAME_TO_CODE.get(script_lang, "xx"),
                "confidence": 0.97,
                "method": "script",
                "success": True,
            }

        # Strategy 2: langdetect for Latin script and Devanagari disambiguation
        lang_name, confidence = self.detect_by_langdetect(text)

        if lang_name and confidence > 0.5:
            # Trust langdetect result
            return {
                "language": lang_name,
                "language_code": NAME_TO_CODE.get(lang_name, "en"),
                "confidence": confidence,
                "method": "langdetect",
                "success": True,
            }

        # If script said Devanagari but langdetect is unsure, default to Hindi
        if script_lang in ("Hindi", "Marathi"):
            return {
                "language": script_lang,
                "language_code": NAME_TO_CODE.get(script_lang, "hi"),
                "confidence": 0.75,
                "method": "script_fallback",
                "success": True,
            }

        # Final fallback
        return {
            "language": lang_name or "English",
            "language_code": NAME_TO_CODE.get(lang_name or "English", "en"),
            "confidence": confidence or 0.5,
            "method": "fallback",
            "success": True,
        }

    def detect_multiple(self, text: str, top_k: int = 3) -> List[Dict]:
        """
        Return top-k language candidates with probabilities.
        Useful for ambiguous texts.
        """
        results = []
        try:
            from langdetect import detect_langs
            candidates = detect_langs(text)
            for candidate in candidates[:top_k]:
                lang_code = candidate.lang
                lang_name = LANGDETECT_MAP.get(lang_code, lang_code.upper())
                results.append({
                    "language": lang_name,
                    "language_code": lang_code,
                    "confidence": round(candidate.prob, 3),
                })
        except Exception:
            pass

        if not results:
            primary = self.detect(text)
            results = [primary]

        return results

    def _default_result(self) -> Dict:
        """Default result for empty/very short text."""
        return {
            "language": "English",
            "language_code": "en",
            "confidence": 0.5,
            "method": "default",
            "success": False,
        }

    @staticmethod
    def get_language_code(language_name: str) -> str:
        """Convert display name to ISO code."""
        return NAME_TO_CODE.get(language_name, "en")

    @staticmethod
    def get_language_name(language_code: str) -> str:
        """Convert ISO code to display name."""
        return LANGDETECT_MAP.get(language_code, language_code.upper())

    @staticmethod
    def get_supported_languages() -> List[str]:
        """Return list of supported language names."""
        return SUPPORTED_LANGUAGES
