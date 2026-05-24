"""
nmt/translator.py
------------------
Multilingual Neural Machine Translation for Indian languages.

Uses Helsinki-NLP/opus-mt models from HuggingFace for Indian language pairs.
Falls back to Google Translate API (via deep_translator) when a direct model
is not available for a specific language pair.

Supported translation pairs include all major Indian language combinations.
"""

import logging
import time
from typing import Dict, Optional, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)

# Helsinki-NLP model registry for Indian language pairs
# Format: {(src_lang, tgt_lang): model_name}
HELSINKI_MODELS = {
    # English ↔ Indian languages (most reliable pairs)
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "ta"): "Helsinki-NLP/opus-mt-en-ta",
    ("ta", "en"): "Helsinki-NLP/opus-mt-ta-en",
    ("en", "te"): "Helsinki-NLP/opus-mt-en-te",
    ("te", "en"): "Helsinki-NLP/opus-mt-mul-en",
    ("en", "bn"): "Helsinki-NLP/opus-mt-en-bn",
    ("bn", "en"): "Helsinki-NLP/opus-mt-bn-en",
    ("en", "mr"): "Helsinki-NLP/opus-mt-en-mr",
    ("mr", "en"): "Helsinki-NLP/opus-mt-mr-en",
    ("en", "gu"): "Helsinki-NLP/opus-mt-en-gu",
    ("gu", "en"): "Helsinki-NLP/opus-mt-gu-en",
    ("en", "kn"): "Helsinki-NLP/opus-mt-en-kn",
    ("kn", "en"): "Helsinki-NLP/opus-mt-kn-en",
    ("en", "ml"): "Helsinki-NLP/opus-mt-en-ml",
    ("ml", "en"): "Helsinki-NLP/opus-mt-ml-en",
    ("en", "pa"): "Helsinki-NLP/opus-mt-en-pa",
    ("pa", "en"): "Helsinki-NLP/opus-mt-pa-en",
}

# Language code to full name mapping
LANG_NAMES = {
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

# All supported language pairs (direct + via English pivot)
SUPPORTED_LANGUAGES = list(LANG_NAMES.keys())


class Translator:
    """
    Indian language translator with multiple backend support.

    Translation strategy:
    1. Direct Helsinki-NLP model (fastest, offline)
    2. Via English pivot: src→en→tgt (for Indian↔Indian pairs)
    3. Google Translate fallback (requires internet)
    """

    def __init__(self, use_gpu: bool = False):
        self.device = "cuda" if use_gpu else "cpu"
        self._model_cache: Dict[str, Tuple] = {}  # {model_name: (tokenizer, model)}
        logger.info("Translator initialized (lazy model loading)")

    def _load_helsinki_model(self, model_name: str) -> Optional[Tuple]:
        """Load and cache a Helsinki-NLP translation model."""
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        try:
            from transformers import MarianMTModel, MarianTokenizer
            logger.info(f"Loading translation model: {model_name}")
            start = time.time()

            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            model.eval()

            # Move to device (CPU for our use case)
            import torch
            if self.device == "cpu":
                model = model.to("cpu")

            self._model_cache[model_name] = (tokenizer, model)
            logger.info(f"Model loaded in {time.time()-start:.1f}s")
            return tokenizer, model

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return None

    def _translate_helsinki(self, text: str, src: str, tgt: str) -> Optional[str]:
        """Translate using Helsinki-NLP Marian model."""
        model_key = (src, tgt)
        model_name = HELSINKI_MODELS.get(model_key)

        if not model_name:
            return None

        result = self._load_helsinki_model(model_name)
        if not result:
            return None

        tokenizer, model = result

        try:
            import torch
            # Tokenize
            inputs = tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Generate translation
            with torch.no_grad():
                translated = model.generate(
                    **inputs,
                    num_beams=3,
                    max_length=512,
                    early_stopping=True,
                )

            # Decode
            output = tokenizer.batch_decode(translated, skip_special_tokens=True)
            return output[0] if output else None

        except Exception as e:
            logger.error(f"Helsinki translation error: {e}")
            return None

    def _translate_google(self, text: str, src: str, tgt: str) -> Optional[str]:
        """Translate using Google Translate via deep_translator (free tier)."""
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source=src, target=tgt)
            result = translator.translate(text)
            return result
        except ImportError:
            logger.debug("deep_translator not installed")
        except Exception as e:
            logger.warning(f"Google Translate error: {e}")
        return None

    def _translate_via_english(self, text: str, src: str, tgt: str) -> Optional[str]:
        """
        Pivot translation: src → English → tgt
        Used when direct Indian↔Indian model is unavailable.
        """
        # Step 1: src → English
        en_text = self._translate_direct(text, src, "en")
        if not en_text:
            return None

        # Step 2: English → tgt
        final_text = self._translate_direct(en_text, "en", tgt)
        return final_text

    def _translate_direct(self, text: str, src: str, tgt: str) -> Optional[str]:
        """Try direct translation using available backends."""
        # 1. Helsinki model
        result = self._translate_helsinki(text, src, tgt)
        if result:
            return result

        # 2. Google Translate fallback
        result = self._translate_google(text, src, tgt)
        if result:
            return result

        return None

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> Dict:
        """
        Translate text between any supported Indian language pair.

        Args:
            text: Source text to translate
            source_language: ISO 639-1 code (e.g., "hi") or full name (e.g., "Hindi")
            target_language: ISO 639-1 code or full name

        Returns:
            Dict with keys: translated_text, source_language, target_language,
                           method, processing_time, success, error
        """
        if not text or len(text.strip()) == 0:
            return self._empty_result(source_language, target_language)

        # Normalize language codes
        src = self._normalize_lang(source_language)
        tgt = self._normalize_lang(target_language)

        # Same language: return as-is
        if src == tgt:
            return {
                "translated_text": text,
                "source_language": src,
                "target_language": tgt,
                "method": "passthrough",
                "processing_time": 0,
                "success": True,
                "error": None,
            }

        start_time = time.time()
        method = "unknown"

        try:
            translated = None

            # Try direct translation first
            model_key = (src, tgt)
            if model_key in HELSINKI_MODELS:
                translated = self._translate_helsinki(text, src, tgt)
                method = "helsinki_direct"

            # If direct not available, try via English pivot
            if not translated and src != "en" and tgt != "en":
                translated = self._translate_via_english(text, src, tgt)
                method = "english_pivot"

            # Last resort: Google Translate
            if not translated:
                translated = self._translate_google(text, src, tgt)
                method = "google_translate"

            if translated:
                return {
                    "translated_text": translated,
                    "source_language": src,
                    "target_language": tgt,
                    "method": method,
                    "processing_time": round(time.time() - start_time, 2),
                    "success": True,
                    "error": None,
                }
            else:
                # Could not translate — return original with error
                return {
                    "translated_text": text,
                    "source_language": src,
                    "target_language": tgt,
                    "method": "failed",
                    "processing_time": round(time.time() - start_time, 2),
                    "success": False,
                    "error": f"No translation available for {src}→{tgt}",
                }

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {
                "translated_text": text,
                "source_language": src,
                "target_language": tgt,
                "method": "error",
                "processing_time": round(time.time() - start_time, 2),
                "success": False,
                "error": str(e),
            }

    def _normalize_lang(self, lang: str) -> str:
        """Convert full language name to ISO code if needed."""
        if lang in LANG_NAMES:
            return lang  # Already a code
        # Try reverse lookup
        for code, name in LANG_NAMES.items():
            if name.lower() == lang.lower():
                return code
        return lang.lower()[:2]  # Best guess

    def _empty_result(self, src: str, tgt: str) -> Dict:
        return {
            "translated_text": "",
            "source_language": self._normalize_lang(src),
            "target_language": self._normalize_lang(tgt),
            "method": "empty",
            "processing_time": 0,
            "success": True,
            "error": "Empty input text",
        }

    def get_supported_pairs(self) -> list:
        """Return all directly supported translation pairs."""
        pairs = []
        for (src, tgt) in HELSINKI_MODELS:
            pairs.append({
                "source": LANG_NAMES.get(src, src),
                "target": LANG_NAMES.get(tgt, tgt),
                "source_code": src,
                "target_code": tgt,
            })
        return pairs

    def is_pair_supported(self, src: str, tgt: str) -> bool:
        """Check if a language pair has direct model support."""
        src = self._normalize_lang(src)
        tgt = self._normalize_lang(tgt)
        return (src, tgt) in HELSINKI_MODELS
