"""
emotion/inference.py
----------------------
Emotion detection from transcribed text using a lightweight transformer model.
Uses j-hartmann/emotion-english-distilroberta-base (CPU-friendly).

Detects 7 emotions, mapped to simplified 4-category output with emoji.
"""

import logging
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)

# Emotion display config: (emoji, display_name, hex_color)
EMOTION_CONFIG = {
    "happy":    ("😊", "Happy",    "#FFD700"),
    "sad":      ("😢", "Sad",      "#6495ED"),
    "angry":    ("😠", "Angry",    "#FF4500"),
    "neutral":  ("😐", "Neutral",  "#90EE90"),
    "fear":     ("😨", "Fearful",  "#9370DB"),
    "disgust":  ("🤢", "Disgusted","#8FBC8F"),
    "surprise": ("😲", "Surprised","#FFA500"),
}

# Map from model's raw labels to simplified categories
EMOTION_LABEL_MAP = {
    # j-hartmann model labels
    "joy":      "happy",
    "happiness":"happy",
    "sadness":  "sad",
    "anger":    "angry",
    "neutral":  "neutral",
    "fear":     "fear",
    "disgust":  "disgust",
    "surprise": "surprise",
    # Fallback
    "positive": "happy",
    "negative": "sad",
}


class EmotionDetector:
    """
    Text-based emotion classifier using DistilRoBERTa.
    
    Model: j-hartmann/emotion-english-distilroberta-base
    - 7 emotion classes
    - ~82MB model size
    - ~50ms inference on CPU per sentence
    """

    MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

    def __init__(self):
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load emotion classification pipeline."""
        try:
            from transformers import pipeline
            logger.info(f"Loading emotion model: {self.MODEL_ID}")
            start = time.time()
            self.pipeline = pipeline(
                "text-classification",
                model=self.MODEL_ID,
                top_k=None,          # Return all scores
                device=-1,           # CPU
                truncation=True,
                max_length=512,
            )
            logger.info(f"Emotion model loaded in {time.time()-start:.1f}s")
        except Exception as e:
            logger.warning(f"Could not load emotion model: {e}. Using rule-based fallback.")
            self.pipeline = None

    def _rule_based_emotion(self, text: str) -> str:
        """
        Simple keyword-based emotion detection as fallback.
        Works for English and transliterated Indian language text.
        """
        text_lower = text.lower()

        happy_words = ["happy", "joy", "great", "wonderful", "love", "excellent",
                       "good", "nice", "amazing", "fantastic", "khushi", "anand",
                       "sundar", "bahut accha", "mast", "zabardast"]
        sad_words = ["sad", "cry", "upset", "depressed", "unhappy", "miss",
                     "dukh", "rona", "udaas", "bura", "dard"]
        angry_words = ["angry", "mad", "furious", "hate", "terrible", "awful",
                       "gussa", "krodh", "naraaz", "bura lag"]

        happy_score = sum(1 for w in happy_words if w in text_lower)
        sad_score = sum(1 for w in sad_words if w in text_lower)
        angry_score = sum(1 for w in angry_words if w in text_lower)

        max_score = max(happy_score, sad_score, angry_score)
        if max_score == 0:
            return "neutral"
        if max_score == happy_score:
            return "happy"
        if max_score == angry_score:
            return "angry"
        return "sad"

    def detect(self, text: str) -> Dict:
        """
        Detect emotion from text.

        Args:
            text: Input text (any language, but English gives best accuracy)

        Returns:
            Dict with:
                emotion: primary emotion label
                emoji: display emoji
                display_name: human-readable name
                color: hex color for UI
                confidence: float 0-1
                all_scores: dict of all emotion probabilities
        """
        if not text or len(text.strip()) < 3:
            return self._neutral_result()

        try:
            if self.pipeline is not None:
                results = self.pipeline(text[:512])
                # results is a list of lists: [[{label, score}, ...]]
                if results and len(results) > 0:
                    scores = results[0] if isinstance(results[0], list) else results
                    # Sort by score descending
                    scores_sorted = sorted(scores, key=lambda x: x["score"], reverse=True)
                    top = scores_sorted[0]

                    raw_label = top["label"].lower()
                    emotion = EMOTION_LABEL_MAP.get(raw_label, "neutral")
                    confidence = round(top["score"], 3)

                    # Build all_scores dict
                    all_scores = {
                        EMOTION_LABEL_MAP.get(s["label"].lower(), s["label"].lower()):
                        round(s["score"], 3)
                        for s in scores_sorted
                    }

                    config = EMOTION_CONFIG.get(emotion, EMOTION_CONFIG["neutral"])
                    return {
                        "emotion": emotion,
                        "emoji": config[0],
                        "display_name": config[1],
                        "color": config[2],
                        "confidence": confidence,
                        "all_scores": all_scores,
                        "method": "transformer",
                        "success": True,
                    }

            # Fallback to rule-based
            emotion = self._rule_based_emotion(text)
            config = EMOTION_CONFIG.get(emotion, EMOTION_CONFIG["neutral"])
            return {
                "emotion": emotion,
                "emoji": config[0],
                "display_name": config[1],
                "color": config[2],
                "confidence": 0.6,
                "all_scores": {emotion: 0.6},
                "method": "rule_based",
                "success": True,
            }

        except Exception as e:
            logger.error(f"Emotion detection error: {e}")
            return self._neutral_result(error=str(e))

    def detect_batch(self, texts: List[str]) -> List[Dict]:
        """Detect emotions for multiple texts efficiently."""
        return [self.detect(text) for text in texts]

    def _neutral_result(self, error: Optional[str] = None) -> Dict:
        """Return neutral emotion as default."""
        config = EMOTION_CONFIG["neutral"]
        return {
            "emotion": "neutral",
            "emoji": config[0],
            "display_name": config[1],
            "color": config[2],
            "confidence": 0.5,
            "all_scores": {"neutral": 0.5},
            "method": "default",
            "success": error is None,
            "error": error,
        }

    @staticmethod
    def get_emotion_config() -> Dict:
        """Return full emotion display configuration."""
        return EMOTION_CONFIG
