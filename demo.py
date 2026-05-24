"""
demo.py
--------
Quick demo script to test the pipeline without a microphone.
Generates a synthetic audio signal and runs it through the full pipeline.

Usage:
    python demo.py
    python demo.py --lang Tamil
    python demo.py --file path/to/audio.wav
"""

import argparse
import sys
import os
import numpy as np
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")

# ─── ANSI Colors for terminal output ─────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
ORANGE = "\033[38;5;208m"
TEAL   = "\033[38;5;80m"
GOLD   = "\033[38;5;220m"
RED    = "\033[38;5;196m"
GREEN  = "\033[38;5;82m"
BLUE   = "\033[38;5;75m"
GRAY   = "\033[38;5;245m"


def banner():
    print(f"""
{ORANGE}{BOLD}
╔══════════════════════════════════════════════════════════════╗
║        🇮🇳  SpeechBridge                 —  Demo Mode         ║
║     Real-Time Indian Multilingual Speech Translation         ║
╚══════════════════════════════════════════════════════════════╝
{RESET}""")


def divider(label: str = ""):
    width = 60
    if label:
        side = (width - len(label) - 2) // 2
        print(f"{GRAY}{'─'*side} {label} {'─'*side}{RESET}")
    else:
        print(f"{GRAY}{'─'*width}{RESET}")


def print_result(result):
    """Pretty-print a TranslationResult to the terminal."""
    divider("PIPELINE RESULTS")
    print(f"  {BOLD}📝 Transcript:{RESET}         {result.transcript or '(empty)'}")
    print(f"  {BOLD}🌐 Detected Language:{RESET}  {GOLD}{result.detected_language}{RESET}")
    print(f"  {BOLD}😊 Emotion:{RESET}            {result.emotion_emoji}  {result.emotion_display}  ({result.emotion_confidence*100:.0f}%)")
    print(f"  {BOLD}🔁 Translation ({result.target_language}):{RESET}")
    print(f"     {TEAL}{result.translated_text or '(no translation)'}{RESET}")
    print(f"  {BOLD}⏱  Processing Time:{RESET}    {result.total_processing_time:.2f}s")
    print(f"  {BOLD}🔊 TTS Audio:{RESET}          {'✓ Generated' if result.audio_bytes else '✗ Not generated'}")
    if result.audio_file_path:
        print(f"  {BOLD}💾 Audio saved to:{RESET}    {result.audio_file_path}")
    if not result.success and result.error:
        print(f"  {RED}⚠  Error: {result.error}{RESET}")
    divider()


def generate_test_audio(duration: float = 3.0, sr: int = 16000) -> np.ndarray:
    """
    Generate a synthetic sine wave audio for pipeline testing.
    Real results require actual speech audio.
    """
    t = np.linspace(0, duration, int(sr * duration))
    # Mix of frequencies to simulate formants
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 500 * t) +
        0.1 * np.sin(2 * np.pi * 1000 * t) +
        0.05 * np.random.randn(len(t))
    )
    return audio.astype(np.float32)


def demo_text_pipeline(pipeline, target_lang: str):
    """
    Demo using hardcoded text (bypasses ASR) to test Translation + Emotion + TTS.
    More reliable for verifying non-ASR components.
    """
    divider("TEXT-ONLY PIPELINE DEMO (No ASR needed)")

    test_cases = [
        ("hi", "नमस्ते! आप कैसे हैं? यह एक सुंदर दिन है।", "Hello! How are you? It is a beautiful day."),
        ("en", "The train to Mumbai departs at 3 PM from platform 4.", None),
        ("ta", "வணக்கம்! நான் சென்னையில் இருந்து வருகிறேன்.", None),
        ("bn", "আমি বাংলা ভাষায় কথা বলছি।", None),
    ]

    for src_code, text, expected_en in test_cases:
        print(f"\n  {GRAY}Source ({src_code}):{RESET} {text[:60]}...")

        # Language detection
        start = time.time()
        lang_result = pipeline.lang_detector.detect(text)
        print(f"  {GOLD}Detected:{RESET} {lang_result['language']} ({lang_result['confidence']*100:.0f}%)")

        # Emotion
        emotion_result = pipeline.emotion_detector.detect(text)
        print(f"  {GOLD}Emotion:{RESET}  {emotion_result['emoji']} {emotion_result['display_name']}")

        # Translation
        trans_result = pipeline.translator.translate(text, src_code, target_lang)
        print(f"  {TEAL}→ {target_lang}:{RESET} {trans_result['translated_text'][:80]}")
        print(f"  {GRAY}Time: {time.time()-start:.2f}s | Method: {trans_result['method']}{RESET}")
        divider()


def demo_audio_file(pipeline, file_path: str, target_lang: str):
    """Run the pipeline on an uploaded audio file."""
    divider(f"FILE MODE: {os.path.basename(file_path)}")

    if not os.path.exists(file_path):
        print(f"{RED}File not found: {file_path}{RESET}")
        return

    print(f"  Processing: {file_path}")
    print(f"  Target Language: {target_lang}")

    from noise_reduction.noise_filter import NoiseFilter
    nf = NoiseFilter()
    audio, sr = nf.load_audio(file_path)
    stats = nf.get_audio_stats(audio, sr)
    print(f"  Audio: {stats['duration_seconds']:.1f}s @ {sr}Hz | SNR: {stats['snr_db']}dB")

    result = pipeline.process_audio(
        audio=audio,
        sample_rate=sr,
        target_language=target_lang,
        generate_tts=True,
        input_mode="file",
    )
    print_result(result)


def demo_synthetic_audio(pipeline, target_lang: str):
    """Run pipeline on synthetic test audio (will give poor ASR results but tests the chain)."""
    divider("SYNTHETIC AUDIO DEMO (Tests noise reduction + pipeline chain)")
    print(f"  {GRAY}Note: Synthetic audio won't produce real transcription.{RESET}")
    print(f"  {GRAY}Use a real audio file for meaningful ASR results.{RESET}\n")

    audio = generate_test_audio(duration=4.0)
    print(f"  Generated {len(audio)/16000:.1f}s synthetic audio")

    result = pipeline.process_audio(
        audio=audio,
        sample_rate=16000,
        target_language=target_lang,
        generate_tts=False,  # Skip TTS for synthetic
        input_mode="microphone",
    )
    print_result(result)


def demo_metrics():
    """Demonstrate WER and BLEU metric computation."""
    divider("EVALUATION METRICS DEMO")
    from utils.metrics import compute_wer, compute_bleu, evaluate_translation

    # WER examples
    pairs = [
        ("नमस्ते दुनिया", "नमस्ते दुनिया"),
        ("the quick brown fox", "the quick brown box"),
        ("hello world", "hello"),
    ]
    print(f"\n  {BOLD}Word Error Rate (WER):{RESET}")
    for ref, hyp in pairs:
        wer = compute_wer(ref, hyp)
        print(f"    REF: '{ref}' | HYP: '{hyp}'")
        print(f"    WER: {wer['wer_percent']:.1f}% | Accuracy: {wer['accuracy_percent']:.1f}%\n")

    # BLEU examples
    print(f"  {BOLD}BLEU Score:{RESET}")
    bleu_pairs = [
        ("The cat is on the mat", "The cat is on the mat"),
        ("I love India", "I enjoy India greatly"),
        ("Hello world", "Goodbye world"),
    ]
    for ref, hyp in bleu_pairs:
        bleu = compute_bleu(ref, hyp)
        print(f"    REF: '{ref}'")
        print(f"    HYP: '{hyp}'")
        print(f"    BLEU: {bleu['bleu_percent']:.1f}%\n")


def main():
    banner()

    parser = argparse.ArgumentParser(description="Bharat Voice Translator AI — Demo")
    parser.add_argument("--lang", default="English",
                        choices=["Hindi","English","Tamil","Telugu","Bengali",
                                 "Marathi","Gujarati","Kannada","Malayalam","Punjabi"],
                        help="Target translation language")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to audio file (.wav or .mp3)")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Only run metrics demo (no model loading)")
    parser.add_argument("--text-only", action="store_true",
                        help="Run text pipeline demo only (faster, no ASR)")
    args = parser.parse_args()

    # Metrics-only mode
    if args.metrics_only:
        demo_metrics()
        return

    # Load pipeline
    print(f"{ORANGE}Loading AI pipeline...{RESET}")
    print(f"{GRAY}(First run downloads models — this may take a few minutes){RESET}\n")

    try:
        from pipeline import TranslationPipeline
        pipeline = TranslationPipeline(
            asr_model_size="small",
            enable_noise_reduction=True,
            enable_emotion=True,
            enable_tts=True,
        )
        print(f"{GREEN}✓ Pipeline loaded{RESET}\n")
    except Exception as e:
        print(f"{RED}✗ Pipeline load failed: {e}{RESET}")
        print(f"{GRAY}Try: pip install -r requirements.txt{RESET}")
        sys.exit(1)

    target_lang = args.lang
    print(f"  {BOLD}Target Language:{RESET} {GOLD}{target_lang}{RESET}")

    # Run demos
    if args.file:
        demo_audio_file(pipeline, args.file, target_lang)
    elif args.text_only:
        demo_text_pipeline(pipeline, target_lang)
    else:
        # Run all demos
        demo_text_pipeline(pipeline, target_lang)
        demo_synthetic_audio(pipeline, target_lang)
        demo_metrics()

    print(f"\n{GREEN}{BOLD}Demo complete!{RESET}")
    print(f"{GRAY}Start the full UI with:{RESET}  {ORANGE}streamlit run app/main.py{RESET}\n")


if __name__ == "__main__":
    main()
