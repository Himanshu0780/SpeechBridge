#!/usr/bin/env python3
"""
setup_check.py
---------------
Environment validation script for Bharat Voice Translator AI.
Run this after installing requirements to verify everything is set up correctly.

Usage:
    python setup_check.py
"""

import sys
import importlib
import subprocess
import os

# ─── ANSI Colors ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
GRAY   = "\033[90m"

def check(label: str, fn, required: bool = True):
    """Run a check and print pass/fail."""
    try:
        result = fn()
        status = f"{GREEN}✓{RESET}"
        detail = f" {GRAY}({result}){RESET}" if result and result is not True else ""
        print(f"  {status} {label}{detail}")
        return True
    except Exception as e:
        icon = f"{RED}✗{RESET}" if required else f"{YELLOW}⚠{RESET}"
        tag  = "(REQUIRED)" if required else "(optional)"
        print(f"  {icon} {label} {GRAY}{tag} — {e}{RESET}")
        return False


def check_import(module: str, attr: str = None):
    """Check if a module can be imported."""
    def _check():
        mod = importlib.import_module(module)
        if attr:
            val = getattr(mod, attr, "?")
            return str(val)
        return "ok"
    return _check


def check_ffmpeg():
    """Check FFmpeg availability."""
    def _check():
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            raise RuntimeError("ffmpeg not found")
        version_line = result.stdout.split("\n")[0]
        return version_line.split(" ")[2]
    return _check


def check_microphone():
    """Check if a microphone is accessible."""
    def _check():
        import sounddevice as sd
        devices = sd.query_devices()
        input_devs = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devs:
            raise RuntimeError("No input devices found")
        return f"{len(input_devs)} input device(s)"
    return _check


def check_python_version():
    def _check():
        v = sys.version_info
        if v < (3, 8):
            raise RuntimeError(f"Python 3.8+ required (got {v.major}.{v.minor})")
        return f"{v.major}.{v.minor}.{v.micro}"
    return _check


def check_disk_space():
    def _check():
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        if free_gb < 2.0:
            raise RuntimeError(f"Only {free_gb:.1f}GB free — need 2GB+ for models")
        return f"{free_gb:.1f}GB free"
    return _check


def check_pipeline_import():
    def _check():
        sys.path.insert(0, os.path.dirname(__file__))
        from pipeline import TranslationPipeline
        return "importable"
    return _check


def main():
    print(f"""
{BLUE}{BOLD}╔══════════════════════════════════════════════════════════╗
║  🇮🇳 Bharat Voice Translator AI — Environment Check     ║
╚══════════════════════════════════════════════════════════╝{RESET}
""")

    all_passed = True
    any_failed = []

    # ── System ────────────────────────────────────────────────────────────
    print(f"{BOLD}System{RESET}")
    checks = [
        ("Python 3.8+",         check_python_version(),       True),
        ("Disk space (2GB+)",   check_disk_space(),           True),
        ("FFmpeg",              check_ffmpeg(),               True),
    ]
    for label, fn, required in checks:
        passed = check(label, fn, required)
        if not passed and required:
            any_failed.append(label)

    # ── Core AI Libraries ─────────────────────────────────────────────────
    print(f"\n{BOLD}Core AI Libraries{RESET}")
    ai_checks = [
        ("PyTorch",             check_import("torch", "__version__"),         True),
        ("Transformers",        check_import("transformers", "__version__"),   True),
        ("OpenAI Whisper",      check_import("whisper"),                       True),
        ("torchaudio",          check_import("torchaudio", "__version__"),     True),
    ]
    for label, fn, required in ai_checks:
        passed = check(label, fn, required)
        if not passed and required:
            any_failed.append(label)

    # ── Audio Processing ──────────────────────────────────────────────────
    print(f"\n{BOLD}Audio Processing{RESET}")
    audio_checks = [
        ("librosa",             check_import("librosa", "__version__"),        True),
        ("soundfile",           check_import("soundfile", "__version__"),      True),
        ("noisereduce",         check_import("noisereduce"),                   False),
        ("sounddevice",         check_import("sounddevice", "__version__"),    False),
        ("pydub",               check_import("pydub"),                         False),
    ]
    for label, fn, required in audio_checks:
        check(label, fn, required)

    # ── NLP / Translation ─────────────────────────────────────────────────
    print(f"\n{BOLD}NLP & Translation{RESET}")
    nlp_checks = [
        ("langdetect",          check_import("langdetect"),                    True),
        ("sacrebleu",           check_import("sacrebleu", "__version__"),      False),
        ("sentencepiece",       check_import("sentencepiece"),                 True),
        ("nltk",                check_import("nltk", "__version__"),           False),
        ("deep_translator",     check_import("deep_translator"),               False),
    ]
    for label, fn, required in nlp_checks:
        check(label, fn, required)

    # ── UI & Visualization ────────────────────────────────────────────────
    print(f"\n{BOLD}UI & Visualization{RESET}")
    ui_checks = [
        ("Streamlit",           check_import("streamlit", "__version__"),      True),
        ("Plotly",              check_import("plotly", "__version__"),         True),
        ("gTTS",                check_import("gtts"),                          True),
        ("numpy",               check_import("numpy", "__version__"),          True),
        ("pandas",              check_import("pandas", "__version__"),         False),
    ]
    for label, fn, required in ui_checks:
        check(label, fn, required)

    # ── Project Modules ───────────────────────────────────────────────────
    print(f"\n{BOLD}Project Modules{RESET}")
    proj_checks = [
        ("pipeline.py",               check_pipeline_import(),                True),
        ("noise_reduction module",    check_import("noise_reduction.noise_filter"), True),
        ("asr module",                check_import("asr.speech_to_text"),     True),
        ("language_detection module", check_import("language_detection.detector"), True),
        ("emotion module",            check_import("emotion.inference"),       True),
        ("nmt module",                check_import("nmt.translator"),          True),
        ("tts module",                check_import("tts.text_to_speech"),      True),
        ("app.map_visualization",     check_import("app.map_visualization"),   True),
        ("utils.metrics",             check_import("utils.metrics"),           True),
    ]
    for label, fn, required in proj_checks:
        passed = check(label, fn, required)
        if not passed and required:
            any_failed.append(label)

    # ── Hardware ──────────────────────────────────────────────────────────
    print(f"\n{BOLD}Hardware{RESET}")
    def check_torch_cpu():
        import torch
        return f"{'CUDA' if torch.cuda.is_available() else 'CPU only'}"

    check("PyTorch device",     check_torch_cpu,          True)
    check("Microphone",         check_microphone(),       False)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─'*56}")
    if not any_failed:
        print(f"{GREEN}{BOLD}✓ All required checks passed! You're ready to run.{RESET}")
        print(f"\n  {BLUE}Start the app:{RESET}  streamlit run app/main.py")
        print(f"  {BLUE}Run demo:{RESET}       python demo.py --text-only\n")
    else:
        print(f"{RED}{BOLD}✗ Some required checks failed:{RESET}")
        for f in any_failed:
            print(f"  {RED}•{RESET} {f}")
        print(f"\n{YELLOW}Fix issues then re-run:{RESET}  pip install -r requirements.txt")
        print(f"{YELLOW}Install FFmpeg:{RESET}           sudo apt install ffmpeg  (Linux)")
        print(f"                          brew install ffmpeg        (Mac)\n")


if __name__ == "__main__":
    main()
