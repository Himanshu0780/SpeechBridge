#!/bin/bash
# ============================================================
# run.sh — Bharat Voice Translator AI Launcher
# ============================================================
# Usage:
#   chmod +x run.sh
#   ./run.sh              # Start the Streamlit UI
#   ./run.sh demo         # Run text demo (no models needed)
#   ./run.sh check        # Run environment check
#   ./run.sh install      # Install dependencies
#   ./run.sh --help       # Show help

set -e

ORANGE='\033[38;5;208m'
TEAL='\033[38;5;80m'
GREEN='\033[92m'
RED='\033[91m'
GRAY='\033[90m'
BOLD='\033[1m'
RESET='\033[0m'

banner() {
  echo -e "${ORANGE}${BOLD}"
  echo "  ╔══════════════════════════════════════════════════╗"
  echo "  ║    🇮🇳  SpeechBridge                              ║"
  echo "  ║    Real-Time Indian Multilingual Translation     ║"
  echo "  ╚══════════════════════════════════════════════════╝"
  echo -e "${RESET}"
}

help_text() {
  banner
  echo -e "${BOLD}Usage:${RESET}  ./run.sh [command]"
  echo ""
  echo -e "${BOLD}Commands:${RESET}"
  echo "  (none)     Start the Streamlit web UI"
  echo "  demo       Run terminal demo (no mic required)"
  echo "  check      Verify environment setup"
  echo "  install    Install Python dependencies"
  echo "  help       Show this message"
  echo ""
  echo -e "${BOLD}Examples:${RESET}"
  echo "  ./run.sh"
  echo "  ./run.sh demo --lang Tamil"
  echo "  ./run.sh demo --file audio.wav"
  echo "  ./run.sh check"
  echo ""
}

cmd="${1:-ui}"

case "$cmd" in
  --help|-h|help)
    help_text
    ;;

  install)
    banner
    echo -e "${TEAL}Installing dependencies...${RESET}"
    pip install -r requirements.txt
    echo ""
    echo -e "${TEAL}Installing FFmpeg (Ubuntu/Debian)...${RESET}"
    if command -v apt-get &> /dev/null; then
      sudo apt-get install -y ffmpeg
    elif command -v brew &> /dev/null; then
      brew install ffmpeg
    else
      echo -e "${GRAY}Please install FFmpeg manually: https://ffmpeg.org${RESET}"
    fi
    echo -e "${GREEN}✓ Installation complete${RESET}"
    echo -e "  Run: ${ORANGE}./run.sh check${RESET} to verify"
    ;;

  check)
    banner
    python setup_check.py
    ;;

  demo)
    banner
    shift
    python demo.py "$@"
    ;;

  ui|"")
    banner
    echo -e "${TEAL}Starting Bharat Voice Translator AI...${RESET}"
    echo -e "${GRAY}Open your browser at: http://localhost:8501${RESET}"
    echo -e "${GRAY}Press Ctrl+C to stop${RESET}"
    echo ""
    streamlit run app/main.py \
      --server.port 8501 \
      --server.headless false \
      --browser.gatherUsageStats false
    ;;

  *)
    echo -e "${RED}Unknown command: $1${RESET}"
    echo "Run './run.sh --help' for usage."
    exit 1
    ;;
esac
