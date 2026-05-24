# 🇮🇳 SpeechBridge

> **Real-Time Indian Multilingual Speech Translation System**

A production-grade, CPU-friendly AI system that translates speech between 10 major Indian languages in real-time. Built with a modern SaaS-style UI — feels like Google Translate Live, designed for India.

---

## 🎯 Motivation

India has 22 scheduled languages and hundreds of dialects. Language barriers are a daily challenge in healthcare, education, commerce, and governance. This system bridges those gaps by enabling real-time voice communication across languages like Hindi, Tamil, Telugu, Bengali, and more — running entirely on a laptop CPU, no GPU required.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎤 Live Mic Translation | Continuous real-time speech capture and translation |
| 📁 Audio File Upload | .wav and .mp3 file translation with preview |
| 🗣 Speech Recognition | OpenAI Whisper (small) — 99 languages, Indian accent support |
| 🌐 Language Detection | Script-based + langdetect statistical model |
| 😊 Emotion Detection | DistilRoBERTa emotion classifier (happy/sad/angry/neutral) |
| 🔁 Translation | Helsinki-NLP Marian MT + Google Translate fallback |
| 🔊 TTS Output | gTTS with Indian language support, downloadable MP3 |
| 🗺 India Map | Interactive Plotly map highlighting language regions |
| 📊 Metrics | WER and BLEU score evaluation utilities |
| 🎨 Designer UI | Dark glassmorphism theme, animated cards, gradient buttons |

---

## 🏗 Architecture

```
Speech Input (Mic / File)
        │
        ▼
┌─────────────────┐
│  Noise Filter   │  librosa + noisereduce (spectral reduction, normalization)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Whisper ASR    │  openai-whisper small (CPU, ~2-4s for 5s audio)
└────────┬────────┘
         │
    ┌────┴────────────┐
    │                 │
    ▼                 ▼
┌──────────┐   ┌──────────────┐
│  Lang    │   │   Emotion    │
│ Detector │   │  Classifier  │
│(langdet) │   │(DistilRoBERTa│
└────┬─────┘   └──────┬───────┘
     │                │
     └──────┬──────────┘
            │
            ▼
┌─────────────────────┐
│  Helsinki-NLP NMT   │  MarianMT models per language pair
│  (+ English pivot   │  src→en→tgt when direct unavailable
│   + Google fallback)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────┐
│   gTTS Engine   │  Translated speech output (MP3)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit UI   │  Dark glassmorphism dashboard with India map
└─────────────────┘
```

---

## 🇮🇳 Supported Languages

| Language | Script | Region | Speakers |
|---|---|---|---|
| Hindi | Devanagari | North & Central India | ~600M |
| English | Latin | Pan-India | ~125M |
| Bengali | Bengali | Eastern India | ~100M |
| Telugu | Telugu | South India | ~85M |
| Marathi | Devanagari | Western India | ~83M |
| Tamil | Tamil | South India | ~75M |
| Gujarati | Gujarati | Western India | ~57M |
| Kannada | Kannada | South India | ~45M |
| Malayalam | Malayalam | South India | ~38M |
| Punjabi | Gurmukhi | North India | ~33M |

---

## 📁 Project Structure

```
speech_translation_system/
├── app/
│   ├── main.py              ← Streamlit UI (run this)
│   ├── audio_stream.py      ← Live microphone capture
│   └── map_visualization.py ← India language map
├── asr/
│   └── speech_to_text.py    ← Whisper ASR engine
├── nmt/
│   └── translator.py        ← Helsinki-NLP translation
├── emotion/
│   └── inference.py         ← DistilRoBERTa emotion detection
├── language_detection/
│   └── detector.py          ← Script + langdetect detection
├── noise_reduction/
│   └── noise_filter.py      ← librosa noise reduction
├── tts/
│   └── text_to_speech.py    ← gTTS synthesis
├── utils/
│   └── metrics.py           ← WER and BLEU evaluation
├── pipeline.py              ← Main orchestration pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone / Download the project
```bash
cd speech_translation_system
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** First run downloads Whisper small (~500MB) and Helsinki-NLP models (~300MB each). Ensure you have 2-3GB free disk space and a stable internet connection for the first launch.

### 4. Install FFmpeg (required by Whisper)
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# Mac (Homebrew)
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/download.html
```

### 5. Run the application
```bash
streamlit run app/main.py
```

The app opens at `http://localhost:8501`

---

## 🎤 Live Microphone Translation

1. Open the app at `http://localhost:8501`
2. Select **Target Language** (e.g., Tamil)
3. Optionally set a **Source Language Hint** for better accuracy
4. Click **▶ Start Live** in the Microphone tab
5. Speak naturally — the system processes 2-second audio chunks
6. Watch the Transcript, Language, Emotion, and Translation cards update in real-time
7. Click **■ Stop** when done

**Tips:**
- Speak 20-30cm from microphone
- First chunk takes 3-5s to process (model warm-up)
- Works best in quiet environments

---

## 📁 Audio File Upload Translation

1. Click the **📁 Audio File Upload** tab
2. Drag & drop or select a `.wav` or `.mp3` file
3. Preview the audio in the built-in player
4. Click **🚀 Translate File**
5. Results appear in the right panel (Transcript, Language, Emotion, Translation)
6. Download translated speech as MP3

---

## 🗺 India Language Map

The interactive map at the bottom highlights:
- 🟠 **Orange bubbles** — regions where detected source language is spoken
- 🔵 **Teal bubbles** — regions where target language is spoken  
- 🟣 **Purple bubbles** — states where both languages are spoken

Click on state bubbles for details.

---

## 🔁 Real-Time Pipeline Flow

```
Mic Chunks (2s) → Noise Reduce → Whisper ASR → Lang Detect
                                                      │
                               Emotion Detect ←───────┤
                                      │                │
                               Translation ←───────────┘
                                      │
                                  gTTS TTS
                                      │
                              UI Update (2s cycle)
```

---

## 📊 Evaluation Metrics

```python
from utils.metrics import compute_wer, compute_bleu

# Word Error Rate for ASR evaluation
wer = compute_wer(reference="नमस्ते दुनिया", hypothesis="नमस्ते दुनिया")
print(f"WER: {wer['wer_percent']}%")

# BLEU score for translation quality
bleu = compute_bleu(reference="Hello world", hypothesis="Hello world")
print(f"BLEU: {bleu['bleu_percent']}")
```

---

## 🔮 Future Improvements

- [ ] **IndicTrans2** integration for higher quality Indian language pairs
- [ ] **WebRTC** for lower latency browser-based audio capture
- [ ] **Speaker diarization** for multi-speaker conversations
- [ ] **Offline mode** — fully local models without internet
- [ ] **Mobile app** via Streamlit Community Cloud deployment
- [ ] **Regional dialect support** (Bhojpuri, Maithili, etc.)
- [ ] **Live captioning overlay** mode for presentations
- [ ] **Batch file processing** for long recordings
- [ ] **Custom vocabulary** support for domain-specific terms (medical, legal)

---

## 🛠 Technology Stack

| Component | Technology |
|---|---|
| ASR | OpenAI Whisper (small) |
| Translation | Helsinki-NLP Marian MT + deep_translator |
| Emotion | j-hartmann/emotion-english-distilroberta-base |
| Language Detection | langdetect + Unicode script analysis |
| Noise Reduction | noisereduce + librosa |
| TTS | Google Text-to-Speech (gTTS) |
| UI | Streamlit + Custom CSS (glassmorphism) |
| Maps | Plotly Scattergeo |
| Audio Streaming | sounddevice |

---

## 📄 License

MIT License — Free for educational and research use.

---

*Built with ❤️ for India — Ek Bharat, Shreshtha Bharat*
