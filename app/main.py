"""
app/main.py  (run with: streamlit run app/main.py)
---------------------------------------------------
SpeechBridge AI — Main Streamlit Application

Real-Time Indian Multilingual Speech Translation System
Supports live microphone input and audio file upload.
"""

import streamlit as st
import numpy as np
import time
import io
import base64
import threading
import sys
import os
import logging

# ─── Path setup ──────────────────────────────────────────────────────────────
# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─── Global result buffer ────────────────────────────────────────────────────
# Module-level globals survive Streamlit reruns without serialization issues.
# queue.Queue stored in session_state gets silently recreated each rerun
# (Streamlit can't serialize it), so we use a plain list + Lock instead.
_result_lock    = threading.Lock()
_result_buffer  = []    # list[TranslationResult] — background thread appends
_pipeline_store = None  # plain Python ref set once on load; read-only in thread

# ─── Page Config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="SpeechBridge",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

/* ── Global Reset & Theme ───────────────────────── */
:root {
    --bg-deep:     #050818;
    --bg-card:     rgba(13, 20, 50, 0.85);
    --bg-glass:    rgba(20, 35, 80, 0.6);
    --accent-1:    #FF6B35;
    --accent-2:    #4ECDC4;
    --accent-3:    #FFD700;
    --accent-glow: rgba(255, 107, 53, 0.3);
    --text-primary:#E8EAF6;
    --text-muted:  #7986CB;
    --border:      rgba(100, 130, 220, 0.2);
    --border-glow: rgba(255, 107, 53, 0.4);
    --font-main:   'Exo 2', sans-serif;
    --font-mono:   'Space Mono', monospace;
}

html, body, [class*="css"] {
    background-color: var(--bg-deep) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-main) !important;
}

/* Animated background mesh */
.stApp {
    background:
        radial-gradient(ellipse at 20% 20%, rgba(255,107,53,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(78,205,196,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(63,81,181,0.04) 0%, transparent 70%),
        linear-gradient(180deg, #050818 0%, #080d24 100%) !important;
    min-height: 100vh;
}

/* ── Header ─────────────────────────────────────── */
.hero-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    position: relative;
}

.hero-title {
    font-family: var(--font-main);
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #FF6B35 0%, #FFD700 40%, #4ECDC4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.4rem;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 300;
}

.hero-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-1), var(--accent-2), transparent);
    margin: 1.5rem auto;
    max-width: 600px;
    opacity: 0.6;
}

/* ── Glass Card ──────────────────────────────────── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow:
        0 4px 30px rgba(0,0,0,0.4),
        inset 0 1px 0 rgba(255,255,255,0.05);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(100,130,220,0.35);
    box-shadow: 0 8px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.07);
}

.card-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-content {
    font-size: 1.1rem;
    line-height: 1.7;
    color: var(--text-primary);
    font-weight: 400;
    min-height: 2.5rem;
}

/* ── Status Badges ───────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.85rem;
    border-radius: 100px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.badge-active {
    background: rgba(78,205,196,0.15);
    border: 1px solid rgba(78,205,196,0.4);
    color: #4ECDC4;
}

.badge-inactive {
    background: rgba(150,150,170,0.1);
    border: 1px solid rgba(150,150,170,0.25);
    color: #8890A0;
}

.badge-processing {
    background: rgba(255,107,53,0.15);
    border: 1px solid rgba(255,107,53,0.4);
    color: #FF6B35;
}

/* ── Gradient Buttons ────────────────────────────── */
.stButton > button {
    font-family: var(--font-main) !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-size: 0.82rem !important;
    padding: 0.7rem 1.5rem !important;
    border-radius: 10px !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #FF6B35, #E85D04) !important;
    color: white !important;
    box-shadow: 0 4px 20px rgba(255,107,53,0.35) !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(255,107,53,0.5) !important;
}

.stButton > button[kind="secondary"] {
    background: rgba(78,205,196,0.12) !important;
    color: #4ECDC4 !important;
    border: 1px solid rgba(78,205,196,0.35) !important;
}

.stButton > button[kind="secondary"]:hover {
    background: rgba(78,205,196,0.2) !important;
    transform: translateY(-1px) !important;
}

/* ── Selectbox & Inputs ──────────────────────────── */
.stSelectbox > div > div {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

.stFileUploader {
    background: var(--bg-glass) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s ease !important;
}

.stFileUploader:hover {
    border-color: rgba(255,107,53,0.4) !important;
}

/* ── Waveform ─────────────────────────────────────── */
.waveform-container {
    background: rgba(5, 8, 24, 0.8);
    border-radius: 12px;
    border: 1px solid var(--border);
    padding: 0.8rem;
    height: 70px;
    overflow: hidden;
    position: relative;
}

.waveform-bars {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 3px;
    height: 100%;
}

.waveform-bar {
    width: 4px;
    background: linear-gradient(to top, #FF6B35, #FFD700);
    border-radius: 2px;
    transition: height 0.1s ease;
    min-height: 4px;
}

/* ── Mic Pulse ────────────────────────────────────── */
@keyframes mic-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.08); }
}

@keyframes spin-glow {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.mic-active {
    animation: mic-pulse 1.5s ease-in-out infinite;
    color: #FF6B35 !important;
}

/* ── Emotion Badge ────────────────────────────────── */
.emotion-badge {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 1rem;
    border-radius: 12px;
    font-size: 1rem;
    font-weight: 600;
}

/* ── Language Pill ─────────────────────────────────── */
.lang-pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 100px;
    background: rgba(255,215,0,0.12);
    border: 1px solid rgba(255,215,0,0.3);
    color: #FFD700;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* ── Typing animation ─────────────────────────────── */
@keyframes typing-cursor {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}

.typing-cursor::after {
    content: '▋';
    animation: typing-cursor 1s infinite;
    color: var(--accent-1);
    margin-left: 2px;
}

/* ── Processing spinner ───────────────────────────── */
@keyframes process-spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.processing-indicator {
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 2px solid rgba(255,107,53,0.2);
    border-top-color: #FF6B35;
    border-radius: 50%;
    animation: process-spin 0.8s linear infinite;
    vertical-align: middle;
    margin-right: 6px;
}

/* ── Metrics Row ──────────────────────────────────── */
.metric-mini {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    text-align: center;
}

.metric-mini .value {
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--accent-1);
    font-family: var(--font-mono);
    line-height: 1;
}

.metric-mini .label {
    font-size: 0.68rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 3px;
}

/* ── Section Labels ───────────────────────────────── */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

/* ── Hide Streamlit Defaults ──────────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.stDeployButton { visibility: hidden; }

/* ── Tabs ─────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-glass) !important;
    border-radius: 10px !important;
    padding: 0.25rem !important;
    border: 1px solid var(--border) !important;
    gap: 0.25rem !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(255,107,53,0.2), rgba(255,215,0,0.1)) !important;
    color: #FF6B35 !important;
    border: 1px solid rgba(255,107,53,0.3) !important;
}

/* ── Progress Bars ────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #FF6B35, #FFD700) !important;
}

/* ── Scrollbar ────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(5,8,24,0.5); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #FF6B35, #4ECDC4);
    border-radius: 3px;
}

/* ── Audio player ─────────────────────────────────── */
audio {
    width: 100%;
    filter: invert(0.85) hue-rotate(180deg);
    border-radius: 8px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─── Supported Languages ───────────────────────────────────────────────────
SUPPORTED_LANGUAGES = [
    "Hindi", "English", "Tamil", "Telugu", "Bengali",
    "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi"
]

LANG_FLAGS = {
    "Hindi": "🔵", "English": "🌐", "Tamil": "🟠",
    "Telugu": "🟡", "Bengali": "🟢", "Marathi": "🔴",
    "Gujarati": "🟣", "Kannada": "⚫", "Malayalam": "🟤", "Punjabi": "🔶"
}

LANG_CODES = {
    "Hindi": "hi", "English": "en", "Tamil": "ta",
    "Telugu": "te", "Bengali": "bn", "Marathi": "mr",
    "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml", "Punjabi": "pa"
}

# ─── Session State Init ────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "pipeline": None,
        "is_mic_active": False,
        "mic_stream": None,
        "transcript": "",
        "translated_text": "",
        "detected_language": "—",
        "emotion": "neutral",
        "emotion_emoji": "😐",
        "emotion_color": "#90EE90",
        "emotion_display": "Neutral",
        "target_language": "English",
        "audio_bytes": None,
        "processing": False,
        "last_result": None,
        "processing_time": 0.0,
        "waveform_data": [0.2] * 30,
        "chunk_count": 0,
        "history": [],
        "source_lang_hint": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ─── Mic Stop Callback ─────────────────────────────────────────────────────
def stop_mic_callback():
    """
    Called immediately when Stop button is clicked via on_click.
    Fires BEFORE st.rerun(), so it is never swallowed by the fast rerun loop.
    """
    st.session_state.is_mic_active = False
    if st.session_state.get("mic_stream") is not None:
        try:
            st.session_state.mic_stream.stop()
        except Exception:
            pass
        st.session_state.mic_stream = None

# ─── Pipeline Loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the translation pipeline (cached across sessions)."""
    global _pipeline_store
    try:
        from pipeline import TranslationPipeline
        p = TranslationPipeline(
            asr_model_size="small",
            enable_noise_reduction=True,
            enable_emotion=True,
            enable_tts=True,
        )
        # Store as plain module-level global so background threads can access
        # it without going through st.session_state or @st.cache_resource wrappers.
        _pipeline_store = p
        return p
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None

# ─── Helper Functions ──────────────────────────────────────────────────────
def render_waveform(levels: list, active: bool = False):
    """Render animated waveform bars."""
    max_levels = 32
    if len(levels) < max_levels:
        levels = [0.15] * (max_levels - len(levels)) + levels
    levels = levels[-max_levels:]

    bars = ""
    for i, lvl in enumerate(levels):
        h = max(4, min(55, int(lvl * 60)))
        opacity = 0.4 + lvl * 0.6
        if active:
            color = f"linear-gradient(to top, #FF6B35, #FFD700)"
        else:
            color = f"rgba(100,120,180,0.4)"
        bars += f'<div class="waveform-bar" style="height:{h}px;background:{color};opacity:{opacity}"></div>'

    st.markdown(f"""
    <div class="waveform-container">
        <div class="waveform-bars">{bars}</div>
    </div>
    """, unsafe_allow_html=True)


def render_emotion_badge(emotion: str, emoji: str, display_name: str, color: str, confidence: float):
    """Render colored emotion badge."""
    bg_color = f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.15,)}"
    border_color = color + "60"
    st.markdown(f"""
    <div class="emotion-badge" style="background:{bg_color};border:1px solid {border_color};color:{color}">
        <span style="font-size:1.6rem">{emoji}</span>
        <div>
            <div style="font-weight:700;font-size:1rem">{display_name}</div>
            <div style="font-size:0.72rem;opacity:0.7;font-family:'Space Mono'">{confidence*100:.0f}% confidence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_output_card(icon: str, label: str, content: str, accent_color: str = "#FF6B35", typing: bool = False):
    """Render a glassmorphism output card."""
    cursor_class = "typing-cursor" if typing and content else ""
    st.markdown(f"""
    <div class="glass-card">
        <div class="card-title">
            <span style="font-size:1rem">{icon}</span>
            <span>{label}</span>
        </div>
        <div class="card-content {cursor_class}" style="color:{accent_color if content else '#4A5080'}">
            {content if content else '<i style="opacity:0.4">Waiting for audio input...</i>'}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_row(metrics: dict):
    """Render a row of mini metrics."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.markdown(f"""
            <div class="metric-mini">
                <div class="value">{value}</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# ─── Hero Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🇮🇳 SpeechBridge</div>
    <div class="hero-subtitle">Real-Time Indian Multilingual Speech Translation System</div>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ─── Pipeline Loading Banner ───────────────────────────────────────────────
with st.expander("⚙️ System Status", expanded=False):
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown("""
        <div style="font-size:0.85rem;color:#7986CB;line-height:1.8">
        <b style="color:#FF6B35">First launch:</b> Models will download automatically (~500MB for Whisper Small).<br>
        Supported: Hindi · English · Tamil · Telugu · Bengali · Marathi · Gujarati · Kannada · Malayalam · Punjabi
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        if st.button("Load Models", type="primary", key="load_btn"):
            with st.spinner("Loading AI models..."):
                st.session_state.pipeline = load_pipeline()
            if st.session_state.pipeline:
                st.success("✓ Pipeline ready!")
            else:
                st.error("Pipeline load failed")

# Auto-load pipeline
if st.session_state.pipeline is None:
    st.session_state.pipeline = load_pipeline()  # also sets _pipeline_store

# Keep _pipeline_store in sync in case of hot-reload / new session
if _pipeline_store is None and st.session_state.pipeline is not None:
    _pipeline_store = st.session_state.pipeline

# ─── Main Layout ───────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.4], gap="large")

# ══════════════════════════════════════════════════════════════════════════
# LEFT PANEL — Input Controls
# ══════════════════════════════════════════════════════════════════════════
with left_col:

    # ── Language Settings ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">🌐 Translation Settings</div>', unsafe_allow_html=True)

    target_lang = st.selectbox(
        "Target Language",
        options=SUPPORTED_LANGUAGES,
        index=SUPPORTED_LANGUAGES.index(st.session_state.target_language),
        key="target_lang_select",
        label_visibility="collapsed",
        format_func=lambda x: f"{LANG_FLAGS.get(x, '🌐')} {x}",
    )
    st.session_state.target_language = target_lang

    source_hint = st.selectbox(
        "Source Language Hint (optional)",
        options=["Auto Detect"] + SUPPORTED_LANGUAGES,
        key="source_hint_select",
        label_visibility="visible",
        format_func=lambda x: f"{LANG_FLAGS.get(x, '🔍')} {x}" if x != "Auto Detect" else "🔍 Auto Detect",
    )
    st.session_state.source_lang_hint = None if source_hint == "Auto Detect" else LANG_CODES.get(source_hint)

    st.markdown("---")

    # ── Tabs: Mic vs File ─────────────────────────────────────────────────
    mic_tab, file_tab = st.tabs(["🎤 Live Microphone", "📁 Audio File Upload"])

    # ── MICROPHONE TAB ────────────────────────────────────────────────────
    with mic_tab:
        st.markdown('<div class="section-label">🎙 Microphone Input</div>', unsafe_allow_html=True)

        # Waveform display
        waveform_placeholder = st.empty()
        waveform_placeholder.markdown("""
        <div class="waveform-container">
            <div class="waveform-bars">
                <div style="color:#4A5080;font-size:0.8rem;font-family:'Space Mono'">
                    🎤 Waveform activates when mic is live
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Status indicator
        status_placeholder = st.empty()
        if st.session_state.is_mic_active:
            status_placeholder.markdown("""
            <div style="text-align:center;margin:0.5rem 0">
                <span class="badge badge-active">
                    <span class="mic-active">●</span> LIVE — Listening
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            status_placeholder.markdown("""
            <div style="text-align:center;margin:0.5rem 0">
                <span class="badge badge-inactive">● Microphone Idle</span>
            </div>""", unsafe_allow_html=True)

        btn_col1, btn_col2 = st.columns(2)

        with btn_col1:
            start_mic = st.button(
                "▶ Start Live",
                type="primary",
                key="start_mic_btn",
                disabled=st.session_state.is_mic_active,
            )

        with btn_col2:
            stop_mic = st.button(
                "■ Stop",
                type="secondary",
                key="stop_mic_btn",
                disabled=not st.session_state.is_mic_active,
                on_click=stop_mic_callback,   # fires before rerun — never swallowed
            )

        # Start microphone
        if start_mic and not st.session_state.is_mic_active:
            try:
                from app.audio_stream import MicrophoneStream

                # Resolve pipeline — prefer module-level store (no Streamlit wrapper),
                # fall back to session_state if store not yet populated.
                # NOTE: no "global" keyword needed — _pipeline_store is already a
                # module-level variable and this code runs at module scope.
                if _pipeline_store is None:
                    _pipeline_store = st.session_state.pipeline
                _pl = _pipeline_store

                if _pl is None:
                    st.error("Pipeline not loaded. Click 'Load Models' first.")
                else:
                    # Snapshot settings as plain Python strings — the thread
                    # must never touch st.session_state or any Streamlit API.
                    _target_lang = str(st.session_state.target_language)
                    _source_hint = st.session_state.source_lang_hint  # str or None

                    def process_chunk(chunk):
                        """
                        Runs in the AudioProcessing background thread.
                        Only uses: _pl (plain obj), _target_lang (str),
                        _source_hint (str|None), _result_buffer (list), _result_lock.
                        No Streamlit APIs touched — safe from any thread.
                        """
                        try:
                            result = _pl.process_audio(
                                audio=chunk.audio,
                                sample_rate=16000,
                                target_language=_target_lang,
                                source_language_hint=_source_hint,
                                generate_tts=True,
                                input_mode="microphone",
                            )
                            if result.success and result.transcript:
                                with _result_lock:
                                    _result_buffer.append(result)
                                    # Cap buffer — keep only 3 most recent
                                    if len(_result_buffer) > 3:
                                        del _result_buffer[:-3]
                        except Exception as e:
                            # Use stdlib logger only — no st.* calls here
                            logger.error(f"Chunk processing error: {e}")

                    stream = MicrophoneStream()
                    stream.start(callback=process_chunk)
                    st.session_state.mic_stream = stream
                    st.session_state.is_mic_active = True
                    st.rerun()

            except ImportError as e:
                st.error(f"sounddevice not installed: {e}\nRun: pip install sounddevice")
            except Exception as e:
                st.error(f"Microphone error: {e}\nCheck microphone permissions.")

        # Stop is handled by stop_mic_callback (on_click above) — no handler needed here

        # Live waveform update
        if st.session_state.is_mic_active and st.session_state.mic_stream:
            levels = st.session_state.mic_stream.get_level_history(30)
            with waveform_placeholder:
                render_waveform(levels, active=True)

        # Tips
        with st.expander("💡 Microphone Tips", expanded=False):
            st.markdown("""
            <div style="font-size:0.82rem;color:#7986CB;line-height:1.8">
            • Speak clearly, 20–30cm from microphone<br>
            • Processing happens in 2-second chunks<br>
            • Reduce background noise for best accuracy<br>
            • First chunk may take 3-5s to process
            </div>
            """, unsafe_allow_html=True)

    # ── FILE UPLOAD TAB ───────────────────────────────────────────────────
    with file_tab:
        st.markdown('<div class="section-label">📂 Audio File</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload .wav or .mp3 file",
            type=["wav", "mp3"],
            key="audio_uploader",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            # Audio preview
            st.markdown('<div style="font-size:0.75rem;color:#7986CB;margin-bottom:0.3rem">▶ Preview</div>',
                       unsafe_allow_html=True)
            st.audio(uploaded_file, format=uploaded_file.type)

            file_info_col1, file_info_col2 = st.columns(2)
            with file_info_col1:
                st.markdown(f"""
                <div class="metric-mini">
                    <div class="value" style="font-size:0.9rem">{uploaded_file.name[:18]}</div>
                    <div class="label">Filename</div>
                </div>
                """, unsafe_allow_html=True)
            with file_info_col2:
                size_kb = len(uploaded_file.getvalue()) / 1024
                st.markdown(f"""
                <div class="metric-mini">
                    <div class="value" style="font-size:1rem">{size_kb:.0f}KB</div>
                    <div class="label">File Size</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin:0.75rem 0'></div>", unsafe_allow_html=True)

            if st.button("🚀 Translate File", type="primary", key="translate_file_btn"):
                pipeline = st.session_state.pipeline
                if pipeline is None:
                    st.error("Pipeline not loaded. Click 'Load Models' first.")
                else:
                    with st.spinner("🔄 Processing audio..."):
                        progress_bar = st.progress(0)

                        # Save uploaded file temporarily
                        import tempfile
                        suffix = ".wav" if uploaded_file.type == "audio/wav" else ".mp3"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        try:
                            progress_bar.progress(25)
                            # Load audio
                            from noise_reduction.noise_filter import NoiseFilter
                            nf = NoiseFilter()
                            audio, sr = nf.load_audio(tmp_path)
                            progress_bar.progress(50)

                            # Process through pipeline
                            result = pipeline.process_audio(
                                audio=audio,
                                sample_rate=sr,
                                target_language=st.session_state.target_language,
                                source_language_hint=st.session_state.source_lang_hint,
                                generate_tts=True,
                                input_mode="file",
                            )
                            progress_bar.progress(90)

                            if result.success:
                                st.session_state.transcript = result.transcript
                                st.session_state.translated_text = result.translated_text
                                st.session_state.detected_language = result.detected_language
                                st.session_state.emotion = result.emotion
                                st.session_state.emotion_emoji = result.emotion_emoji
                                st.session_state.emotion_color = result.emotion_color
                                st.session_state.emotion_display = result.emotion_display
                                st.session_state.audio_bytes = result.audio_bytes
                                st.session_state.processing_time = result.total_processing_time
                                st.session_state.last_result = result
                                progress_bar.progress(100)
                                st.success(f"✓ Processed in {result.total_processing_time:.1f}s")
                            else:
                                st.error(f"Processing failed: {result.error}")

                        finally:
                            os.unlink(tmp_path)

# ══════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Live Output
# ══════════════════════════════════════════════════════════════════════════
with right_col:

    # ── Metrics Row ───────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📊 Live Metrics</div>', unsafe_allow_html=True)
    render_metric_row({
        "Chunks": str(st.session_state.chunk_count),
        "Lang": st.session_state.detected_language[:8],
        "Time": f"{st.session_state.processing_time:.1f}s",
        "Status": "LIVE" if st.session_state.is_mic_active else "IDLE",
    })

    st.markdown("<div style='margin:0.8rem 0'></div>", unsafe_allow_html=True)

    # ── Transcript Card ───────────────────────────────────────────────────
    render_output_card(
        "📝", "TRANSCRIPT",
        st.session_state.transcript,
        accent_color="#E8EAF6",
        typing=st.session_state.is_mic_active,
    )

    # ── Language + Emotion Row ────────────────────────────────────────────
    lang_col, emo_col = st.columns([1, 1])

    with lang_col:
        st.markdown(f"""
        <div class="glass-card">
            <div class="card-title">🌐 DETECTED LANGUAGE</div>
            <div style="margin-top:0.5rem">
                <span class="lang-pill">{st.session_state.detected_language}</span>
            </div>
            <div style="font-size:0.72rem;color:#4A5080;margin-top:0.5rem;font-family:'Space Mono'">
                → {st.session_state.target_language}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with emo_col:
        st.markdown('<div class="glass-card"><div class="card-title">😊 EMOTION</div>', unsafe_allow_html=True)
        render_emotion_badge(
            st.session_state.emotion,
            st.session_state.emotion_emoji,
            st.session_state.emotion_display,
            st.session_state.emotion_color,
            0.75,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Translation Card ──────────────────────────────────────────────────
    render_output_card(
        "🔁", f"TRANSLATION → {st.session_state.target_language.upper()}",
        st.session_state.translated_text,
        accent_color="#4ECDC4",
        typing=st.session_state.is_mic_active and bool(st.session_state.transcript),
    )

    # ── Audio Output ──────────────────────────────────────────────────────
    if st.session_state.audio_bytes:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">🔊 TRANSLATED AUDIO OUTPUT</div>
        </div>
        """, unsafe_allow_html=True)
        st.audio(st.session_state.audio_bytes, format="audio/mp3")

        st.download_button(
            label="⬇ Download Translated Audio",
            data=st.session_state.audio_bytes,
            file_name=f"translation_{st.session_state.target_language.lower()}.mp3",
            mime="audio/mp3",
            key="download_audio_btn",
        )

# ══════════════════════════════════════════════════════════════════════════
# INDIA MAP VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center;margin:1rem 0 0.5rem">
    <div class="hero-subtitle" style="font-size:0.8rem">🗺 INDIA LANGUAGE MAP</div>
</div>
""", unsafe_allow_html=True)

map_col1, map_col2 = st.columns([3, 1])

with map_col1:
    try:
        from app.map_visualization import create_india_language_map
        fig = create_india_language_map(
            source_language=(
                st.session_state.detected_language
                if st.session_state.detected_language not in ("—", "Unknown", "")
                else None
            ),
            target_language=st.session_state.target_language,
            title="🇮🇳 Indian Language Regions",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False, "scrollZoom": False})
    except ImportError:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:3rem">
            <div style="font-size:2rem">🗺</div>
            <div style="color:#7986CB;margin-top:1rem">plotly not installed</div>
            <div style="font-size:0.75rem;color:#4A5080;margin-top:0.5rem">
                Run: <code>pip install plotly&gt;=5.17.0</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        # Show the real error so it's diagnosable — not a silent fallback
        st.markdown(f"""
        <div class="glass-card" style="border-color:rgba(255,80,80,0.3);padding:1.5rem">
            <div style="font-size:0.75rem;font-weight:700;color:#FF6B6B;
                        letter-spacing:2px;text-transform:uppercase;margin-bottom:0.5rem">
                ⚠ Map Error
            </div>
            <div style="font-family:'Space Mono',monospace;font-size:0.75rem;
                        color:#FF9090;word-break:break-all">{str(e)}</div>
        </div>
        """, unsafe_allow_html=True)

with map_col2:
    # Language region info cards
    detected = st.session_state.detected_language
    if detected not in ("—", "Unknown"):
        try:
            from app.map_visualization import get_language_region_info
            info = get_language_region_info(detected)
            st.markdown(f"""
            <div class="glass-card" style="border-color:rgba(255,107,53,0.3)">
                <div class="card-title">🎤 SOURCE REGION</div>
                <div style="font-size:1rem;font-weight:700;color:#FF6B35">{detected}</div>
                <div style="font-size:0.78rem;color:#7986CB;margin-top:0.4rem">{info['region']}</div>
                <div style="font-size:0.72rem;color:#4A5080;margin-top:0.25rem">{info['speakers']}</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass

    target = st.session_state.target_language
    try:
        from app.map_visualization import get_language_region_info
        tinfo = get_language_region_info(target)
        st.markdown(f"""
        <div class="glass-card" style="border-color:rgba(78,205,196,0.3)">
            <div class="card-title">🔁 TARGET REGION</div>
            <div style="font-size:1rem;font-weight:700;color:#4ECDC4">{target}</div>
            <div style="font-size:0.78rem;color:#7986CB;margin-top:0.4rem">{tinfo['region']}</div>
            <div style="font-size:0.72rem;color:#4A5080;margin-top:0.25rem">{tinfo['speakers']}</div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════
# TRANSLATION HISTORY
# ══════════════════════════════════════════════════════════════════════════
if st.session_state.history:
    st.markdown("---")
    with st.expander("📜 Translation History", expanded=False):
        for item in reversed(st.session_state.history[-10:]):
            st.markdown(f"""
            <div style="
                background:rgba(13,20,50,0.6);
                border:1px solid rgba(100,130,220,0.15);
                border-radius:10px;
                padding:0.75rem 1rem;
                margin-bottom:0.5rem;
                font-size:0.82rem;
            ">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem">
                    <span style="color:#7986CB;font-family:'Space Mono'">{item['time']}</span>
                    <span>{item['emotion']} <span class="lang-pill" style="font-size:0.72rem">{item['source']}</span></span>
                </div>
                <div style="color:#C0C8E8">📝 {item['transcript']}</div>
                <div style="color:#4ECDC4;margin-top:0.25rem">🔁 {item['translation']}</div>
            </div>
            """, unsafe_allow_html=True)

    if st.button("🗑 Clear History", key="clear_history_btn"):
        st.session_state.history = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;margin:3rem 0 1rem;padding-top:1.5rem;
    border-top:1px solid rgba(100,130,220,0.15)">
    <div style="font-size:0.72rem;color:#3D4A6B;letter-spacing:2px;text-transform:uppercase">
        🇮🇳 SpeechBridge &nbsp;·&nbsp;
        Powered by Whisper + Helsinki-NLP + gTTS &nbsp;·&nbsp;
        Built for India
    </div>
</div>
""", unsafe_allow_html=True)

# ── Drain global result buffer (always runs on main thread every rerun) ────
# The background audio thread appends to _result_buffer.
# We swap it out under a lock, then safely update session_state here.
with _result_lock:
    _pending  = _result_buffer[:]
    _result_buffer.clear()

for result in _pending:
    st.session_state.transcript        = result.transcript
    st.session_state.translated_text   = result.translated_text
    st.session_state.detected_language = result.detected_language
    st.session_state.emotion           = result.emotion
    st.session_state.emotion_emoji     = result.emotion_emoji
    st.session_state.emotion_color     = result.emotion_color
    st.session_state.emotion_display   = result.emotion_display
    st.session_state.audio_bytes       = result.audio_bytes
    st.session_state.processing_time   = result.total_processing_time
    st.session_state.chunk_count      += 1
    st.session_state.last_result       = result
    st.session_state.history.append({
        "time":        time.strftime("%H:%M:%S"),
        "source":      result.detected_language,
        "transcript":  result.transcript[:80],
        "translation": result.translated_text[:80],
        "emotion":     result.emotion_emoji,
    })
    if len(st.session_state.history) > 10:
        st.session_state.history.pop(0)

# Auto-refresh when mic is active (every 0.5 seconds)
if st.session_state.is_mic_active:
    time.sleep(0.5)
    st.rerun()
