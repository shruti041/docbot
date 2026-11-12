from dotenv import load_dotenv

load_dotenv()

import os
import html as html_lib
import re
import json
import math
import random
import time
import base64
import hashlib
import hmac
import io
import logging
import wave
import tempfile
import threading
from collections import Counter, OrderedDict, deque
from contextlib import contextmanager
from itertools import chain

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-3.5-flash-lite"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

## How many passages each search method returns (merged, duplicates removed)
TOP_K = 3
NOT_FOUND_ANSWER = "I couldn't find that in the document."
## Documents up to this many characters (~25 pages, e.g. a resume or report) are
## sent to Gemini whole, so every question -- including "summarize this" -- can be
## answered. The search and the relevance check are only used for longer documents.
WHOLE_DOCUMENT_MAX_CHARS = 60_000
## For "summarize" questions about longer documents: passages sampled across the document
OVERVIEW_PASSAGES = 12
## How many previous chat messages to send to the LLM (3 questions + 3 answers)
MAX_HISTORY_MESSAGES = 6

## Answers to spoken questions are always read aloud automatically.
## Default for the sidebar switch that also reads aloud answers to typed questions.
AUTO_SPEAK_ALL_ANSWERS = False

APP_TITLE = "Document Q&A Chatbot"
BOT_AVATAR = ":material/smart_toy:"
USER_AVATAR = ":material/person:"

## Limits that protect the server and your Gemini quota once the app is public
MAX_UPLOAD_MB = 20                    # keep in sync with maxUploadSize in .streamlit/config.toml
MAX_PAGES = 500                       # longer PDFs take minutes of CPU to index
MAX_QUESTIONS_PER_HOUR = 30           # per visitor (browser session)
MAX_QUESTIONS_PER_HOUR_ALL_USERS = 300  # across everyone using this server

## When SAVE_INDEX_TO_DISK=1 (in .env or the host's secrets), processed PDFs are
## saved here, one file per PDF, so re-uploading a PDF skips embedding even after
## the app restarts. Off by default: on a shared server it would keep the text of
## every uploaded PDF on disk. Recently used PDFs are always kept in memory.
VECTOR_CACHE_DIR = "./vector_cache"
## Part of the cache key: changing the model or chunk settings re-processes PDFs
## instead of reusing results made with the old settings.
CACHE_VERSION = f"{EMBEDDING_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}"

## Skip online Hugging Face Hub checks once the embedding model is downloaded.
## (Must be set before langchain_huggingface is imported.)
_HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE") or os.path.join(
    os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "hub"
)
if os.path.isdir(os.path.join(_HF_HUB_CACHE, "models--" + EMBEDDING_MODEL.replace("/", "--"))):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    ## Hide the harmless "Could not reach the Hub ... Using cached commit hash"
    ## warning that offline mode triggers.
    logging.getLogger("huggingface_hub.hf_api").setLevel(logging.ERROR)

import numpy as np
import pymupdf
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
import streamlit as st

st.set_page_config(page_title=APP_TITLE, page_icon=BOT_AVATAR, layout="centered")


def _load_settings_from_secrets():
    ## Locally, settings come from .env. On a host, they can be set either as
    ## environment variables or in Streamlit secrets (.streamlit/secrets.toml or
    ## the host's secrets page); copy the latter into the environment, where
    ## Gemini's client and the rest of the app look for them.
    for name in ("GOOGLE_API_KEY", "APP_PASSWORD", "SAVE_INDEX_TO_DISK"):
        if os.environ.get(name):
            continue
        try:
            value = st.secrets.get(name)
        except Exception:  # no secrets file at all
            return
        if value:
            os.environ[name] = str(value)


_load_settings_from_secrets()
SAVE_INDEX_TO_DISK = os.environ.get("SAVE_INDEX_TO_DISK", "").strip().lower() in ("1", "true", "yes")

## Colours, robot mascot and animations.
## Light and dark mode each get their own palette: Streamlit sets `color-scheme`
## on the app, and CSS light-dark(<light>, <dark>) picks the matching colour.
## Elements are targeted through Streamlit's data-testid attributes and the
## "st-key-<key>" class that keyed containers get. All motion is switched off for
## people who ask their system for reduced motion.
STYLES = """
<style>
.stApp {
  /* ---- Light: "Morning Mint" (teal + sky + coral)  |  Dark: "Midnight Neon" (violet + cyan + pink) ---- */
  --c1: light-dark(#0D9488, #8B5CF6);
  --c2: light-dark(#0284C7, #22D3EE);
  --accent: light-dark(#FB7185, #F472B6);
  --brand: light-dark(#0F766E, #C4B5FD);          /* brand-coloured text: readable on tints */
  --ink: light-dark(#0F2A2E, #E7E7FA);
  --muted: light-dark(#4B6064, #A9A8C9);
  /* "your message" bubbles: darker than --c1/--c2 so white text stays readable (WCAG AA) */
  --u1: light-dark(#0F766E, #7C3AED);
  --u2: light-dark(#0369A1, #0E7490);
  --tint: light-dark(rgba(13, 148, 136, .09), rgba(139, 92, 246, .14));
  --tint-strong: light-dark(rgba(13, 148, 136, .16), rgba(139, 92, 246, .24));
  --line: light-dark(rgba(13, 148, 136, .28), rgba(167, 139, 250, .32));
  --card: light-dark(#FFFFFF, #15142C);
  --avatar-glow: light-dark(transparent, rgba(139, 92, 246, .9));
  --glow: light-dark(0 16px 32px -16px rgba(13, 148, 136, .55), 0 0 28px -6px rgba(139, 92, 246, .65));
  --aurora-1: light-dark(rgba(45, 212, 191, .30), rgba(139, 92, 246, .36));
  --aurora-2: light-dark(rgba(56, 189, 248, .24), rgba(34, 211, 238, .24));
  --aurora-3: light-dark(rgba(251, 146, 60, .20), rgba(244, 114, 182, .22));
  --gradient: linear-gradient(120deg, var(--c1), var(--c2));
  background-image:
    radial-gradient(60rem 30rem at 50% -12rem, var(--tint-strong), transparent 70%);
}

@keyframes gradient-shift { 0%, 100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
@keyframes rise-in { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes from-left { from { opacity: 0; transform: translateX(-14px); } to { opacity: 1; transform: none; } }
@keyframes from-right { from { opacity: 0; transform: translateX(14px); } to { opacity: 1; transform: none; } }
@keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-7px); } }
@keyframes blink { 0%, 92%, 100% { transform: scaleY(1); } 95% { transform: scaleY(.1); } }
@keyframes look { 0%, 35%, 100% { transform: translateX(0); } 45%, 60% { transform: translateX(calc(var(--u) * -5)); } 70%, 85% { transform: translateX(calc(var(--u) * 5)); } }
@keyframes scan-eyes { 0%, 100% { transform: translateX(calc(var(--u) * -6)); } 50% { transform: translateX(calc(var(--u) * 6)); } }
@keyframes bulb { 0%, 100% { opacity: 1; filter: drop-shadow(0 0 2px var(--accent)); } 50% { opacity: .45; filter: none; } }
@keyframes ping { 0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, .55); } 100% { box-shadow: 0 0 0 9px rgba(34, 197, 94, 0); } }
@keyframes aurora { 0%, 100% { transform: translate(0, 0) scale(1); } 33% { transform: translate(6%, 8%) scale(1.1); } 66% { transform: translate(-6%, 4%) scale(.95); } }
@keyframes dot-bounce { 0%, 80%, 100% { transform: translateY(0); opacity: .35; } 40% { transform: translateY(-6px); opacity: 1; } }
@keyframes scan-line { 0% { top: 8%; } 50% { top: 82%; } 100% { top: 8%; } }
@keyframes avatar-glow { 0%, 100% { box-shadow: 0 0 0 0 var(--tint-strong); } 50% { box-shadow: 0 0 0 7px transparent, 0 0 18px var(--c1); } }
@keyframes shimmer { from { background-position: 0% 50%; } to { background-position: 200% 50%; } }
@keyframes spin { to { transform: rotate(360deg); } }

/* ---------- robot mascot (plain HTML shapes, so st.html keeps it) ---------- */
/* Drawn on a 120 x 120 grid; --u is one grid unit, so the robot scales with its size */
.robot { position: relative; flex: none; width: calc(var(--u) * 120); height: calc(var(--u) * 120); }
.robot.floating { animation: float 4s ease-in-out infinite; }
.robot i { position: absolute; display: block; }
.robot .bulb { left: calc(var(--u) * 53); top: calc(var(--u) * 4); width: calc(var(--u) * 14); height: calc(var(--u) * 14);
               border-radius: 50%; background: var(--accent); box-shadow: 0 0 calc(var(--u) * 12) var(--accent); animation: bulb 1.8s ease-in-out infinite; }
.robot .antenna { left: calc(var(--u) * 58); top: calc(var(--u) * 15); width: calc(var(--u) * 4); height: calc(var(--u) * 15); border-radius: 99px; background: var(--c1); }
.robot .ear { top: calc(var(--u) * 52); width: calc(var(--u) * 12); height: calc(var(--u) * 24); border-radius: calc(var(--u) * 6); background: var(--c2); }
.robot .ear.l { left: calc(var(--u) * 9); }
.robot .ear.r { left: calc(var(--u) * 99); }
.robot .head { left: calc(var(--u) * 18); top: calc(var(--u) * 28); width: calc(var(--u) * 84); height: calc(var(--u) * 72);
               border-radius: calc(var(--u) * 26); background: var(--gradient); box-shadow: inset 0 calc(var(--u) * -6) 0 rgba(0, 0, 0, .10), var(--glow); }
.robot .face { left: calc(var(--u) * 30); top: calc(var(--u) * 42); width: calc(var(--u) * 60); height: calc(var(--u) * 42);
               border-radius: calc(var(--u) * 17); background: light-dark(#FFFFFF, #0E0C24); }
.robot .eyes { left: calc(var(--u) * 42); top: calc(var(--u) * 52); width: calc(var(--u) * 36); height: calc(var(--u) * 15); animation: look 7s ease-in-out infinite; }
.robot .eye { top: 0; width: calc(var(--u) * 12); height: calc(var(--u) * 15); border-radius: 50%;
              background: light-dark(#0F2A2E, #67E8F9); box-shadow: 0 0 calc(var(--u) * 7) light-dark(transparent, #22D3EE); animation: blink 5s infinite; }
.robot .eye.l { left: 0; }
.robot .eye.r { left: calc(var(--u) * 24); }
.robot .mouth { left: calc(var(--u) * 51); top: calc(var(--u) * 66); width: calc(var(--u) * 18); height: calc(var(--u) * 9);
                border-bottom: calc(var(--u) * 3) solid light-dark(#0F2A2E, #67E8F9); border-radius: 0 0 50% 50% / 0 0 100% 100%; }
.robot.reading .eyes { animation: scan-eyes 1.1s ease-in-out infinite; }

/* ---------- text helpers ---------- */
.grad-title {
  margin: 0; font-family: "Plus Jakarta Sans", sans-serif; font-weight: 800; line-height: 1.12;
  background: var(--gradient); background-size: 200% 200%;
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: gradient-shift 8s ease infinite;
}
.mi { font-family: "Material Symbols Rounded"; font-weight: normal; font-style: normal; line-height: 1;
      letter-spacing: normal; white-space: nowrap; direction: ltr; -webkit-font-feature-settings: "liga"; font-feature-settings: "liga"; }
.online { display: inline-flex; align-items: center; gap: .4rem; font-size: .82rem; opacity: .85; }
.online::before { content: ""; width: .5rem; height: .5rem; border-radius: 50%; background: #22C55E; animation: ping 1.8s ease-out infinite; }

/* ---------- upload screen ---------- */
.st-key-hero {
  position: relative; overflow: hidden; isolation: isolate;
  padding: 1.75rem 1.75rem 1.4rem; border-radius: 1.75rem; border: 1px solid var(--line);
  background: var(--card); box-shadow: var(--glow);
  animation: rise-in .6s ease-out both;
}
.st-key-hero::before, .st-key-hero::after {
  content: ""; position: absolute; z-index: -1; width: 70%; aspect-ratio: 1; border-radius: 50%; filter: blur(48px);
  animation: aurora 16s ease-in-out infinite;
}
.st-key-hero::before { top: -45%; left: -20%; background: radial-gradient(circle, var(--aurora-1), transparent 65%); }
.st-key-hero::after { bottom: -55%; right: -20%; background: radial-gradient(circle, var(--aurora-2), var(--aurora-3) 45%, transparent 70%); animation-delay: -8s; }
.hero-head { display: flex; align-items: center; gap: 1.25rem; flex-wrap: wrap; margin-bottom: .25rem; }
.hero-head .grad-title { font-size: clamp(1.8rem, 5.5vw, 2.6rem); }
.eyebrow { display: inline-flex; align-items: center; gap: .4rem; padding: .2rem .7rem; margin-bottom: .45rem; border-radius: 999px;
           font-size: .8rem; font-weight: 600; color: var(--brand); background: var(--tint); border: 1px solid var(--line); }
.hero-sub { margin: .5rem 0 0; font-size: 1.02rem; line-height: 1.55; color: var(--muted); }
[data-testid="stFileUploaderDropzone"] {
  border: 2px dashed var(--line) !important; background: var(--tint) !important;
  transition: border-color .2s ease, background .2s ease, transform .2s ease, box-shadow .2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--brand) !important; background: var(--tint-strong) !important; transform: translateY(-2px); box-shadow: var(--glow);
}
/* progress bar: a gradient that keeps moving while it fills */
[data-testid="stProgressBarTrack"] { background: var(--tint-strong) !important; }
[data-testid="stProgressBarTrack"] > div {
  background: linear-gradient(90deg, var(--c1), var(--c2), var(--c1)) !important; background-size: 200% 100% !important;
  animation: shimmer 2.4s linear infinite;
}
/* processing checklist: done, running, still to do */
.steps { list-style: none; margin: .2rem 0 .1rem !important; padding: 0 !important; display: flex; flex-direction: column; gap: .45rem; font-size: .9rem; }
.steps li { display: flex; align-items: center; gap: .5rem; margin: 0; }
.steps .mi { font-size: 1.1rem; }
.steps .done .mi { color: #22C55E; }
.steps .running { font-weight: 600; }
.steps .todo { color: var(--muted); }
.spinner { flex: none; width: .9rem; height: .9rem; margin: 0 .1rem; border-radius: 50%;
           border: 2px solid var(--tint-strong); border-top-color: var(--c2); animation: spin .9s linear infinite; }
.scan-doc { position: relative; width: 46px; height: 58px; margin: .25rem auto .6rem; border-radius: 8px;
            background: var(--card); border: 2px solid var(--line); overflow: hidden; }
.scan-doc i { position: absolute; left: 8px; right: 8px; height: 3px; border-radius: 2px; background: var(--tint-strong); }
.scan-doc i:nth-child(1) { top: 12px; } .scan-doc i:nth-child(2) { top: 21px; right: 14px; }
.scan-doc i:nth-child(3) { top: 30px; } .scan-doc i:nth-child(4) { top: 39px; right: 18px; }
.scan-doc b { position: absolute; left: 0; right: 0; height: 4px; background: var(--gradient); box-shadow: 0 0 12px var(--c2); animation: scan-line 1.6s ease-in-out infinite; }

.features { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: .9rem; margin-top: 1.4rem; }
.feature { padding: 1.1rem 1.1rem 1rem; border-radius: 1.25rem; background: var(--card); border: 1px solid var(--line);
           animation: rise-in .6s ease-out both; animation-delay: calc(var(--i) * .13s + .2s);
           transition: transform .2s ease, box-shadow .2s ease; }
.feature:hover { transform: translateY(-4px); box-shadow: var(--glow); }
.feature-icon { display: grid; place-items: center; width: 2.6rem; height: 2.6rem; border-radius: .9rem; margin-bottom: .7rem;
                color: #fff; background: var(--gradient); box-shadow: 0 8px 18px -8px var(--c1); }
.feature-icon .mi { font-size: 1.4rem; }
.feature:nth-child(3) .feature-icon { background: linear-gradient(120deg, var(--c2), var(--accent)); }
.feature h4 { margin: 0 0 .3rem; font-family: "Plus Jakarta Sans", sans-serif; font-size: 1rem; font-weight: 700; color: var(--ink); padding: 0; }
.feature p { margin: 0; font-size: .88rem; line-height: 1.5; color: var(--muted); }

/* ---------- chat header ---------- */
.chat-head { display: flex; align-items: center; gap: .9rem; padding: .2rem 0 .4rem; animation: rise-in .5s ease-out both; }
.chat-head .grad-title { font-size: clamp(1.35rem, 4vw, 1.75rem); }
.doc-chip { display: inline-flex; align-items: center; gap: .3rem; padding: .15rem .6rem; border-radius: 999px; font-size: .8rem;
            font-weight: 600; color: var(--brand); background: var(--tint); border: 1px solid var(--line); max-width: 100%; }
.doc-chip .mi { font-size: 1rem; }
.chat-sub { display: flex; flex-wrap: wrap; align-items: center; gap: .6rem; margin-top: .3rem; }

/* ---------- chat messages: bot on the left, you on the right ---------- */
/* speech-bubble corners: the corner nearest the avatar stays sharp */
[class*="st-key-bot_"] [data-testid="stChatMessage"] {
  background: var(--card); border: 1px solid var(--line); border-radius: 1.25rem 1.25rem 1.25rem .4rem;
  padding-right: 1.25rem; animation: from-left .35s ease-out both;
}
[class*="st-key-user_msg"] [data-testid="stChatMessage"] {
  flex-direction: row-reverse; width: fit-content; max-width: min(85%, 40rem); margin-left: auto;
  border-radius: 1.25rem 1.25rem .4rem 1.25rem;
  background: linear-gradient(120deg, var(--u1), var(--u2)); color: #fff; border: none; box-shadow: var(--glow);
  animation: from-right .35s ease-out both;
}
[class*="st-key-user_msg"] [data-testid="stChatMessage"] * { color: #fff !important; }
[class*="st-key-user_msg"] [data-testid="stChatMessageAvatarCustom"] { background: rgba(255, 255, 255, .22); border-color: transparent; }
[class*="st-key-bot_"] [data-testid="stChatMessageAvatarCustom"] {
  background: var(--gradient); border-color: transparent; box-shadow: 0 0 18px -2px var(--avatar-glow);
}
[class*="st-key-bot_"] [data-testid="stChatMessageAvatarCustom"] [data-testid="stIconMaterial"] { color: #fff !important; }
/* the bot's avatar glows while it's thinking */
[class*="st-key-bot_"]:has(.typing) [data-testid="stChatMessageAvatarCustom"] { animation: avatar-glow 1.4s ease-in-out infinite; }
[data-testid="stButtonGroup"] button { transition: transform .15s ease, border-color .15s ease, background .15s ease, box-shadow .15s ease; }
[data-testid="stButtonGroup"] button:hover { transform: translateY(-2px); border-color: var(--brand) !important; background: var(--tint) !important; box-shadow: var(--glow); }
[data-testid="stExpander"] details { border-color: var(--line); background: var(--tint); transition: box-shadow .2s ease; }
[data-testid="stExpander"] details:hover { box-shadow: var(--glow); }
[data-testid="stChatInput"] > div:first-child { transition: box-shadow .2s ease; }
[data-testid="stChatInput"]:focus-within > div:first-child { box-shadow: 0 0 0 4px var(--tint-strong), var(--glow); }
[data-testid="stChatInputMicButton"]:hover { transform: scale(1.1); transition: transform .15s ease; }
/* live microphone level (drawn by MIC_LEVEL_SCRIPT): small dots while it's quiet,
   which stretch into tall glowing bars while you speak. Streamlit's own thin
   waveform is hidden only while recording, so playback afterwards is unchanged. */
[data-testid="stChatInputWaveform"].mic-live > div:not(.mic-dots) { opacity: 0; }
.mic-dots { position: absolute; inset: 0; z-index: 1; display: flex; align-items: center; justify-content: flex-end;
            gap: 4px; padding: 0 6px; overflow: hidden; pointer-events: none; }
.mic-dots i { flex: none; width: 5px; height: 5px; border-radius: 99px; opacity: .45;
              background: linear-gradient(180deg, var(--c2), var(--c1)); transition: height .07s linear, opacity .07s linear; }
.mic-dots i.on { opacity: 1; box-shadow: 0 0 6px var(--c2); }

/* thinking indicator: a little robot reading, plus bouncing dots */
.typing { display: flex; align-items: center; gap: .55rem; padding: .2rem 0; }
.typing .dots { display: inline-flex; gap: .3rem; }
.typing .dots span { width: .5rem; height: .5rem; border-radius: 50%; background: var(--gradient); animation: dot-bounce 1.2s infinite ease-in-out both; }
.typing .dots span:nth-child(2) { animation-delay: .15s; }
.typing .dots span:nth-child(3) { animation-delay: .3s; }
.typing em { font-style: normal; font-size: .9rem; color: var(--muted); }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background-image: linear-gradient(180deg, var(--tint-strong), transparent 60%); }
.side-bot { display: flex; align-items: center; gap: .7rem; padding: .7rem .8rem; border-radius: 1.1rem;
            background: var(--card); border: 1px solid var(--line); box-shadow: var(--glow); }
.side-bot b { display: block; font-family: "Plus Jakarta Sans", sans-serif; font-size: 1rem; color: var(--ink); }
.st-key-doc_card { background: var(--card); border-color: var(--line) !important; }
[data-testid="stSidebar"] button { transition: transform .15s ease, box-shadow .15s ease; }
[data-testid="stSidebar"] button:not(:disabled):hover { transform: translateY(-1px); box-shadow: var(--glow); }

/* ---------- phones: compact header and hero, bigger touch targets ---------- */
@media (max-width: 640px) {
  .chat-head { gap: .6rem; }
  .chat-head .robot { --u: .36px !important; }
  .chat-head .grad-title { font-size: 1.2rem; }
  .st-key-hero { padding: 1.3rem 1.1rem 1.1rem; border-radius: 1.6rem; }
  .hero-head { gap: .6rem; }
  .hero-head .robot { --u: .62px !important; }
  .hero-head .grad-title { font-size: 1.8rem; }
}
@media (pointer: coarse) {
  [data-testid="stButtonGroup"] button { min-height: 2.75rem; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""
st.html(STYLES)

## Live microphone level while recording a question. Streamlit draws thin lines
## that barely move with normal speech, and their look can't be configured, so
## this listens to the same microphone stream and draws dots over them: each new
## dot enters on the right, and its height shows how loud you are at that moment
## (measured in decibels, like hearing, so speech clearly stands out from silence).
MIC_LEVEL_SCRIPT = """
<script>
(() => {
  if (window.__micLevel) return;
  window.__micLevel = true;
  const media = navigator.mediaDevices;
  if (!media || !media.getUserMedia) return;
  const getUserMedia = media.getUserMedia.bind(media);
  media.getUserMedia = async (constraints) => {
    const stream = await getUserMedia(constraints);
    if (constraints && constraints.audio) {
      try { showLevel(stream); } catch (e) { /* never break recording */ }
    }
    return stream;
  };

  let current = null;  // the microphone being shown; a new recording replaces it

  function showLevel(stream) {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    if (current) current.stop();
    const audio = new AudioCtx();
    audio.resume().catch(() => {});
    const analyser = audio.createAnalyser();
    analyser.fftSize = 1024;
    audio.createMediaStreamSource(stream).connect(analyser);
    const samples = new Float32Array(analyser.fftSize);
    let box = null, row = null, dots = [], levels = [], last = 0, smooth = 0, stopped = false;

    const live = () => stream.getAudioTracks().some(t => t.readyState === "live");
    function hide() {
      if (row) row.remove();
      if (box) box.classList.remove("mic-live");
      box = row = null;
    }
    const me = { stop() { stopped = true; hide(); audio.close().catch(() => {}); } };
    current = me;

    function frame(now) {
      if (stopped) return;
      if (!live()) return me.stop();
      // Streamlit shows the waveform box only while it's recording
      const target = document.querySelector('[data-testid="stChatInputWaveform"]');
      if (!target || !target.offsetWidth) {
        hide();
        return setTimeout(() => requestAnimationFrame(frame), 250);  // check again soon
      }
      if (target !== box || !row || !row.isConnected) {
        hide();
        box = target;
        row = document.createElement("div");
        row.className = "mic-dots";
        row.setAttribute("aria-hidden", "true");
        const count = Math.max(8, Math.floor((box.offsetWidth - 12) / 9));
        for (let i = 0; i < count; i++) row.appendChild(document.createElement("i"));
        box.appendChild(row);
        box.classList.add("mic-live");
        dots = Array.from(row.children);
        levels = new Array(count).fill(0);
        smooth = 0;
      }
      analyser.getFloatTimeDomainData(samples);
      let sum = 0;
      for (const s of samples) sum += s * s;
      const decibels = 20 * Math.log10(Math.sqrt(sum / samples.length) + 1e-8);
      // -50 dB (room hush) -> 0 ... -15 dB (clear speech) -> 1
      const level = Math.min(1, Math.max(0, (decibels + 50) / 35));
      smooth = Math.max(level, smooth * 0.8);  // rise instantly, fall gently
      if (now - last > 70) {  // a new dot about 14 times a second
        last = now;
        levels.push(smooth);
        levels.shift();
        const tallest = Math.max(10, box.offsetHeight - 6);
        dots.forEach((dot, i) => {
          dot.style.height = (5 + levels[i] * (tallest - 5)).toFixed(1) + "px";
          dot.classList.toggle("on", levels[i] > 0.1);
        });
      }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
})();
</script>
"""


def robot_html(size, extra_class=""):
    """The app's robot mascot, built from plain HTML shapes (st.html removes SVG).
    It blinks, glances around and its antenna light pulses. "floating" makes it
    bob; "reading" makes its eyes scan back and forth."""
    return (f'<div class="robot {extra_class}" style="--u: {size / 120:.4f}px" aria-hidden="true">'
            '<i class="bulb"></i><i class="antenna"></i><i class="ear l"></i><i class="ear r"></i>'
            '<i class="head"></i><i class="face"></i>'
            '<i class="eyes"><i class="eye l"></i><i class="eye r"></i></i><i class="mouth"></i></div>')


@contextmanager
def thinking(text):
    ## A little robot "reading" with bouncing dots (used like st.spinner) until the block finishes
    slot = st.empty()
    slot.html(f'<div class="typing">{robot_html(30, "reading")}'
              f'<span class="dots"><span></span><span></span><span></span></span><em>{text}</em></div>')
    try:
        yield
    finally:
        slot.empty()


class ProcessingSteps:
    """The checklist shown while a PDF is processed: finished steps get a tick,
    the running step a spinner, and later steps are greyed out."""

    ## step: (label while running or waiting, label once done)
    STEPS = {"read": ("Reading pages", "Read the pages"),
             "split": ("Splitting into passages", "Split into passages"),
             "index": ("Indexing passages", "Indexed the passages"),
             "suggest": ("Writing suggested questions", "Wrote suggested questions")}

    def __init__(self, container):
        self.slot = container.empty()
        self.results = {}
        self.current = 0
        self.render()

    def start(self, step, result=None):
        ## `step` begins and every step before it is done. `result` describes
        ## the step that just finished, e.g. "Read 12 pages".
        keys = list(self.STEPS)
        index = keys.index(step)
        if result and index > 0:
            self.results[keys[index - 1]] = result
        self.current = index
        self.render()

    def render(self):
        rows = []
        for i, (key, (label, done_label)) in enumerate(self.STEPS.items()):
            if i < self.current:
                rows.append(f'<li class="done"><span class="mi" aria-label="Done">check_circle</span>'
                            f'{self.results.get(key, done_label)}</li>')
            elif i == self.current:
                rows.append(f'<li class="running"><span class="spinner" role="img" aria-label="In progress">'
                            f'</span>{label}</li>')
            else:
                rows.append(f'<li class="todo"><span class="mi" aria-hidden="true">radio_button_unchecked'
                            f'</span>{label}</li>')
        self.slot.html(f'<ol class="steps">{"".join(rows)}</ol>')


# ---------------------------------------------------------------- models

@st.cache_resource(show_spinner=False)
def get_llm():
    ## Created once instead of on every rerun.
    ## "minimal" thinking is this model's default; set explicitly to keep answers fast.
    return ChatGoogleGenerativeAI(model=LLM_MODEL, thinking_level="minimal")


@st.cache_resource(show_spinner=False)
def get_embeddings():
    ## Free local embedding model -- runs on your PC, no API
    #  key or rate limits. Loaded once and reused for every upload.
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ---------------------------------------------------------------- document processing

## Passages embedded per step; smaller steps = smoother progress bar
EMBED_BATCH = 64
## Processed PDFs kept in memory (more are loaded from VECTOR_CACHE_DIR when needed)
MEMORY_CACHE_SIZE = 5


@st.cache_resource(show_spinner=False)
def _memory_cache():
    ## {document key: vector store}, shared by all users of this app, newest last
    return OrderedDict()


def document_key(pdf_bytes):
    return hashlib.sha256(CACHE_VERSION.encode() + pdf_bytes).hexdigest()


def _build_vector_db(pdf_bytes, source, report):
    ## Document loading (pages are numbered from 1, like in a PDF viewer)
    report("Reading pages...", 0.0, step="read")
    try:
        pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise ValueError("This file isn't a valid PDF, or it's damaged.") from None
    with pdf:
        if pdf.needs_pass:
            raise ValueError("This PDF is password-protected. Remove the password and upload it again.")
        if pdf.page_count > MAX_PAGES:
            raise ValueError(f"This PDF has {pdf.page_count:,} pages. The limit is {MAX_PAGES:,} pages; "
                             "split it into smaller PDFs and upload the part you need.")
        docs = [
            Document(page_content=page.get_text(),
                     metadata={"source": source, "page": i})
            for i, page in enumerate(pdf, start=1)
        ]

    ## Splitting
    report(f"Read {len(docs)} pages. Splitting into passages...", 0.03,
           step="split", result=f"Read {plural(len(docs), 'page')}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = [d for d in splitter.split_documents(docs) if d.page_content.strip()]
    if not docs:
        raise ValueError("No text found in this PDF. It may be a scanned image -- "
                         "only PDFs with selectable text are supported.")
    report(f"Indexing passages: 0 of {len(docs):,}", 0.05,
           step="index", result=f"Split into {plural(len(docs), 'passage')}")

    ## Vector embeddings and vector store, in batches so progress can be shown
    vector_db = InMemoryVectorStore(embedding=get_embeddings())
    for start in range(0, len(docs), EMBED_BATCH):
        vector_db.add_documents(docs[start:start + EMBED_BATCH])
        done = min(start + EMBED_BATCH, len(docs))
        report(f"Indexing passages: {done:,} of {len(docs):,}", 0.05 + 0.95 * done / len(docs))
    return vector_db


def document_process(path, on_progress=None):
    """Loads, splits and embeds a PDF. Results are cached in memory (and on disk
    if SAVE_INDEX_TO_DISK is on), so the same PDF is only processed once. While a PDF is processed,
    on_progress(message, fraction, step=None, result=None) is called: `step`
    names a step that's starting ("read", "split" or "index"), and `result`
    describes the step that just finished (e.g. "Read 12 pages")."""
    report = on_progress or (lambda message, fraction, step=None, result=None: None)
    with open(path, "rb") as f:
        pdf_bytes = f.read()
    if not pdf_bytes:
        raise ValueError("The file is empty (0 bytes). If it's stored in OneDrive or Google "
                         "Drive, download it to your PC first, then upload it again.")
    key = document_key(pdf_bytes)

    memory = _memory_cache()
    if key in memory:
        memory.move_to_end(key)
        return memory[key]

    vector_db = None
    cache_path = os.path.join(VECTOR_CACHE_DIR, f"{key}.json")
    if SAVE_INDEX_TO_DISK and os.path.exists(cache_path):
        try:
            report("Opening the saved index (this PDF was read before)...", 0.5, step="index")
            vector_db = InMemoryVectorStore.load(cache_path, embedding=get_embeddings())
        except Exception:
            vector_db = None  # damaged cache file -- rebuild it below
    if vector_db is None:
        vector_db = _build_vector_db(pdf_bytes, os.path.basename(path), report)
        if SAVE_INDEX_TO_DISK:
            ## Write to a temporary file first, so a crash can't leave a half-written cache
            os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)
            vector_db.dump(cache_path + ".tmp")
            os.replace(cache_path + ".tmp", cache_path)

    memory[key] = vector_db
    while len(memory) > MEMORY_CACHE_SIZE:
        memory.popitem(last=False)
    return vector_db


@st.cache_data(show_spinner=False, max_entries=20)
def suggest_questions(doc_key, _vector_db):
    """A few starter questions, written by Gemini from passages spread across the
    document (so they're answerable). Cached per document."""
    texts = [item["text"] for item in _vector_db.store.values()]
    picks = sorted({int(len(texts) * (i + 0.5) / 8) for i in range(8)})
    sample = "\n\n---\n\n".join(texts[i][:700] for i in picks)
    try:
        reply = get_llm().invoke(
            "Below are excerpts from a document. Write 4 short, specific questions (at most 10 "
            "words each) that a reader might ask and that these excerpts answer. Write one "
            "question per line, with no numbering or extra text.\n\n" + sample).text
    except Exception:
        return []
    questions = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip() for line in reply.splitlines()]
    return [q for q in questions if q.endswith("?") and len(q) <= 90][:4]


# ---------------------------------------------------------------- search

def _words(text):
    return re.findall(r"\w+", text.lower())


class KeywordIndex:
    """BM25 keyword search -- finds exact words and names that meaning-based
    search can miss. Building it takes ~0.1s even for a 400-page PDF."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.counts = [Counter(_words(d.page_content)) for d in docs]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        doc_freq = Counter(word for c in self.counts for word in c)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in doc_freq.items()}

    def search(self, query, k):
        query_words = set(_words(query)) & self.idf.keys()
        if not query_words:
            return []
        scores = []
        for i, counts in enumerate(self.counts):
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_length)
            score = sum(self.idf[w] * counts[w] * (self.k1 + 1) / (counts[w] + norm)
                        for w in query_words if w in counts)
            if score > 0:
                scores.append((score, i))
        return [self.docs[i] for _, i in sorted(scores, reverse=True)[:k]]

    def find_numbered_heading(self, query, k=2):
        ## "Article 21", "Section 4A", ... -> the first passages with a heading line
        ## like "21. Protection of life..." (in document order, so the main text comes
        ## before schedules/appendices; a bare "21." line, as in a table of
        ## contents, is skipped).
        refs = re.findall(r"\b(?:article|section|clause|rule)\s+(\d+[a-z]?)\b", query, re.IGNORECASE)
        found = []
        for ref in refs:
            heading = re.compile(rf"^[ \t]*{re.escape(ref)}\.[ \t]+\S", re.MULTILINE | re.IGNORECASE)
            found += [d for d in self.docs if heading.search(d.page_content)][:k]
        return found


def build_keyword_index(vector_db):
    docs = [Document(page_content=item["text"], metadata=item["metadata"])
            for item in vector_db.store.values()]
    return KeywordIndex(docs)


def _is_non_latin(text):
    ## True for text in scripts like Hindi (Devanagari), Tamil, Bengali, ...
    return re.search(r"[^\x00-ɏ -⁯]", text) is not None


def to_english(query):
    ## The embedding model and keyword search only understand English, so questions
    ## in other scripts (e.g. Hindi) are translated for searching. The answer is
    ## still given in the original language.
    if not _is_non_latin(query):
        return query
    translation = get_llm().invoke("Translate this question into English. Reply with only "
                                   f"the translation.\n\n{query}").text.strip()
    return translation or query


## Messages that are only a greeting / thanks / goodbye, and the replies to them
_POLITE_WORDS = r"(?:\s+(?:there|bot|ji|sir|madam|ma'?am|so much|a lot|again))*"
_SMALL_TALK = [
    (r"hi+|hello+|hey+|hii+|namaste|namaskar|good\s+(?:morning|afternoon|evening)",
     "Hello! Ask me anything about **{doc}** — type your question or tap the mic."),
    (r"नमस्ते|नमस्कार|हेलो|हाय",
     "नमस्ते! **{doc}** के बारे में कुछ भी पूछिए — लिखकर या माइक दबाकर।"),
    (r"thanks|thank\s+(?:you|u)|thx|ty",
     "You're welcome! Ask me anything else about **{doc}**."),
    (r"धन्यवाद|शुक्रिया",
     "आपका स्वागत है! **{doc}** के बारे में और कुछ पूछिए।"),
    (r"bye|goodbye|good\s+bye|see\s+you|ok(?:ay)?\s+bye",
     "Goodbye! Come back anytime you have questions about **{doc}**."),
]


def small_talk_reply(query):
    ## Only when the whole message is small talk, so "Hello, what does Article 21
    ## say?" is still answered from the document.
    for pattern, reply in _SMALL_TALK:
        if re.fullmatch(rf"\s*(?:{pattern}){_POLITE_WORDS}\s*[.!?,।]*\s*", query, re.IGNORECASE):
            return reply.format(doc=st.session_state.get("document_name", "the document"))
    return None


def whole_document():
    """All passages in document order if the document is short enough to send
    to Gemini in full, else None."""
    docs = st.session_state.keyword_index.docs
    return docs if sum(len(d.page_content) for d in docs) <= WHOLE_DOCUMENT_MAX_CHARS else None


_DOC_WORD = r"(?:document|doc|pdf|file|book|report|paper|text|resume|cv|handbook|manual)"
_END = r"\s*[?.!]*\s*$"
## Questions about the document as a whole, in the many ways people (and speech
## recognition) phrase them. Most patterns must end the sentence, so specific
## questions like "what does the document say about leave?" use normal search.
_OVERVIEW = re.compile(
    r"\b(?:summar\w*|overview|gist|key points|main points|highlights|tl;?dr)\b"
    # what is this (document) (is) (all) about / what's this pdf about
    rf"|\b(?:this|the)\s+(?:{_DOC_WORD}\s+)?(?:is\s+)?(?:all\s+)?about{_END}"
    # tell me about this pdf
    rf"|\babout\s+(?:this|the)\s+{_DOC_WORD}{_END}"
    # explain / describe this pdf
    rf"|\b(?:describe|explain|introduce)\s+(?:this|the)\s+{_DOC_WORD}{_END}"
    # what does this document contain / cover / talk about
    rf"|\b(?:this|the|it)\s+(?:{_DOC_WORD}\s+)?(?:contains?|covers?|talks?\s+about|"
    rf"discuss(?:es)?|deals?\s+with){_END}"
    # what's the main idea / topic (of this document)
    rf"|\b(?:main|central|overall)\s+(?:idea|topic|theme|subject|purpose|message)s?"
    rf"(?:\s+of\s+(?:this|the|it)(?:\s+{_DOC_WORD})?)?{_END}"
    # the purpose / topic of this document
    rf"|\b(?:topic|subject|purpose|theme)s?\s+of\s+(?:this|the)\s+{_DOC_WORD}\b",
    re.IGNORECASE)


def wants_overview(query):
    ## "Summarize this", "what's this pdf about", "what does it cover", "key points", ...
    return _OVERVIEW.search(query.strip()) is not None


def overview_sample():
    ## The opening passages plus passages spread evenly across a long document
    docs = st.session_state.keyword_index.docs
    picks = {0, 1} | {int(len(docs) * (i + 0.5) / (OVERVIEW_PASSAGES - 2))
                      for i in range(OVERVIEW_PASSAGES - 2)}
    return [docs[i] for i in sorted(picks) if i < len(docs)]


def retrieve(query, previous_question=None):
    """The passages most related to the question, from any PDF. There's no fixed
    relevance cutoff: Gemini decides from them whether the document answers the
    question (and replies "not found" if it doesn't)."""
    index, vector_db = st.session_state.keyword_index, st.session_state.vector_db
    ## Combine exact references, meaning-based search and keyword search
    candidates = chain(index.find_numbered_heading(query),
                       vector_db.similarity_search(query, k=TOP_K),
                       index.search(query, k=TOP_K))
    if previous_question:
        ## Follow-ups like "explain that" or "tell me more" match nothing on their
        ## own, so also include a few passages for the previous question
        combined = f"{previous_question} {query}"
        candidates = chain(candidates, index.find_numbered_heading(combined),
                           vector_db.similarity_search(combined, k=2), index.search(combined, k=2))
    unique = {}
    for doc in candidates:
        unique.setdefault(doc.page_content, doc)
    return list(unique.values())


# ---------------------------------------------------------------- answering

def build_prompt(query, documents, history_messages, note=""):
    context = "\n\n".join(f"[page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in documents)
    history = "\n".join(f"{m['role']}: {m['content']}" for m in history_messages)
    ## Decided from the question itself, so an English question after Hindi
    ## messages still gets an English answer.
    language_rule = ("Answer in the same language as the question."
                     if _is_non_latin(query) else "Answer in English.")
    document = st.session_state.get("document_name", "the uploaded document")
    return f"""You answer questions about the user's uploaded document, "{document}", using ONLY the
document excerpts below.

Rules:
- Use only facts stated in the excerpts. Do not add outside knowledge, even if you know the answer.
- The conversation so far is only for understanding follow-up questions (like "explain that").
  You may rephrase or simplify earlier answers, but don't add facts that aren't in the excerpts.
  For "tell me more"-style requests, give details from the excerpts that the earlier answer didn't
  include instead of repeating it.
- For laws, rules, numbers, dates and names, stay close to the exact wording of the excerpts.
- Questions may come from speech recognition and be oddly worded. If a question is unclear,
  answer its most likely meaning.
- You may draw simple, direct conclusions from the excerpts (for example, whether someone has
  work experience, or how many years), and say what they're based on.
- Don't put page numbers or citations like "(page 41)" inside the answer; the app shows the
  sources below the answer.
- If the excerpts answer only part of the question, answer that part and say what's missing.
- If the excerpts contain nothing that answers the question, reply exactly: "{NOT_FOUND_ANSWER}"
- If the whole message is just small talk (a greeting, "how are you", thanks, "who are you") and
  asks nothing about the document, reply briefly and warmly and invite questions about the
  document. Otherwise, answer directly without greetings or pleasantries.
- {language_rule}{note}
- End with one last line, in English and plain text, listing the page numbers of the excerpts
  your answer used: "{SOURCES_MARKER} 12, 41". Write "{SOURCES_MARKER} none" if you used none
  (small talk, or nothing found).

document excerpts:
{context}

conversation so far:
{history or "(none)"}

question: {query}"""


## Gemini ends each answer with a line like "SOURCES: 12, 41". AnswerStream takes it
## off before the answer is shown, and pick_sources turns it into the Sources box.
SOURCES_MARKER = "SOURCES:"
## "page 41", "pages 41 and 42", Hindi "पृष्ठ 41"
_PAGE_REF = r"(?:\bpages?|\bp\.|पृष्ठ)\s*(\d+(?:\s*(?:,|and|&|-|–|और|तथा)\s*\d+)*)"


class AnswerStream:
    """Passes Gemini's answer through as it streams, but holds back the final
    "SOURCES: ..." line (kept in .sources_line), so readers never see it."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.sources_line = ""

    def __iter__(self):
        marker = SOURCES_MARKER.upper()
        pending = ""
        for piece in self.chunks:
            if self.sources_line:  # already past the marker
                self.sources_line += piece
                continue
            pending += piece
            at = pending.upper().find(marker)
            if at >= 0:
                self.sources_line = pending[at + len(marker):] or " "
                text = pending[:at].rstrip(" \t\n*_")
                if text:
                    yield text
                pending = ""
                continue
            ## Hold back the end if it could be the start of the marker, plus any
            ## spaces / newlines / bold markers just before it
            keep = 0
            for size in range(min(len(marker) - 1, len(pending)), 0, -1):
                if marker.startswith(pending[-size:].upper()):
                    keep = size
                    break
            while keep < len(pending) and pending[-keep - 1] in " \t\n*_":
                keep += 1
            if len(pending) > keep:
                yield pending[:len(pending) - keep]
                pending = pending[len(pending) - keep:]
        if pending and not self.sources_line:
            yield pending.rstrip()


def remove_inline_citations(text):
    ## In case Gemini still writes "(page 41)" inside the answer
    return re.sub(rf"\s*\((?:see\s+)?{_PAGE_REF}\)", "", text, flags=re.IGNORECASE).strip()


def pick_sources(answer, sources_line, documents):
    """The passages from the pages listed on the answer's SOURCES line, plus any
    listed pages that weren't among the passages, which may be wrong. An answer
    that lists no pages (e.g. a reply to small talk) gets no sources."""
    cited = {int(n) for n in re.findall(r"\d+", sources_line)}
    if not sources_line.strip():
        ## No SOURCES line: fall back to page references inside the answer
        for group in re.findall(_PAGE_REF, answer, re.IGNORECASE):
            cited.update(int(n) for n in re.findall(r"\d+", group))
    used = [d for d in documents if d.metadata.get("page") in cited]
    sources = [{"page": d.metadata.get("page"), "text": d.page_content} for d in used]
    unverified = sorted(cited - {d.metadata.get("page") for d in documents})
    return sources, unverified


def _escape_markdown(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return re.sub(r"([\\`*_{}\[\]<>()#+\-.!|~$])", r"\\\1", text)


def show_sources(sources, unverified_pages=()):
    ## Collapsible list of the exact passages the answer was based on
    if unverified_pages:
        st.caption(":material/warning: Cited page(s) not among the passages used, please check: "
                   + ", ".join(map(str, unverified_pages)))
    if not sources:
        return
    pages = sorted({s["page"] for s in sources}, key=lambda p: (p is None, p))
    label = "Source: page " if len(pages) == 1 else "Sources: pages "
    label += ", ".join(map(str, pages[:5])) + (f" +{len(pages) - 5} more" if len(pages) > 5 else "")
    with st.expander(label, icon=":material/menu_book:"):
        for s in sources:
            st.markdown(f"**Page {s['page']}**")
            st.markdown(_escape_markdown(s["text"]))


def _answer_text(stream):
    ## Yield only the text from Gemini's streamed chunks
    for chunk in stream:
        if chunk.text:
            yield chunk.text


# ---------------------------------------------------------------- voice

UNCLEAR_SPEECH = "[unclear]"


def transcribe(audio_file):
    ## Speech -> text with Gemini (same API key, no extra packages)
    message = HumanMessage(content=[
        {"type": "text", "text": "Transcribe the speech in this recording exactly, in the language "
                                 "spoken. Reply with only the transcription. If there is no speech, "
                                 "or it is too faint, noisy or garbled to understand, reply with "
                                 f"exactly: {UNCLEAR_SPEECH}"},
        {"type": "audio", "base64": base64.b64encode(audio_file.getvalue()).decode(),
         "mime_type": "audio/wav"},
    ])
    return get_llm().invoke([message]).text.strip()


def sounds_silent(audio_bytes):
    """True if a recording is (nearly) silent or just an accidental tap. Checked
    before transcription: given silence, speech models sometimes invent words
    (e.g. a random sentence in another language)."""
    try:
        with wave.open(io.BytesIO(audio_bytes)) as recording:
            width, rate = recording.getsampwidth(), recording.getframerate()
            channels = recording.getnchannels()
            frames = recording.readframes(recording.getnframes())
        samples = np.frombuffer(frames, dtype={1: np.uint8, 2: np.int16, 4: np.int32}[width])
    except Exception:
        return False  # can't read it here; let Gemini try
    samples = samples.astype(np.float32)
    if width == 1:
        samples -= 128  # 8-bit WAV is unsigned
    samples /= float(2 ** (8 * width - 1))
    if len(samples) / (rate * channels) < 0.4:
        return True
    ## Loudness of each 30 ms slice; very soft speech is around 0.005 and up
    slice_len = int(rate * channels * 0.03)
    count = len(samples) // slice_len
    loudness = np.sqrt((samples[:count * slice_len].reshape(count, slice_len) ** 2).mean(axis=1))
    return int((loudness > 0.005).sum()) < 8  # under ~0.25 s of sound


_FILLER_WORDS = {"uh", "uhh", "um", "umm", "hmm", "hm", "mm", "mmm", "ah", "er", "erm", "oh", "huh", "eh"}


def is_unclear(transcript):
    """True if nothing usable was heard: silence, noise, or only "um"/"hmm"."""
    text = (transcript or "").strip()
    ## "[unclear]", and tags like "(inaudible)" or "[no speech]" that models sometimes write
    if not text or re.fullmatch(r"[\[(].*[\])]\W*", text):
        return True
    words = re.findall(r"\w+", text.lower())
    return not words or all(word in _FILLER_WORDS for word in words)


## Gentle replies when a recording couldn't be understood (one is picked at random,
## so the bot doesn't sound like a machine repeating itself)
_REPEAT_REPLIES = [
    "Sorry, I couldn't quite hear you. Could you please say that again?",
    "I'm sorry, I didn't catch that. Could you repeat it, maybe a little closer to the mic?",
    "Hmm, I wasn't able to hear you clearly. Would you mind saying it once more?",
    "Sorry, that didn't come through clearly. Please try again, or type your question if that's easier.",
]
_TROUBLE_REPLY = ("I'm so sorry, I'm having a little trouble hearing you right now. "
                  "Please try again in a moment, or type your question below.")
## Added to "not found" replies to spoken questions, in case the question was misheard
MISHEARD_HINT = "If I misheard you, please say it again, or type your question."


def ask_to_repeat(index, trouble=False):
    ## Shown (and spoken) as the bot's reply instead of an answer. Not saved to the
    ## chat history, so it doesn't confuse follow-up questions.
    reply = _TROUBLE_REPLY if trouble else random.choice(_REPEAT_REPLIES)
    with chat_bubble("assistant", index):
        st.markdown(f":material/hearing: {reply}")
    speak(reply)


def _run_voice_script(js):
    ## Runs in the page itself (not an iframe), so the browser allows speech after
    ## the user's click. The timestamp makes each call new, so it always runs.
    st.html(f"<script>(() => {{ {js} }})(); /* {time.time_ns()} */</script>",
            unsafe_allow_javascript=True)


def speak(markdown_text):
    ## Text -> speech with the browser's built-in voice (free, works offline)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", markdown_text)  # links -> link text
    text = re.sub(r"[*_`#>|~]+", " ", text)                        # markdown symbols
    text = re.sub(r"\s+", " ", text).strip()
    ## Speak sentence by sentence: some browsers cut off very long utterances
    sentences = [s for s in re.split(r"(?<=[.!?।])\s+", text) if s]
    lang = "hi-IN" if re.search(r"[ऀ-ॿ]", text) else ""  # Hindi script -> Hindi voice
    payload = json.dumps(sentences).replace("</", "<\\/")
    _run_voice_script(f"""
        const synth = window.speechSynthesis;
        if (!synth) return;
        synth.cancel();
        setTimeout(() => {{
            for (const sentence of {payload}) {{
                const utterance = new SpeechSynthesisUtterance(sentence);
                if ("{lang}") utterance.lang = "{lang}";
                synth.speak(utterance);
            }}
        }}, 50);""")


def voice_buttons(key, text):
    ## Listen / Stop buttons under an answer
    with st.container(horizontal=True, gap="small"):
        if st.button(":material/volume_up: Listen", key=f"listen_{key}", type="tertiary"):
            speak(text)
        if st.button(":material/stop_circle: Stop", key=f"stop_{key}", type="tertiary"):
            _run_voice_script("window.speechSynthesis && window.speechSynthesis.cancel();")


# ---------------------------------------------------------------- UI

def reset_document():
    ## Sidebar "Upload a different PDF": back to the upload screen with a fresh uploader
    for key in ["vector_db", "keyword_index", "document_name", "document_pages",
                "document_passages", "suggestions"]:
        st.session_state.pop(key, None)
    st.session_state.document_uploaded = False
    st.session_state.messages = []
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1


def plural(count, word):
    return f"{count:,} {word}" + ("" if count == 1 else "s")


def clear_chat():
    st.session_state.messages = []


def use_suggestion():
    ## A suggested question was clicked: ask it, and clear the selection
    st.session_state.pending_query = st.session_state.suggestion_pick
    st.session_state.suggestion_pick = None


@st.cache_resource(show_spinner=False)
def _server_question_log():
    ## When recent questions were asked, by all visitors (shared across sessions)
    return {"times": deque(), "lock": threading.Lock()}


def check_question_limit():
    """Records a question, or returns a message if this visitor or the whole
    server has reached the hourly limit (which protects the Gemini quota)."""
    now = time.time()
    mine = [t for t in st.session_state.question_times if now - t < 3600]
    st.session_state.question_times = mine
    if len(mine) >= MAX_QUESTIONS_PER_HOUR:
        wait = max(1, math.ceil((3600 - (now - mine[0])) / 60))
        return (f"You've reached the limit of {MAX_QUESTIONS_PER_HOUR} questions per hour. "
                f"Please try again in {plural(wait, 'minute')}.")
    log = _server_question_log()
    with log["lock"]:
        times = log["times"]
        while times and now - times[0] >= 3600:
            times.popleft()
        if len(times) >= MAX_QUESTIONS_PER_HOUR_ALL_USERS:
            return "The bot is very busy right now. Please try again in a few minutes."
        times.append(now)
    mine.append(now)
    return None


def require_password():
    """If APP_PASSWORD is set, visitors enter it once per browser session
    before they can use the app."""
    password = os.environ.get("APP_PASSWORD", "")
    if not password or st.session_state.get("authenticated"):
        return
    gate = st.container(key="hero")
    gate.html(f"""<div class="hero-head">
      {robot_html(96, "floating")}
      <div style="flex: 1 1 260px; min-width: 0;">
        <span class="eyebrow"><span class="mi">lock</span>Private access</span>
        <h1 class="grad-title">{APP_TITLE}</h1>
        <p class="hero-sub">Enter the password you were given to start chatting with your documents.</p>
      </div></div>""")
    with gate.form("login", border=False):
        attempt = st.text_input("Password", type="password")
        if st.form_submit_button("Unlock", icon=":material/lock_open:", type="primary"):
            if hmac.compare_digest(attempt.encode(), password.encode()):
                st.session_state.authenticated = True
                st.rerun()
            time.sleep(1)  # slows down password guessing
            st.error("That password isn't right. Please try again.", icon=":material/error:")
    st.stop()


def chat_bubble(role, index):
    ## A chat message inside a keyed container, so the styles can colour user and
    ## bot messages differently (Streamlit doesn't mark which is which)
    is_user = role == "user"
    box = st.container(key=f"{'user_msg' if is_user else 'bot_msg'}_{index}")
    return box.chat_message(role, avatar=USER_AVATAR if is_user else BOT_AVATAR)


def show_message(message, index):
    with chat_bubble(message["role"], index):
        st.markdown((":material/mic: " if message.get("voice") else "") + message["content"])
        if message["role"] == "assistant":
            show_answer_extras(message, index)


def show_answer_extras(message, index):
    ## Under an answer: sources, a tip if nothing was found, and Listen / Stop
    show_sources(message.get("sources"), message.get("unverified_pages"))
    if message.get("not_found") and message.get("asked_by_voice"):
        st.caption(f":material/hearing: {MISHEARD_HINT}")
    elif message.get("not_found"):
        st.caption(":material/lightbulb: Try rephrasing, or use names, numbers or terms "
                   "that appear in the document.")
    voice_buttons(index, message["content"])


st.session_state.setdefault("document_uploaded", False)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("auto_speak", AUTO_SPEAK_ALL_ANSWERS)
st.session_state.setdefault("question_times", [])

if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
    st.error("The Gemini API key is missing. Set GOOGLE_API_KEY in .env (when running locally) "
             "or in your host's secrets / environment settings, then restart the app.",
             icon=":material/key_off:")
    st.stop()

require_password()

### ------------------------------------------------ upload screen
if not st.session_state.document_uploaded:
    ## Hero card: title, uploader and processing progress
    hero = st.container(key="hero")
    ## While a picked PDF is being processed, the robot reads and the text says so
    busy = st.session_state.get(f"uploader_{st.session_state.uploader_key}") is not None
    subtitle = ("Hang tight, I'm reading your PDF." if busy else
                "Upload a PDF and ask me anything about it, by typing or speaking. "
                "Every answer shows the pages it came from.")
    hero.html(f"""<div class="hero-head">
      {robot_html(96, "floating reading" if busy else "floating")}
      <div style="flex: 1 1 260px; min-width: 0;">
        <span class="eyebrow"><span class="mi">auto_awesome</span>Your AI document assistant</span>
        <h1 class="grad-title">{APP_TITLE}</h1>
        <p class="hero-sub">{subtitle}</p>
      </div></div>""")

    ## Error from the previous attempt, shown above a fresh uploader
    if "upload_error" in st.session_state:
        hero.error(st.session_state.pop("upload_error"), icon=":material/error:")

    ## Uploader sits in a placeholder so it can be removed once a file is picked.
    ## Changing its key after an error gives a fresh, empty uploader.
    uploader_slot = hero.empty()
    file = uploader_slot.file_uploader(
        "Upload a PDF", type="pdf", accept_multiple_files=False,
        key=f"uploader_{st.session_state.uploader_key}",
        help=f"PDFs with selectable text, up to {MAX_UPLOAD_MB} MB and {MAX_PAGES} pages. "
             "Scanned images and password-protected files aren't supported.")

    if file and file.size > MAX_UPLOAD_MB * 1024 * 1024:
        ## Normally Streamlit already refuses these (maxUploadSize in config.toml)
        st.session_state.upload_error = (f"\"{file.name}\" is {file.size / 1024 / 1024:.0f} MB. "
                                         f"The limit is {MAX_UPLOAD_MB} MB.")
        st.session_state.uploader_key += 1
        st.rerun()

    if file:
        ## Hide the uploader (and its "+" replace button) while processing
        uploader_slot.empty()
        status = hero.status(f"Reading **{file.name}**", expanded=True)
        ## Created directly on the status box: leaving a "with status:" block
        ## would mark it complete while processing is still running
        status.html('<div class="scan-doc"><i></i><i></i><i></i><i></i><b></b></div>')
        progress = status.progress(0.0, text="Starting...")
        steps = ProcessingSteps(status)
        status.caption(":material/schedule: Large PDFs take about a minute. "
                       "A PDF that was opened recently opens almost instantly.")

        def on_progress(message, fraction, step=None, result=None):
            progress.progress(min(fraction, 1.0), text=message)
            if step:
                steps.start(step, result)

        try:
            with status:
                ## Each upload gets its own temporary folder, so two users
                ## uploading at once can't overwrite each other's file.
                with tempfile.TemporaryDirectory() as tmp_dir:
                    path = os.path.join(tmp_dir, os.path.basename(file.name))
                    with open(path, "wb") as f:
                        f.write(file.getvalue())
                    vector_db = document_process(path, on_progress=on_progress)

                on_progress("Writing suggested questions...", 1.0, step="suggest",
                            result=f"Indexed {plural(len(vector_db.store), 'passage')}")
                st.session_state.vector_db = vector_db
                st.session_state.keyword_index = build_keyword_index(vector_db)
                st.session_state.document_name = file.name
                st.session_state.document_pages = len({item["metadata"].get("page")
                                                       for item in vector_db.store.values()})
                st.session_state.document_passages = len(vector_db.store)
                st.session_state.suggestions = suggest_questions(document_key(file.getvalue()), vector_db)
            status.update(label=f"**{file.name}** is ready", state="complete", expanded=False)
        except Exception as e:
            ## Show the error with the uploader right back, so another file can be picked
            st.session_state.upload_error = f"Couldn't process \"{file.name}\": {e}"
            st.session_state.uploader_key += 1
            st.rerun()

        st.session_state.document_uploaded = True
        st.session_state.show_upload_success = True
        st.rerun()

    hero.caption(":material/lock: Your PDF is indexed by this app's server"
                 + (" and saved there so it reopens quickly. " if SAVE_INDEX_TO_DISK else
                    " and kept in memory only, never saved to disk. ")
                 + "To answer, the relevant passages (or the whole document, if it's short) "
                   "and any voice recordings are sent to Google Gemini.")

    ## What the app can do: cards that fade in one after another and lift on hover
    features = [
        ("forum", "Ask naturally",
         "Type or tap the mic, in English or Hindi. Follow-ups like \"explain that\" work too."),
        ("menu_book", "Answers with sources",
         "Below each answer, see the pages and exact passages it came from."),
        ("record_voice_over", "Listen to answers",
         "Spoken questions get spoken answers. Replay any answer anytime."),
    ]
    st.html('<div class="features">' + "".join(
        f'<div class="feature" style="--i: {i}"><div class="feature-icon"><span class="mi">{icon}</span></div>'
        f'<h4>{heading}</h4><p>{text}</p></div>'
        for i, (icon, heading, text) in enumerate(features)) + "</div>")

    if not file:
        ## Load the embedding model now, while the user is picking a file,
        ## so it's ready by the time they upload.
        get_embeddings()

### ------------------------------------------------ chat screen
else:
    name = st.session_state.get("document_name", "your document")
    pages = st.session_state.get("document_pages")

    with st.sidebar:
        st.html(f'<div class="side-bot">{robot_html(42)}'
                f'<div><b>DocBot</b><span class="online">Online · ready to help</span></div></div>')
        st.subheader(":material/description: Document")
        with st.container(border=True, key="doc_card"):
            st.markdown(f"**{name}**")
            if pages:
                passages = st.session_state.get("document_passages", 0)
                st.caption(f"{plural(pages, 'page')} · {plural(passages, 'passage')} indexed")
        st.button("Upload a different PDF", icon=":material/upload_file:", width="stretch",
                  on_click=reset_document)
        st.button("Clear chat", icon=":material/delete_sweep:", width="stretch",
                  on_click=clear_chat, disabled=not st.session_state.messages)

        st.subheader(":material/tune: Settings")
        st.toggle("Read every answer aloud", key="auto_speak",
                  help="Answers to spoken questions are always read aloud.")

        st.subheader(":material/lightbulb: Tips")
        st.markdown(
            "- Use exact names, numbers or terms from the document for precise answers.\n"
            "- Tap the :material/mic: mic to ask by voice.\n"
            "- Open **Sources** under an answer to check the exact text.\n"
            "- Follow up naturally: *explain that simply*, *tell me more*.")
        st.caption("Answers come only from your PDF. For important decisions, "
                   "always check the sources.")

    ## Header: the robot, the title and the loaded document
    st.html(f"""<div class="chat-head">{robot_html(54, "floating")}
      <div style="min-width: 0;"><h2 class="grad-title">{APP_TITLE}</h2>
        <div class="chat-sub"><span class="doc-chip"><span class="mi">description</span>{html_lib.escape(name)}{
            f" · {plural(pages, 'page')}" if pages else ""}</span>
          <span class="online">Online</span></div></div></div>""")

    if st.session_state.pop("show_upload_success", False):
        st.toast(f"**{name}** is ready. Ask away!", icon=":material/check_circle:")

    ## Welcome message, with suggested questions until the chat starts
    with st.container(key="bot_welcome").chat_message("assistant", avatar=BOT_AVATAR):
        about = f" ({plural(pages, 'page')})" if pages else ""
        st.markdown(f"Hi! I've read **{name}**{about}. Ask me anything about it: "
                    "type below or tap the :material/mic: mic.")
        ## In a placeholder, so they disappear as soon as the first question is asked
        suggestions_slot = st.empty()
        if not st.session_state.messages and st.session_state.get("suggestions"):
            suggestions_slot.pills("Try asking", st.session_state.suggestions,
                                   key="suggestion_pick", on_change=use_suggestion)

    ## Chat history (Streamlit redraws the page on every rerun)
    for i, message in enumerate(st.session_state.messages):
        show_message(message, i)

    ## Type a question, or click the mic to speak it
    short_name = name if len(name) <= 28 else name[:25] + "..."
    st.html(MIC_LEVEL_SCRIPT, unsafe_allow_javascript=True)  # dots that show your voice level
    user_input = st.chat_input(f"Ask about {short_name}...", accept_audio=True)
    pending = st.session_state.pop("pending_query", None)

    if user_input or pending:
        suggestions_slot.empty()
        limit_message = check_question_limit()
        if limit_message:
            st.warning(limit_message, icon=":material/hourglass_top:")
            st.stop()
        voice = bool(user_input) and user_input.audio is not None
        typed = (user_input.text if user_input else pending) or ""
        spoken = ""
        if voice:
            try:
                if not sounds_silent(user_input.audio.getvalue()):
                    with thinking("Listening to your question..."):
                        spoken = transcribe(user_input.audio)
            except Exception:
                logging.getLogger(__name__).exception("Transcription failed")
                if not typed.strip():
                    ask_to_repeat(len(st.session_state.messages) + 1, trouble=True)
                    st.stop()
            if is_unclear(spoken):
                spoken = ""
                if not typed.strip():
                    ## Nothing understandable was said: gently ask to repeat it
                    ask_to_repeat(len(st.session_state.messages) + 1)
                    st.stop()
        query = " ".join(part for part in [typed, spoken] if part).strip()
        if not query:
            st.stop()
        voice = bool(spoken)  # only questions that were actually heard count as spoken
        with chat_bubble("user", len(st.session_state.messages)):
            st.markdown((":material/mic: " if voice else "") + query)

        with chat_bubble("assistant", len(st.session_state.messages) + 1):
            documents = []
            search_query = query
            answer = small_talk_reply(query)
            is_small_talk = answer is not None
            if is_small_talk:
                ## Greeting / thanks / goodbye -- reply directly, no search needed
                st.markdown(answer)
            else:
                try:
                    with thinking("Reading the document..."):
                        previous_questions = [m.get("search_query", m["content"])
                                              for m in st.session_state.messages
                                              if m["role"] == "user" and not m.get("small_talk")]
                        note = ""
                        whole = whole_document()
                        if whole:
                            ## Short document (resume, report...): Gemini reads all of it,
                            ## so summaries and questions in any wording work
                            documents = whole
                        else:
                            search_query = to_english(query)
                            documents = retrieve(search_query,
                                                 previous_questions[-1] if previous_questions else None)
                            if wants_overview(search_query):
                                ## "What's this pdf about?": passages from across the long
                                ## document, plus the search results in case the question
                                ## is about one part ("summarize Article 21")
                                overview = overview_sample()
                                seen = {d.page_content for d in overview}
                                documents = overview + [d for d in documents if d.page_content not in seen]
                                note = ("\n- Some excerpts are a sample from across a long document. For "
                                        "questions about the whole document, describe what it is and "
                                        "its main topics, and say this is based on selected sections.")

                        if documents:
                            ## Last few messages so follow-ups like "explain that more simply" make sense
                            history = st.session_state.messages[-MAX_HISTORY_MESSAGES:]
                            prompt = build_prompt(query, documents, history, note)

                            ## Stream the answer; keep the dots until the first words arrive
                            answer_stream = AnswerStream(_answer_text(get_llm().stream(prompt)))
                            stream = iter(answer_stream)
                            first_part = next(stream, "")

                    if documents:
                        ## Show the answer word by word as Gemini generates it
                        answer = st.write_stream(chain([first_part], stream))
                        answer = remove_inline_citations(answer if isinstance(answer, str) else "")
                    else:
                        ## Small-talk reply, or "not found" (never answered from Gemini's own knowledge)
                        answer = answer or NOT_FOUND_ANSWER
                        st.markdown(answer)
                except Exception as e:
                    st.error(f"Couldn't get an answer from Gemini: {e}", icon=":material/error:")
                    st.stop()  # don't save a question without an answer to the history

            if not answer:
                answer = "Sorry, I didn't get an answer. Please try asking again."
                st.warning(answer, icon=":material/warning:")

            ## Sources (none for "not found"), tips, and Listen / Stop
            sources, unverified_pages = [], []
            if documents and answer.strip() != NOT_FOUND_ANSWER:
                sources, unverified_pages = pick_sources(answer, answer_stream.sources_line, documents)
            reply = {"role": "assistant", "content": answer, "sources": sources,
                     "unverified_pages": unverified_pages,
                     "not_found": answer.strip() == NOT_FOUND_ANSWER, "asked_by_voice": voice}
            ## Same index as this answer gets in the history list on the next rerun
            show_answer_extras(reply, len(st.session_state.messages) + 1)

            ## Read the answer aloud automatically for spoken questions. If nothing was
            ## found for a spoken question, it may have been misheard: gently say so.
            if voice or st.session_state.auto_speak:
                speak(answer + (" " + MISHEARD_HINT if voice and reply["not_found"] else ""))

        st.session_state.messages += [
            {"role": "user", "content": query, "voice": voice, "search_query": search_query,
             ## Small talk (no citations, not "not found") isn't used as context for follow-ups
             "small_talk": is_small_talk or not (sources or reply["not_found"])},
            reply,
        ]
