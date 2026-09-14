#!/usr/bin/env python3
"""
Streamlit version of the "explain it back" practice loop.

Same idea as interview_loop.py (mic -> whisper -> Ollama, reacted against your
own notes) but adapted to Streamlit's request/rerun model: instead of
continuously streaming the mic through Silero VAD, you record one answer with
the built-in st.audio_input recorder (click to record, click to stop -- the
web equivalent of "speak, then press Enter"). The whole clip is transcribed
once, and the transcript lands in an editable text box before you submit --
so you can fix a bad transcription or just type your answer instead, same as
the typed-answer override in interview_loop.py.

Usage:
    streamlit run streamlit_app.py
"""

import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

# Silences a harmless Streamlit-vs-PyTorch dev-server issue: Streamlit's file
# watcher walks every imported module's __path__ to find files to watch, and
# torch.classes.__path__ is a special proxy that raises on that kind of
# introspection. Patching it to a plain list avoids the "Examining the path
# of torch.classes raised" traceback. Cosmetic only -- the app runs fine
# without this, this just keeps the console clean.
torch.classes.__path__ = []

import streamlit as st

import gdoc_picker
from interview_loop import ask_ollama
from topics import load_topics, parse_export
from transcribe_loop import DEFAULT_MODEL

TOPICS_DIR = Path(__file__).parent / "topics"

st.set_page_config(page_title="Explain It Back", page_icon="🗣️")


@st.cache_resource
def get_whisper_cli() -> str | None:
    return shutil.which("whisper-cli")


def transcribe_wav_bytes(whisper_cli: str, model_path: str, wav_bytes: bytes) -> str:
    """Convert whatever format the browser recorded (via ffmpeg, since
    st.audio_input's sample rate/channel count isn't guaranteed to already be
    16kHz mono) to a WAV whisper-cli accepts, then transcribe it."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src:
        src.write(wav_bytes)
        src_path = Path(src.name)
    dst_path = src_path.with_suffix(".16k.wav")
    try:
        conv = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src_path), "-ar", "16000", "-ac", "1", str(dst_path)],
            capture_output=True,
            text=True,
        )
        if conv.returncode != 0:
            st.error(f"ffmpeg conversion failed: {conv.stderr.strip()[-500:]}")
            return ""
        result = subprocess.run(
            [whisper_cli, "-m", model_path, "-f", str(dst_path), "-nt", "-np", "-l", "en"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            st.error(f"whisper-cli failed: {result.stderr.strip()[-500:]}")
            return ""
        return result.stdout.strip()
    finally:
        src_path.unlink(missing_ok=True)
        dst_path.unlink(missing_ok=True)


def load_topic_files() -> list[Path]:
    return sorted(TOPICS_DIR.glob("*.txt"))


def start_session(topics: list[dict], shuffle: bool) -> None:
    order = list(range(len(topics)))
    if shuffle:
        random.shuffle(order)
    st.session_state.topics = topics
    st.session_state.topic_order = order
    st.session_state.current_pos = 0
    st.session_state.history = []
    st.session_state.last_audio_id = None
    st.session_state.reply = None


def advance() -> None:
    st.session_state.current_pos += 1
    st.session_state.last_audio_id = None
    st.session_state.reply = None


def main() -> None:
    st.title("🗣️ Explain It Back")
    st.caption(
        "Pick a topic, record your explanation, get a reaction + a sharper "
        "framing -- graded against your own notes, not general knowledge."
    )

    topic_files = load_topic_files()
    if not topic_files:
        st.error(f"No .txt files found in {TOPICS_DIR}. See topics.py --help for the format.")
        return

    with st.sidebar:
        st.header("Setup")
        chosen_file = st.selectbox("Topics file", topic_files, format_func=lambda p: p.name)
        shuffle = st.checkbox("Shuffle topics", value=False)
        if st.button("Start / restart session", type="primary"):
            topics = load_topics(chosen_file)
            if not topics:
                st.error("No topics found in that file.")
            else:
                start_session(topics, shuffle)
                st.rerun()

        with st.expander("📄 Or load topics from a Google Doc"):
            creds = gdoc_picker.get_credentials()
            if not creds:
                if st.button("Sign In with Google", key="gdoc_signin", use_container_width=True):
                    with st.spinner("Waiting for sign-in in your browser..."):
                        gdoc_picker.sign_in()
                    st.rerun()
            else:
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("↻", key="gdoc_refresh", help="Refresh recent docs list"):
                        st.session_state["gdoc_recent"] = gdoc_picker.fetch_recent_docs(creds)
                with c2:
                    if st.button("Sign out", key="gdoc_signout"):
                        gdoc_picker.sign_out()
                        st.rerun()

                if "gdoc_recent" not in st.session_state:
                    st.session_state["gdoc_recent"] = gdoc_picker.fetch_recent_docs(creds)
                docs_list = st.session_state["gdoc_recent"]

                if not docs_list:
                    st.caption("No recent Google Docs found.")
                else:
                    names = [d["name"] for d in docs_list]
                    gdoc_choice = st.selectbox(
                        "Doc",
                        names,
                        index=None,
                        placeholder="Choose a document…",
                        key="gdoc_choice",
                        label_visibility="collapsed",
                    )
                    if gdoc_choice and st.button(
                        "Start from this doc", key="gdoc_start", use_container_width=True
                    ):
                        chosen = next(d for d in docs_list if d["name"] == gdoc_choice)
                        with st.spinner("Loading + parsing..."):
                            text = gdoc_picker.fetch_doc_text(creds, chosen["id"])
                            gdoc_topics = parse_export(text)["topics"]
                        if not gdoc_topics:
                            st.error(
                                "No '#heading' topics found in that doc. See topics.py --help for the format."
                            )
                        else:
                            start_session(gdoc_topics, shuffle)
                            st.rerun()

    if "topics" not in st.session_state:
        st.info("Pick a topics file and click **Start / restart session** in the sidebar.")
        return

    whisper_cli = get_whisper_cli()
    if not whisper_cli:
        st.error("whisper-cli not found on PATH. Install it with `brew install whisper-cpp`.")
        return
    if not Path(DEFAULT_MODEL).exists():
        st.error(f"Model not found: {DEFAULT_MODEL}")
        return

    topics = st.session_state.topics
    order = st.session_state.topic_order
    pos = st.session_state.current_pos

    if pos >= len(order):
        st.success(f"Done -- all {len(order)} topics covered.")
        if st.session_state.history:
            st.subheader("Session log")
            for i, item in enumerate(st.session_state.history, 1):
                with st.expander(f"{i}. {item['topic']}"):
                    st.markdown(f"**Your answer:** {item['answer']}")
                    st.markdown("**Ollama:**")
                    st.write(item["reply"])
        return

    topic = topics[order[pos]]

    st.subheader(f"Topic {pos + 1} of {len(order)}: {topic['name']}")
    st.caption("Explain it out loud, in your own words, like you're teaching it to someone.")

    transcript_key = f"transcript_{pos}"
    if transcript_key not in st.session_state:
        st.session_state[transcript_key] = ""

    audio = st.audio_input("Record your answer", key=f"audio_{pos}")

    if audio is not None and audio.file_id != st.session_state.last_audio_id:
        with st.spinner("Transcribing..."):
            st.session_state[transcript_key] = transcribe_wav_bytes(
                whisper_cli, DEFAULT_MODEL, audio.getvalue()
            )
        st.session_state.last_audio_id = audio.file_id

    answer = st.text_area(
        "Transcript (edit if whisper got something wrong, or just type your answer here instead)",
        key=transcript_key,
        height=120,
    )

    col1, col2 = st.columns(2)
    submit = col1.button("Submit answer", type="primary", disabled=not answer.strip())
    skip = col2.button("Skip topic")

    if skip:
        advance()
        st.rerun()

    if submit and answer.strip():
        with st.spinner("Thinking..."):
            try:
                reply = ask_ollama(topic["name"], topic["notes"], answer.strip())
            except Exception as e:
                st.error(f"Ollama error: {e}")
                reply = None
        if reply:
            st.session_state.reply = reply
            st.session_state.history.append(
                {"topic": topic["name"], "answer": answer.strip(), "reply": reply}
            )

    if st.session_state.reply:
        st.markdown("### Ollama")
        st.code(st.session_state.reply, language=None, wrap_lines=True)
        st.caption("Click the copy icon in the corner of the box above to copy the reply.")
        if st.button("Next topic →"):
            advance()
            st.rerun()


if __name__ == "__main__":
    main()
