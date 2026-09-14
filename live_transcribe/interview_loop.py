#!/usr/bin/env python3
"""
Step 3: the full explain-it-back loop.

mic -> Silero VAD -> whisper-cli (per transcribe_loop.py, "accumulate" variant)
    -> press Enter when you're done answering, and the accumulated answer is
       sent to Ollama along with the topic's own notes as ground truth
    -> Ollama reacts (1-2 sentences) and shows what a stronger answer would
       have included -- no numeric grade, just "here's the sharper framing"
    -> moves to the next topic automatically

There's no silence-based auto-cutoff -- you decide when an answer is done by
pressing Enter. --silence-ms only controls how often a mid-answer fragment
gets transcribed and printed while you're still talking (keeps it feeling
live), it does not end the turn.

Enter also doubles as a typed-answer override: if you type something before
pressing Enter, that typed text is sent to Ollama as your answer instead of
whatever whisper transcribed -- useful for skipping the mic on a given topic,
or correcting a bad transcription on the fly. A bare Enter (nothing typed)
just finalizes whatever whisper has captured from the mic so far.

Usage:
    python3 interview_loop.py topics/internet_addiction.txt --device 2
"""

import argparse
import queue
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
from silero_vad import VADIterator, load_silero_vad

from topics import load_topics
from transcribe_loop import (
    DEFAULT_MODEL,
    FRAME_SAMPLES,
    PRE_ROLL_MS,
    SAMPLE_RATE,
    find_whisper_cli,
    transcribe_chunk,
)

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:8b"

SYSTEM_PROMPT = """You are running a quick "explain it back" practice session. \
The user is trying to explain a topic from memory, out loud, and you're given \
their spoken answer (transcribed by whisper, so it may contain minor \
transcription errors -- don't nitpick those) plus the topic's actual reference \
notes as ground truth.

Respond in two parts, tight and conversational, not a score:
1. Briefly react to what they said (1-2 sentences, conversational, not a score).
2. Show what a stronger answer would have included -- a short "here's how you \
could've framed that" example, not a lecture, just the key point they missed.

Keep the whole response tight -- this should feel like a real conversation, \
not a review. Do not just repeat the reference notes verbatim."""


def copy_to_clipboard(text: str) -> None:
    try:
        subprocess.run(["pbcopy"], input=text, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"[clipboard copy failed] {e}", file=sys.stderr)


def ask_ollama(topic_name: str, topic_notes: str, spoken_answer: str) -> str:
    user_msg = (
        f"Topic: {topic_name}\n\n"
        f"Reference notes (ground truth, not seen by the user):\n{topic_notes}\n\n"
        f'My spoken answer:\n"{spoken_answer}"'
    )
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def run_topic(stream_state, topic: dict, whisper_cli: str, model_path: str, args) -> None:
    """Captures one spoken answer for `topic` via mic+VAD+whisper, then grades it."""
    print(f"\n{'=' * 60}")
    print(f"Topic: {topic['name']}")
    print("Explain it out loud, in your own words, then press Enter when done.")
    print("(Or type your answer instead and press Enter -- that skips the mic for this one.)\n")

    enter_q = stream_state["enter_q"]
    while not enter_q.empty():  # drain any stray Enter presses from before this topic started
        enter_q.get_nowait()

    audio_q = stream_state["audio_q"]
    vad_iterator = stream_state["vad_iterator"]
    vad_iterator.reset_states()  # fresh turn

    pre_roll_frames = int((PRE_ROLL_MS / 1000) * SAMPLE_RATE / FRAME_SAMPLES) + 1
    pre_roll = []
    speech_buffer = []
    speaking = False
    speech_started_at = None
    answer_parts = []
    typed_answer = None

    def finalize_fragment(reason: str):
        nonlocal speech_buffer, speaking, speech_started_at
        audio_f32 = np.concatenate(speech_buffer)
        audio_i16 = (audio_f32 * 32767).astype(np.int16)
        text = transcribe_chunk(whisper_cli, model_path, audio_i16)
        if text:
            answer_parts.append(text)
            print(f"  ...{text}")
        speech_buffer = []
        speaking = False
        speech_started_at = None

    while True:
        frame = audio_q.get()
        pre_roll.append(frame)
        if len(pre_roll) > pre_roll_frames:
            pre_roll.pop(0)

        event = vad_iterator(frame, return_seconds=True)

        if speaking:
            speech_buffer.append(frame)

        if event and "start" in event and not speaking:
            speaking = True
            speech_started_at = time.time()
            speech_buffer = list(pre_roll)

        if event and "end" in event and speaking:
            finalize_fragment("pause")
            continue

        if speaking and (time.time() - speech_started_at) > args.max_chunk_s:
            finalize_fragment("max length")

        if not enter_q.empty():
            line = enter_q.get_nowait()
            if line.strip():
                typed_answer = line.strip()
            elif speaking and speech_buffer:
                finalize_fragment("manual done, mid-utterance")
            break

    full_answer = typed_answer if typed_answer is not None else " ".join(answer_parts).strip()
    if not full_answer:
        print("(heard nothing -- skipping)")
        return

    print("\nThinking...")
    try:
        reply = ask_ollama(topic["name"], topic["notes"], full_answer)
    except requests.RequestException as e:
        print(f"[ollama error] {e}", file=sys.stderr)
        return
    print(f"\n{reply}")
    if args.clipboard:
        copy_to_clipboard(reply)
        print("(copied to clipboard)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", help="topics .txt file (see topics.py --help for format)")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="whisper.cpp ggml model path")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--silence-ms", type=int, default=700, help="pause that finalizes one transcribed fragment (does not end the turn -- Enter does)")
    parser.add_argument("--max-chunk-s", type=float, default=18.0)
    parser.add_argument("--shuffle", action="store_true", help="go through topics in random order")
    parser.add_argument("--no-clipboard", dest="clipboard", action="store_false", help="don't copy Ollama's reply to the clipboard after each answer")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"File not found: {path}")
    topics = load_topics(path)
    if not topics:
        sys.exit("No topics found in that file.")
    if args.shuffle:
        random.shuffle(topics)

    if not Path(args.model).exists():
        sys.exit(f"Model not found: {args.model}")
    whisper_cli = find_whisper_cli()

    print("Loading Silero VAD...")
    vad_model = load_silero_vad()
    vad_iterator = VADIterator(
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=args.vad_threshold,
        min_silence_duration_ms=args.silence_ms,
        speech_pad_ms=100,
    )

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[audio status] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    # single persistent stdin reader for the whole run -- each Enter press
    # (with whatever, if anything, was typed before it) lands in this queue.
    # Kept as one thread for the whole run rather than one per topic, so a
    # press can't get consumed by a stale thread left over from a topic that
    # ended some other way.
    enter_q: "queue.Queue[str]" = queue.Queue()

    def read_stdin():
        while True:
            try:
                line = input()
            except EOFError:
                return
            enter_q.put(line)

    threading.Thread(target=read_stdin, daemon=True).start()

    stream_state = {"audio_q": audio_q, "vad_iterator": vad_iterator, "enter_q": enter_q}

    if args.device is not None:
        info = sd.query_devices(args.device)
        if info["max_input_channels"] < 1:
            devices = "\n".join(
                f"  {i}: {d['name']} ({d['max_input_channels']} in, {d['max_output_channels']} out)"
                for i, d in enumerate(sd.query_devices())
            )
            sys.exit(
                f"--device {args.device} ('{info['name']}') has no input channels "
                f"({info['max_input_channels']} in) -- pick a device with inputs:\n{devices}"
            )

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        device=args.device,
        channels=1,
        dtype="float32",
        callback=callback,
    )

    print(f"{len(topics)} topics loaded. Ctrl+C to stop early.")
    with stream:
        try:
            for topic in topics:
                run_topic(stream_state, topic, whisper_cli, args.model, args)
        except KeyboardInterrupt:
            print("\nStopping.")


if __name__ == "__main__":
    main()
