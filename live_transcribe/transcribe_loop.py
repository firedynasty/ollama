#!/usr/bin/env python3
"""
Step 1 of the live-feeling transcription pipeline: mic -> Silero VAD -> whisper.cpp -> printed text.

Architecture (see the design doc this implements):
    mic -> VAD (buffer while speaking) -> pause detected -> whisper-cli transcribes
    the finalized chunk -> text printed to terminal.

No Ollama wiring yet on purpose -- get chunk boundaries feeling natural first,
then move to step 3 (wire into Ollama).

Usage:
    python3 transcribe_loop.py                  # list devices if unsure, then run
    python3 transcribe_loop.py --device 2        # use a specific input device index
    python3 transcribe_loop.py --list-devices    # just print devices and exit
"""

import argparse
import queue
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from silero_vad import VADIterator, load_silero_vad

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # required window size for Silero VAD at 16kHz (32ms)
PRE_ROLL_MS = 400  # audio kept before speech-start so the first word isn't clipped

DEFAULT_MODEL = "/opt/homebrew/opt/whisper-cpp/share/whisper-cpp/ggml-base.bin"


def find_whisper_cli() -> str:
    path = shutil.which("whisper-cli")
    if not path:
        sys.exit(
            "whisper-cli not found on PATH. Install it with `brew install whisper-cpp`."
        )
    return path


def write_wav(path: Path, audio_i16: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_i16.tobytes())


def transcribe_chunk(whisper_cli: str, model_path: str, audio_i16: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        write_wav(wav_path, audio_i16)
        result = subprocess.run(
            [
                whisper_cli,
                "-m", model_path,
                "-f", str(wav_path),
                "-nt",  # no timestamps
                "-np",  # no extra prints, just the transcription
                "-l", "en",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[whisper-cli error] {result.stderr.strip()}", file=sys.stderr)
            return ""
        return result.stdout.strip()
    finally:
        wav_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=None, help="input device index (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true", help="print input devices and exit")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="path to a ggml whisper.cpp model")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Silero VAD speech probability threshold (0-1)")
    parser.add_argument("--silence-ms", type=int, default=700, help="ms of silence that ends a chunk (500-1000 is a good start)")
    parser.add_argument("--max-chunk-s", type=float, default=18.0, help="safety cap: force-finalize a chunk after this many seconds")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if not Path(args.model).exists():
        sys.exit(f"Model not found: {args.model}\nPass --model /path/to/ggml-*.bin")

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
        # indata is float32 in [-1, 1]; keep it as-is, convert to int16 only at write time.
        audio_q.put(indata[:, 0].copy())

    pre_roll_frames = int((PRE_ROLL_MS / 1000) * SAMPLE_RATE / FRAME_SAMPLES) + 1
    pre_roll = []  # rolling buffer of recent frames, always maintained
    speech_buffer = []  # frames belonging to the utterance currently being captured
    speaking = False
    speech_started_at = None

    print(f"Model: {args.model}")
    print(f"Silence threshold: {args.silence_ms}ms | Max chunk: {args.max_chunk_s}s")
    print("Listening... (Ctrl+C to stop)\n")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        device=args.device,
        channels=1,
        dtype="float32",
        callback=callback,
    )

    def finalize_chunk(reason: str):
        nonlocal speech_buffer, speaking, speech_started_at
        if not speech_buffer:
            return
        audio_f32 = np.concatenate(speech_buffer)
        audio_i16 = (audio_f32 * 32767).astype(np.int16)
        duration = len(audio_i16) / SAMPLE_RATE
        t0 = time.time()
        text = transcribe_chunk(whisper_cli, args.model, audio_i16)
        elapsed = time.time() - t0
        if text:
            print(f">> {text}    [{duration:.1f}s audio, {elapsed:.2f}s to transcribe, {reason}]")
        speech_buffer = []
        speaking = False
        speech_started_at = None

    with stream:
        try:
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
                    # prepend the pre-roll so the first word isn't clipped
                    speech_buffer = list(pre_roll)

                if event and "end" in event and speaking:
                    finalize_chunk("pause detected")
                    continue

                if speaking and (time.time() - speech_started_at) > args.max_chunk_s:
                    finalize_chunk("max length reached")
        except KeyboardInterrupt:
            print("\nStopping.")


if __name__ == "__main__":
    main()
