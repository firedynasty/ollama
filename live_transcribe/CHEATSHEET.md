# Live Transcription → Ollama — Quick Reference

## What this is

A local "explain it back" practice loop: you pick a topic from your own
notes, speak your explanation, and Ollama reacts + shows a sharper framing —
no numeric grading. Everything runs locally (whisper.cpp + Silero VAD +
Ollama), nothing leaves your machine.

```
mic → Silero VAD (buffers while you talk) → whisper-cli (transcribes each
pause) → you press Enter when done → Ollama reacts against your own notes →
reply printed + copied to clipboard → next topic
```

## What was built

| File | Purpose |
|---|---|
| `transcribe_loop.py` | Standalone mic → VAD → whisper → printed text. No Ollama. Used to tune VAD/whisper before adding anything else. |
| `topics.py` | Parses a `#heading` notes file into topics (same format as `read-nonfiction.html`). |
| `topics/internet_addiction.txt` | Your first topics file, saved from your pasted notes. |
| `interview_loop.py` | The full loop — mic → whisper → Ollama → clipboard, topic by topic. |

Key decisions along the way:
- **Silero VAD** for voice detection (uses your existing `torch` install, no extra model download)
- **`ggml-base.bin`** via `whisper-cli` for transcription (already on disk from `brew install whisper-cpp`)
- **`qwen3:8b`** via local Ollama for the reaction/reframe
- No silence-based auto-cutoff — **you** end an answer by pressing Enter
- Enter also works as a **typed-answer override**: type text before pressing it to skip the mic for that turn
- Ollama's reply is **copied to the clipboard** after every answer (`pbcopy`), so another app (e.g. TTS) can pick it up

## Commands

```bash
cd /Users/stanleytan/Documents/technical/ollama/live_transcribe

# find your real mic (not "Loopback Audio" / "VB-Cable")
python3 transcribe_loop.py --list-devices

# step 1 — transcription only, no Ollama, for tuning VAD feel
python3 transcribe_loop.py --device 2

# list topics in a notes file
python3 topics.py topics/internet_addiction.txt --list

# print one topic as a standalone prompt (no mic)
python3 topics.py topics/internet_addiction.txt --index 1
python3 topics.py topics/internet_addiction.txt --index 1 --show-notes   # reveal notes after answering

# the full practice loop
python3 interview_loop.py topics/internet_addiction.txt --device 2
python3 interview_loop.py topics/internet_addiction.txt --device 2 --shuffle
python3 interview_loop.py topics/internet_addiction.txt --device 2 --no-clipboard
```

## Tuning flags (interview_loop.py & transcribe_loop.py)

| Flag | Default | What it does |
|---|---|---|
| `--silence-ms` | 700 | Pause that finalizes/transcribes one fragment while you're still talking (feel/responsiveness only — doesn't end the turn). |
| `--vad-threshold` | 0.5 | Silero speech-probability threshold. Raise if noise triggers it; lower if quiet speech gets missed. |
| `--max-chunk-s` | 18 | Safety cap so a long ramble is cut into pieces instead of one giant delayed chunk. |
| `--model` | `ggml-base.bin` | Swap for `ggml-large-v3-turbo-q5_0.bin` (already on disk, from VoiceInk) for more accuracy. |
| `--shuffle` | off | Go through topics in random order (`interview_loop.py` only). |
| `--no-clipboard` | off | Don't copy Ollama's reply to the clipboard. |

## Adding more topics

Write a new `.txt` file under `topics/`, formatted like:

```
#topic name
notes/explanation for this topic, any length, until the next # line.

#another topic
more notes.
```

Then run `interview_loop.py` against that file instead.

## Not built yet

- Local TTS (Piper/Kokoro) to have Ollama's reply spoken back automatically instead of read off the clipboard.
