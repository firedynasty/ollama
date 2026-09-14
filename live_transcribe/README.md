# Live-Feeling Transcription

Mic → Silero VAD → whisper.cpp (`whisper-cli`) → Ollama.

Two scripts:
- `transcribe_loop.py` — step 1, prints each transcribed chunk as you speak.
  No Ollama. Good for tuning VAD thresholds in isolation.
- `interview_loop.py` — the full loop: picks a topic from a topics file,
  listens for your spoken answer, and sends it to Ollama once you press Enter.
- `topics.py` — parses a `#heading` notes file (same format as
  `read-nonfiction.html`) into topics you can practice explaining.

## Requirements (already installed on this machine)

- `whisper-cli` (via `brew install whisper-cpp`)
- Python packages: `torch`, `sounddevice`, `numpy`, `silero-vad` (installed via
  `pip3 install silero-vad` — ships its own bundled model, no network call at
  runtime)

## Run it

```bash
# first, find your real microphone (not "Loopback Audio" / "VB-Cable")
python3 transcribe_loop.py --list-devices

# then run against that device, e.g. the built-in mic:
python3 transcribe_loop.py --device 2
```

Speak, pause for ~0.7s, and the finalized chunk gets transcribed and printed
as soon as whisper-cli finishes (usually well under a second on the M2 with
`ggml-base.bin`).

## Tuning knobs

| Flag | Default | What it does |
|---|---|---|
| `--silence-ms` | 700 | How long a pause has to be before a chunk is finalized. Lower = more responsive but risks cutting you off mid-sentence; higher = feels laggier. |
| `--vad-threshold` | 0.5 | Silero speech-probability threshold (0-1). Raise if it's triggering on background noise; lower if it's missing quiet speech. |
| `--max-chunk-s` | 18 | Safety cap — a long ramble gets force-cut into pieces instead of one giant delayed chunk. |
| `--model` | `ggml-base.bin` | Swap to a bigger model for accuracy (e.g. `~/Library/Application Support/com.prakashjoshipax.VoiceInk/WhisperModels/ggml-large-v3-turbo-q5_0.bin`, already on disk from VoiceInk) once base's accuracy feels limiting. |

Start with `--silence-ms`, since the doc calls it out as the setting most
worth iterating on.

## The Ollama handoff (`interview_loop.py`)

No numeric grading — after each answer it reacts briefly, then shows what a
stronger answer would've included, then moves to the next topic. The system
prompt driving that:

```
You are running a quick "explain it back" practice session. The user is
trying to explain a topic from memory, out loud, and you're given their
spoken answer (transcribed by whisper, so it may contain minor transcription
errors -- don't nitpick those) plus the topic's actual reference notes as
ground truth.

Respond in two parts, tight and conversational, not a score:
1. Briefly react to what they said (1-2 sentences, conversational, not a score).
2. Show what a stronger answer would have included -- a short "here's how you
could've framed that" example, not a lecture, just the key point they missed.

Keep the whole response tight -- this should feel like a real conversation,
not a review.
```

The topic's own notes (from the `#heading` file, via `topics.py`) get sent
alongside your transcribed answer as ground truth — so the reaction is judged
against your actual source material, not general model knowledge.

No silence-based auto-cutoff — **you** decide when an answer is done by
pressing Enter. `--silence-ms` (700ms default) only controls how often a
mid-answer fragment gets transcribed and printed while you're still talking
(keeps it feeling live); it doesn't end the turn. Fragments accumulate into
one full answer until you press Enter — matching the "accumulate then send"
approach for multi-sentence answers.

Enter also doubles as a typed-answer override: type something before
pressing it and that text is sent to Ollama instead of whatever whisper
transcribed — useful for skipping the mic on a topic, or fixing a bad
transcription on the fly. A bare Enter (nothing typed) just finalizes
whatever whisper captured from the mic.

```bash
python3 interview_loop.py topics/internet_addiction.txt --device 2
python3 interview_loop.py topics/internet_addiction.txt --device 2 --shuffle
```

Ollama's reply is copied to the clipboard (`pbcopy`) after every answer, so
you can paste it straight into another app — a TTS tool to have it spoken
back, notes, wherever. Pass `--no-clipboard` to turn that off.

Add topics by writing more `#heading` files under `topics/` (see
`topics.py --help` for the format).

## Next steps (not built yet)

4. Add local TTS (Piper/Kokoro) to speak Ollama's response back.
