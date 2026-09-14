# FAQ — setting this up on your own machine

A local "explain it back" study tool: you pick a topic from your own notes,
say your explanation out loud, and a local LLM (via Ollama) reacts and shows
a sharper framing — graded against *your* notes, not general knowledge.
Nothing leaves your machine.

```
mic → Silero VAD → whisper.cpp (whisper-cli) → you finish your answer → Ollama (qwen3) → reply
```

Two ways to run it: a terminal loop (`interview_loop.py`) or a browser UI
(`streamlit_app.py`). Both share the same topics format, transcription
backend, and Ollama call.

## What do I need installed? (macOS only — uses `pbcopy`, Core Audio device names)

1. **Homebrew** deps:
   ```bash
   brew install whisper-cpp ffmpeg ollama
   ```
   - `whisper-cpp` gives you `whisper-cli` + downloads `ggml-base.bin` to
     `/opt/homebrew/opt/whisper-cpp/share/whisper-cpp/ggml-base.bin` — that
     exact path is hardcoded as the default model, so no extra config needed.
   - `ffmpeg` is only used by `streamlit_app.py`, to convert whatever the
     browser records to 16kHz mono WAV before handing it to whisper-cli.
   - `ollama` runs the model that reacts to your answers.

2. **Pull the model** this project uses:
   ```bash
   ollama pull qwen3:8b
   ```
   Ollama needs to be running (`ollama serve`, or just open the Ollama app)
   before you run either script — both call `http://localhost:11434`.

3. **Python packages**:
   ```bash
   pip install -r requirements.txt
   ```
   (Python 3.11+ recommended.)

## How do I find my microphone?

```bash
python3 transcribe_loop.py --list-devices
```
Look for your actual mic (e.g. "MacBook Air Microphone" or an external one)
— skip anything named "Loopback Audio" / "VB-Cable" / a speaker/output-only
device (those have `0 in`). Note the index number on the left; that's what
you pass as `--device N`.

## How do I try it out first, without the LLM?

```bash
python3 transcribe_loop.py --device N
```
Speak, pause ~0.7s, and see the transcribed chunk print. Good for checking
your mic index and tuning `--vad-threshold` / `--silence-ms` before adding
Ollama into the mix. No topics, no grading — just mic → text.

## How does `interview_loop.py` actually work?

1. Loads topics from a notes file (see "topics file format" below) — every
   `#`, `##`, and `###` heading becomes its own topic/turn.
2. For each topic: prints the topic name, then listens continuously. Silero
   VAD detects speech start/pause; each pause gets transcribed by
   `whisper-cli` and printed as a fragment (`...like this`), but **the turn
   doesn't end on silence** — fragments just accumulate.
3. You press **Enter** when you're done answering. Enter also doubles as a
   typed-answer override: type something before pressing it, and that text
   is sent instead of whatever whisper heard (handy for skipping the mic on
   a topic, or fixing a bad transcription).
4. Your full answer + that topic's own notes (as ground truth) get sent to
   Ollama with a system prompt asking it to (a) briefly react, (b) show what
   a stronger answer would've included — no numeric score.
5. The reply prints and gets copied to your clipboard (`pbcopy`) — pass
   `--no-clipboard` to skip that.
6. Moves to the next topic automatically. `--shuffle` randomizes the order.

```bash
python3 interview_loop.py topics/internet_addiction.txt --device N
python3 interview_loop.py topics/internet_addiction.txt --device N --shuffle
```

## How is `streamlit_app.py` different?

Same idea, adapted to a request/response web UI instead of a continuously
streaming mic:

- Instead of live VAD, you click **Record** (`st.audio_input`), speak, click
  **Stop** — the whole clip is converted with `ffmpeg` and transcribed once.
- The transcript lands in an editable text box before you submit, so you can
  fix whisper's mistakes or just type your answer instead.
- Submitting calls the same `ask_ollama()` used by `interview_loop.py`, so
  the reaction logic (react + sharper framing, no score) is identical.
- Has a sidebar to pick which topics file to use, an optional **shuffle**
  toggle, and a session log at the end showing every answer + reply.
- Also supports loading topics straight from a **Google Doc** instead of a
  local file (see below) — same `#heading` parsing either way.

```bash
streamlit run streamlit_app.py
```

## What's the topics file format?

Plain `.txt` files under `topics/`:
```
#topic name here
notes / body text for this topic, any length, until the next # line.

#another topic
more notes.

##a nested sub-point
still parsed as its own topic/turn -- # / ## / ### all count.
```
The optional `=== TOPICS ===` envelope (with an optional `=== INTRO ===`
section above it) is supported too — it's the same format
`read-nonfiction.html`'s notes export uses, so a file exported from there
drops in directly. See `topics.py --help` for the full spec, or run:
```bash
python3 topics.py topics/internet_addiction.txt --list
```
to preview how a file gets split into topics before using it live.

## Can I use a different Ollama model?

`OLLAMA_MODEL = "qwen3:8b"` is set at the top of `interview_loop.py` (and
reused by `streamlit_app.py` via `ask_ollama`). To switch, edit that
constant and make sure you've pulled the model first (`ollama pull
<model>`). There's no `--model` CLI flag for the Ollama model currently —
only `--model` for the *whisper* model path (see below).

## Can I use a bigger/smaller whisper model?

Yes — `--model /path/to/ggml-*.bin` on either CLI script. Bigger models
(e.g. `ggml-large-v3-turbo-q5_0.bin`) are more accurate but slower. Grab
more models via `whisper-cpp`'s download script or point at any ggml model
you already have on disk.

## Do I need the Google Doc feature?

No — it's optional, only reachable via the "Or load topics from a Google
Doc" expander in `streamlit_app.py`, and everything else works without it.
If you want it: create an OAuth client (type **Desktop app**) in [Google
Cloud Console](https://console.cloud.google.com/), enable the **Drive API**
and **Docs API**, download it, and save it as `client_secret.json` next to
`gdoc_picker.py`. That file (and the `gdoc_token.json` it generates after
you sign in) are both gitignored — never committed, and you'll need your
own copy.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `PortAudioError: Invalid number of channels` | Your `--device` index is an output-only device (0 input channels). Run `--list-devices` and pick one with `>0 in`. |
| `whisper-cli not found on PATH` | `brew install whisper-cpp` |
| `Model not found: /opt/homebrew/...ggml-base.bin` | Re-run/repair `brew install whisper-cpp`, or pass `--model /path/to/your/ggml-*.bin` |
| Ollama request hangs or `ConnectionError` | Ollama isn't running — start the app or run `ollama serve`; confirm with `ollama list` that `qwen3:8b` is pulled |
| `streamlit_app.py`: `ffmpeg conversion failed` | `brew install ffmpeg` |
| "No topics found in that file" | Your `.txt` file has no `#heading` lines — see the format above |

## Everything runs locally?

Yes — whisper.cpp and Ollama both run on-device, nothing is sent anywhere
except (optionally) Google's API if you use the Google Doc import.
