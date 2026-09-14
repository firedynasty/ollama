# How `create_paraphrase/script.py` works

Reference notes for connecting `ask_ollama1` (Node/Express) to the Python paraphrase script. Source: `/Users/stanleytan/Documents/technical/python/create_paraphrase/script.py`.

## What it does

Takes a `.txt` file (one English line per line) and writes back a paraphrased `.txt` file, one rewritten line per input line. It's a thin wrapper — it does not talk to an LLM itself; it calls into a shared toolkit two directories up the Python side: `language_capabilities/`.

## The call chain

```
script.py
  └─ builds an LLM client (OpenAI SDK, pointed at either OpenAI's API or a local
     server) — the SAME client shape ask_ollama1 could use, see "Connecting" below
  └─ loads the "paraphrase-en" capability definition from
     language_capabilities/capabilities/paraphrase_en.py
       - system_prompt: instructs the model to return
         `[{"paraphrase": "..."}]` — a JSON array with exactly one object
       - input_format: "plain_lines" (no translation pairing, just raw lines)
       - output_columns: ["Original", "Paraphrase"]
       - default model: "qwen3:8b"
  └─ calls language_capabilities/engine.py's run_capability(), which:
       1. reads the input file, splits it into lines (blank lines skipped,
          `[Section]`-style lines kept as passthrough headers)
       2. for EACH line, makes one separate LLM chat-completion call:
            messages = [
              {role: "system", content: <the system_prompt above>},
              {role: "user",   content: <that one line, verbatim>},
            ]
            temperature = 0
       3. parses the JSON array back out of the response (stripping a
          ```-fenced wrapper if the model added one)
       4. maps the JSON's "paraphrase" key onto the "Paraphrase" output
          column, and the original line onto "Original"
       5. if a line's response isn't valid JSON: logs a warning, skips
          just that line, and keeps going — one bad line never aborts the run
       6. writes everything to a CSV (header: Original,Paraphrase)
  └─ script.py then reads that CSV back and writes out a plain .txt containing
     just the "Paraphrase" column, one line per line (this last step is
     script.py's own — the engine only knows how to produce CSVs)
```

## The CLI surface

```bash
python3 script.py --input notes.txt
#   -> writes notes_paraphrased.txt

python3 script.py --input notes.txt --output out.txt \
  --backend openai --model gpt-4o-mini
```

| Flag | Default | Meaning |
|---|---|---|
| `--input` | *(required)* | Path to the source `.txt` |
| `--output` | `<input>_paraphrased.txt` | Where the paraphrased `.txt` goes |
| `--backend` | `local` | `local` = an Ollama-style OpenAI-compatible server; `openai` = OpenAI's cloud API |
| `--base-url` | `http://localhost:11434/v1/` | Only used when `--backend local` |
| `--model` | `qwen3:8b` (the capability's default) | Any model name your chosen backend understands |

Exit behavior: non-zero + a message on stderr if the input file is missing, or (for `--backend openai`) if `OPENAI_API_KEY` isn't set — before any network call is made. Exit 0 once the output file is written, even if some individual lines failed to parse.

## Connecting: how ask_ollama1 (Node) talks to Ollama vs. how script.py does

Both end up hitting the **same local Ollama server**, but through two different client libraries and two different API surfaces it exposes:

| | `ask_ollama1/server.js` | `create_paraphrase/script.py` |
|---|---|---|
| Language | Node.js | Python |
| Client library | `ollama` npm package | `openai` Python SDK |
| Endpoint hit | Ollama's **native** API (`ollama.chat(...)` → `POST /api/chat` under the hood) | Ollama's **OpenAI-compatible** endpoint (`http://localhost:11434/v1/`) |
| Model (currently) | hardcoded `'llama3.2'` in `server.js` | `qwen3:8b` (capability default), overridable with `--model` |

Neither needs to know about the other's client library — Ollama serves both APIs off the same running instance on port 11434. The two things actually worth aligning, if you want them to feel like one system:
1. **Same model name** — right now they default to different models (`llama3.2` vs `qwen3:8b`). Worth deciding on one, at least for anything meant to behave consistently.
2. **Same base URL constant** — both currently hardcode `localhost:11434` in their own way; fine as-is for a single local machine, but worth a shared config file if this ever runs somewhere the port/host isn't the default.

## Options for the actual Node → Python bridge

Since `ask_ollama1` is Node and `script.py` is Python, there's no in-process function call between them — Node has to launch Python as a subprocess (or call an HTTP endpoint if `script.py` were turned into a server, which it currently is not — it's a one-shot CLI). The straightforward option, callable from `server.js`:

```js
// server.js — sketch, not wired in yet
import { execFile } from 'child_process';

app.post('/api/paraphrase-file', (req, res) => {
    const { inputPath, outputPath } = req.body;
    execFile(
        'python3',
        [
            '/Users/stanleytan/Documents/technical/python/create_paraphrase/script.py',
            '--input', inputPath,
            '--output', outputPath,
        ],
        (error, stdout, stderr) => {
            if (error) {
                return res.status(500).json({ error: stderr || error.message });
            }
            res.json({ status: 'ok', output: outputPath, log: stdout });
        }
    );
});
```

This is the minimal bridge: Node shells out to the same `python3 script.py` you'd run by hand, waits for it to exit, and reads the resulting file. It reuses `script.py` exactly as-is — no changes needed on the Python side for this to work. Not wired into `server.js` yet; this is reference for when you want it.
