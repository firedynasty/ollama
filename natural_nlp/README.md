# Chinese sentence -> CSV

Feed it Chinese text and it produces a `Chinese,Pinyin,English meaning`
table — one row per sentence, followed by a word-by-word pinyin + meaning
breakdown, grouped the same way translate.google.com groups words (e.g. 慈爱
stays paired, 神 stays single) — for pasting into a vocab/flashcard
spreadsheet. **No Ollama, no LLM, no network call** — pure local Chinese
NLP:

- **jieba** — word segmentation
- **pypinyin** (+ `pypinyin-dict`'s larger phrase dataset) — tone-marked,
  heteronym-aware pinyin (银行 reads `yín háng`, not `yín xíng`)
- **CC-CEDICT** (via `pycccedict`, but re-parsed directly — see
  `chinese_csv.py`'s `load_cedict()` docstring for why) — English
  definitions

Two ways to run it, same underlying pipeline:

- `chinese_csv.py` — CLI
- `streamlit_app.py` — browser UI with a real, copyable table

An earlier version of this asked an LLM (Ollama's `qwen3:8b`) to do the
whole thing, but that was slow (minutes per dialogue) and it occasionally
hallucinated sentences that weren't in the source text. This local pipeline
instead is deterministic, accurate, and takes well under a second.

## CLI (`chinese_csv.py`)

```bash
python3 chinese_csv.py --text "你好，很高兴认识你。"
python3 chinese_csv.py --file dialogue1.txt -o dialogue1.csv
python3 chinese_csv.py --pdf ~/Downloads/elementary_chinese.pdf --page 27 -o lesson1.csv
pbpaste | python3 chinese_csv.py
```

The simplest workflow if copy/paste out of the PDF viewer already gives you
clean text (no need for `--pdf`/`extract_pdf_page.py` at all): select the
text in Preview/whatever PDF viewer, Cmd+C, then `pbpaste | python3
chinese_csv.py` -- it reassembles the PDF viewer's arbitrary mid-word line
wraps back into full sentences before segmenting, using "Speaker: " labels
(if present) to know where one line's wrap ends and a new turn begins.

`--pdf`/`--page` shells out to `extract_pdf_page.py` from the separate
`extract_from_pdf` project (`EXTRACT_PDF_PAGE_SCRIPT` in `chinese_csv.py`,
hardcoded to that project's path on this machine) to pull a page's text out
first (0-indexed page number, same as that script), for when copy/paste
isn't convenient.

Works on both Simplified and Traditional input -- Traditional text is
converted to Simplified just to steer jieba's (simplified-biased)
segmentation, then the original script is put back for display/lookup, so
你好 output stays 你好 and 您好 output stays 您好.

Proper names get a `(name)` gloss (or "Mr./Ms. <Surname> (name)" for a name
+ common title) since they're not in CC-CEDICT; copy is on by default
(`--no-clipboard` to skip).

## Browser UI (`streamlit_app.py`)

```bash
streamlit run streamlit_app.py
```

Same conversion, in a page instead of the terminal -- a text box in place of
`pbpaste | python3 chinese_csv.py`, paste whatever you copied and hit
Convert. The result renders as an actual interactive table
(`st.dataframe`), not a text/CSV block:

- Click a cell, drag (or shift-click) to select a range, then ⌘C/Ctrl+C --
  pastes straight into Google Sheets as a real table (not comma-separated
  text landing in one cell).
- The grid's own hover toolbar (top-right corner) also has a CSV download
  icon, and there's an explicit "Download .csv" button below it too.

## Output format

```
Chinese,Pinyin,English meaning
Dialogue 1,,
你好。,,
你好,nǐ hǎo,hello / hi
你好，很高兴认识你。,,
你好,nǐ hǎo,hello / hi
很,hěn,(adverb of degree) / quite
高兴,gāo xìng,happy / glad
认识,rèn shi,to know / to recognize
你,nǐ,you
```

- First row: `Chinese,Pinyin,English meaning`
- Each sentence gets a `Dialogue N,,` header row, then the full sentence on
  its own row (columns B/C empty), then a word-by-word breakdown with all 3
  columns filled. "Dialogue N" is one per *sentence*, not per literal
  back-and-forth exchange -- merge rows by hand afterward if you want true
  multi-line exchanges grouped together.
- Punctuation doesn't get its own breakdown row.

## Setup

```bash
pip install -r requirements.txt
```

Everything (`jieba`, `pypinyin`, `pypinyin-dict`, `pycccedict`,
`opencc-python-reimplemented`, `streamlit`) is pure Python with bundled data
-- no model downloads, no API keys, no network calls at runtime.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: jieba`/`pypinyin`/`pycccedict`/`opencc` | `pip install -r requirements.txt` |
| `--pdf`: `extract_pdf_page.py not found` | That path is hardcoded to a sibling project on this machine (`extract_from_pdf`) -- edit `EXTRACT_PDF_PAGE_SCRIPT` in `chinese_csv.py` if it's moved, or just pass `--file`/stdin with already-extracted text instead |
| "No Chinese sentences found in the input" | The text had no CJK characters after stripping non-Chinese lines (headers/footers) -- check the input actually contains Chinese |

## Related

- `../live_transcribe/` -- a completely separate, unrelated tool in this
  `ollama` project (mic -> whisper -> Ollama "explain it back" practice
  loop). Shares nothing with this tool except living in the same parent
  directory.
- `extract_pdf_page_to_clipboard.py` in
  `/Users/stanleytan/Documents/technical/python/extract_from_pdf/` and the
  "Extract Chinese PDF Page" Automator Quick Action pair with this tool:
  right-click a PDF -> enter a page number -> text lands on the clipboard ->
  `pbpaste | python3 chinese_csv.py`.
