#!/usr/bin/env python3
"""
Chinese sentence(s) -> CSV (Chinese, Pinyin, English meaning).

Pure local NLP pipeline -- no LLM, no Ollama call, no network, and no
hallucination risk (unlike an earlier version of this script that asked
Ollama to do the whole thing): jieba segments each sentence into words the
same way translate.google.com groups them (e.g. 慈爱 stays paired, 神 stays
single), pypinyin gives tone-marked, phrase-aware pinyin (so heteronyms come
out right -- 银行 reads yín háng, not yín xíng), and CC-CEDICT supplies the
English definitions. Runs in under a second even on a full dialogue.

Output format: full sentence on its own row (columns B/C empty), preceded by
a `Dialogue N,,` header -- one per sentence, not per literal back-and-forth
exchange (merge rows by hand afterward if you want true multi-line
exchanges grouped together) -- then a word-by-word breakdown with all 3
columns filled. Punctuation doesn't get its own breakdown row.

Input sources (pick one):
    --text "..."        inline Chinese text
    --file notes.txt     read Chinese text from a .txt file
    --pdf book.pdf --page N   extract page N (0-indexed) from a PDF via
                          extract_pdf_page.py in the extract_from_pdf repo,
                          then convert that
    (none of the above)  read from stdin, e.g. `pbpaste | chinese_csv.py`

A "Speaker: " label (Latin letters before a colon) at the start of a line is
stripped and used only to mark where one turn ends and the next begins --
handles PDF text where a long line got word-wrapped across multiple raw
lines with no punctuation in between.

Usage:
    python3 chinese_csv.py --text "你好，很高兴认识你。"
    python3 chinese_csv.py --file dialogue1.txt -o dialogue1.csv
    python3 chinese_csv.py --pdf ~/Downloads/elementary_chinese.pdf --page 27 -o lesson1.csv
    pbpaste | python3 chinese_csv.py
"""

import argparse
import csv
import gzip
import io
import logging
import re
import subprocess
import sys
from pathlib import Path

import jieba
import jieba.posseg as pseg
import opencc
import pycccedict.cccedict as _cccedict_module
from pypinyin import Style, pinyin
from pypinyin.contrib.tone_convert import tone3_to_tone

jieba.setLogLevel(logging.WARNING)  # silences "Building prefix dict..." on every run

# jieba's segmentation dictionary is simplified-biased -- fed Traditional text
# directly, it frequently fails to keep multi-character Traditional words
# together (e.g. 請問 splits into 請 + 問, though the Simplified 请问 stays
# merged). Converting to Simplified just to steer segmentation, then mapping
# the resulting word boundaries back onto the original text by character
# offset, gets Simplified-quality segmentation while still displaying (and
# looking up) whatever script the input was actually in.
_TRAD_TO_SIMP = opencc.OpenCC("t2s")

try:
    from pypinyin_dict.phrase_pinyin_data import large_pinyin

    large_pinyin.load()  # bigger phrase-level pinyin dataset -> better heteronym accuracy
except ImportError:
    pass

# extract_pdf_page.py lives in a sibling project, not this repo -- see
# --pdf/--page below, which shells out to it (same python env, since
# pdfminer.six is already installed globally on this machine).
EXTRACT_PDF_PAGE_SCRIPT = Path(
    "/Users/stanleytan/Documents/technical/python/extract_from_pdf/extract_pdf_page.py"
)

CJK_RE = re.compile(r"[一-鿿]")
SPEAKER_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z\s.'-]{0,30}[:：]\s*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？])")

# common titles that turn "<surname><title>" into a name CC-CEDICT won't
# have as a single entry (王先生, 李小姐, ...) -- gives a readable English
# gloss instead of a bare "(name)".
TITLE_MEANINGS = {
    "先生": "Mr.",
    "小姐": "Miss/Ms.",
    "太太": "Mrs.",
    "老师": "Teacher",
    "同学": "classmate",
}


def load_cedict() -> dict:
    """Parses CC-CEDICT ourselves instead of using pycccedict.CcCedict directly --
    that class indexes by simplified/traditional text only and silently keeps just
    the LAST entry it sees per hanzi, which drops alternate readings entirely (e.g.
    呢 has both a [ne5] particle entry and a [ni2] "woolen material" entry; we need
    both to pick the one matching the reading pypinyin chose for a given word)."""
    path = Path(_cccedict_module.__file__).parent / "data" / "cedict_1_0_ts_utf-8_mdbg.txt.gz"
    by_word: dict[str, list[dict]] = {}
    with gzip.open(path, mode="rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            line = line.strip().rstrip("/")
            if not line:
                continue
            chinese, english = line.split("/", 1)
            trad_simp, py = chinese.strip().split("[")
            trad, simp = trad_simp.strip().split()
            py = py[:-1]
            definitions = [d for sense in english.split("/") for d in sense.split(";")]
            entry = {"pinyin": py, "definitions": definitions}
            by_word.setdefault(simp, []).append(entry)
            by_word.setdefault(trad, []).append(entry)
    return by_word


def to_pinyin(word: str) -> str:
    return " ".join(s[0] for s in pinyin(word, style=Style.TONE, heteronym=False))


def _cedict_key(word: str) -> str:
    """Numbered-tone reading (e.g. "ne5") to match against CC-CEDICT's pinyin field."""
    syllables = pinyin(word, style=Style.TONE3, heteronym=False, neutral_tone_with_five=True)
    return " ".join(s[0] for s in syllables).lower()


def _cedict_pinyin_to_marked(py: str) -> str:
    """CC-CEDICT's own pinyin field ("jiao1", space-separated per syllable) to
    tone-marked display form -- used instead of pypinyin's independent guess
    once we've picked a CC-CEDICT entry, so the Pinyin and English meaning
    columns always describe the same reading rather than two different ones."""
    return " ".join(tone3_to_tone(syllable).lower() for syllable in py.split())


def _is_verb_def(entry: dict) -> bool:
    return entry["definitions"][0].strip().lower().startswith("to ")


def _is_proper_noun_entry(entry: dict) -> bool:
    """CC-CEDICT capitalizes the first pinyin syllable to flag a proper-noun
    reading (surname, given name, place name, ...) -- e.g. 高 has both
    "Gao1 surname Gao" and "gao1 high/tall" as separate entries. Checking
    this directly is more general (and more reliable) than pattern-matching
    English gloss text like "surname ...", and it matters: a plain lowercase
    comparison treats "Gao1" and "gao1" as the same reading, so without this
    check a common adjective like 高 ("tall") can silently resolve to its
    surname entry any time it isn't in a name context."""
    return entry["pinyin"][:1].isupper()


def _matches_pos(entry: dict, pos_flag: str) -> bool:
    is_proper = _is_proper_noun_entry(entry)
    if pos_flag == "nr":  # jieba: proper noun / person name
        return is_proper
    if is_proper:  # never let a surname/proper-noun entry win outside a name context
        return False
    if pos_flag.startswith("v"):  # verb, incl. vd/vn subtypes
        return _is_verb_def(entry)
    if pos_flag.startswith("n"):  # noun, incl. ns/nt/nz subtypes (but not nr, handled above)
        return not _is_verb_def(entry)
    return True  # adjective/adverb/etc. -- no strong expectation, any non-proper entry is fine


def cedict_lookup(word: str, pos_flag: str, cedict: dict) -> dict | None:
    """Picks the best CC-CEDICT entry for `word` when it has more than one
    (heteronyms -- different readings/meanings for the same characters).

    Prefers whichever entry matches pypinyin's own contextual reading *if*
    that entry's sense agrees with jieba's POS tag -- pypinyin is usually
    right (it has real sentence context for multi-character words) but its
    default single-character guess ignores POS entirely, so on a word like
    教 (jiao1 "to teach", verb) it defaults to jiao4 ("religion"/"surname
    Jiao"), landing on the wrong entry. Only in that mismatch case do we go
    looking for a different entry whose sense matches the POS tag instead --
    and even then, entries aren't ordered by frequency in the source file
    (说's rare shui4 "to persuade" sense is listed before the common shuo1
    "to speak" sense), so this is a heuristic, not a guarantee."""
    entries = cedict.get(word)
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0]

    target = _cedict_key(word)
    default_entry = next((e for e in entries if e["pinyin"].lower() == target), None)
    if default_entry and _matches_pos(default_entry, pos_flag):
        return default_entry

    pos_matches = [e for e in entries if _matches_pos(e, pos_flag)]
    if pos_matches:
        return pos_matches[0]

    return default_entry or entries[0]


def name_title_meaning(word: str) -> str | None:
    for suffix, title in TITLE_MEANINGS.items():
        if word.endswith(suffix) and len(word) > len(suffix):
            surname_py = to_pinyin(word[: -len(suffix)]).title()
            return f"{title} {surname_py} (name)"
    return None


def select_definitions(entry: dict, pos_flag: str, limit: int = 2) -> list[str]:
    """A single CC-CEDICT entry can bundle multiple distinct senses under one
    reading -- 长/zhang3 lists "chief/head/elder/to grow/to develop/to
    increase/to enhance" as ONE entry, mixing noun and verb senses. When
    jieba tags the word as a verb, prefer whichever of *this entry's own*
    definitions read as a verb ("to ...") instead of just taking the first
    `limit`, which can silently drop the sense that's actually relevant.
    (This can't rescue a case where jieba's POS tag itself is wrong -- e.g.
    长 in a "长得很高" resultative-complement construction gets tagged as a
    plain adjective, not a verb, and there's no clean signal left to
    recognize the "to grow" sense wanted there.)"""
    defs = entry["definitions"]
    if pos_flag.startswith("v"):
        verb_defs = [d for d in defs if d.strip().lower().startswith("to ")]
        if verb_defs:
            return verb_defs[:limit]
    return defs[:limit]


def word_pinyin_and_meaning(word: str, pos_flag: str, prev_word: str | None, cedict: dict) -> tuple[str, str]:
    entry = cedict_lookup(word, pos_flag, cedict)
    if entry:
        return _cedict_pinyin_to_marked(entry["pinyin"]), " / ".join(select_definitions(entry, pos_flag))

    # not in the dictionary at all -- pypinyin's own guess is the best pinyin
    # we can offer; only the meaning needs a fallback below
    py = to_pinyin(word)
    if prev_word == "姓" and len(word) <= 2:
        return py, "(surname)"
    if word.startswith("姓") and len(word) > 1:
        # jieba merged "姓" + a surname into one token (e.g. 姓李) instead of
        # two -- handled separately from the per-character fallback below,
        # since that fallback deliberately can't select a proper-noun/surname
        # CC-CEDICT entry (see its own comment), which would otherwise turn
        # this right back into "family name / plum" for 李.
        return py, f"family name / surname {to_pinyin(word[1:]).title()}"
    if pos_flag == "nr":  # jieba's "proper noun, person name" tag
        return py, name_title_meaning(word) or "(name)"
    if len(word) > 1:  # phrase not in the dict -- fall back to per-character defs
        parts = []
        for ch in word:
            # note: no pos_flag here -- that flag describes the whole
            # multi-character word (jieba doesn't tag individual characters),
            # and passing it down to a single character can wrongly exclude
            # the right reading (姓李 tagged "n" excluded 李's "surname Li"
            # sense, since a surname isn't a plain noun, landing on "plum"
            # instead). "" matches no POS category below, so this just takes
            # pypinyin's own contextual guess for that character.
            ch_entry = cedict_lookup(ch, "", cedict)
            if ch_entry:
                parts.append(ch_entry["definitions"][0])
        if parts:
            return py, " / ".join(parts)
    return py, ""


def extract_turns(text: str) -> list[str]:
    """Joins PDF line-wraps back into one continuous string per speaker turn, using
    a "Speaker: " label to mark where a new turn starts. A bare Chinese line with no
    label (no dialogue, just prose/vocab) starts its own implicit turn. Any line with
    no Chinese in it at all (page headers/footers, "Simplified Chinese" captions,
    page numbers) is dropped."""
    turns: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if SPEAKER_LABEL_RE.match(line):
            if current:
                turns.append("".join(current))
            current = [SPEAKER_LABEL_RE.sub("", line)]
        elif CJK_RE.search(line):
            current.append(line)
    if current:
        turns.append("".join(current))
    return turns


def split_sentences(text: str) -> list[str]:
    sentences = []
    for turn in extract_turns(text):
        for piece in SENTENCE_SPLIT_RE.split(turn):
            piece = piece.strip()
            if piece and CJK_RE.search(piece):
                sentences.append(piece)
    return sentences


def chinese_text_to_rows(text: str, cedict: dict) -> list[list[str]]:
    """Returns the table as plain row data (header row first) -- the CSV
    string (chinese_text_to_csv, below) and the Streamlit app's dataframe
    view (streamlit_app.py) both just format these same rows differently,
    so the row-building logic itself only lives here."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    rows = [["Chinese", "Pinyin", "English meaning"]]
    for i, sentence in enumerate(sentences, 1):
        rows.append([f"Dialogue {i}", "", ""])
        rows.append([sentence, "", ""])

        # segment off the Simplified form (see _TRAD_TO_SIMP above) but keep
        # tracking a character offset so each token's ORIGINAL-script text can
        # be sliced back out of `sentence` -- traditional<->simplified is 1:1
        # per character in the overwhelming majority of cases, but fall back
        # to segmenting the original directly if that ever isn't true here.
        simplified = _TRAD_TO_SIMP.convert(sentence)
        if len(simplified) != len(sentence):
            simplified = sentence

        prev_word = None
        pos = 0
        for word, flag in pseg.cut(simplified):
            display_word = sentence[pos : pos + len(word)]
            pos += len(word)
            if not CJK_RE.search(word):
                continue  # skip punctuation/whitespace tokens
            word_pinyin, meaning = word_pinyin_and_meaning(word, flag, prev_word, cedict)
            rows.append([display_word, word_pinyin, meaning])
            prev_word = word
    return rows


def chinese_text_to_csv(text: str, cedict: dict) -> str:
    rows = chinese_text_to_rows(text, cedict)
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerows(rows)
    return buf.getvalue().strip()


def extract_pdf_page_text(pdf_path: Path, page_number: int) -> str:
    if not EXTRACT_PDF_PAGE_SCRIPT.exists():
        sys.exit(f"extract_pdf_page.py not found at {EXTRACT_PDF_PAGE_SCRIPT}")
    result = subprocess.run(
        [sys.executable, str(EXTRACT_PDF_PAGE_SCRIPT), str(pdf_path), str(page_number)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"extract_pdf_page.py failed: {result.stderr.strip()}")
    text = result.stdout.strip()
    if text.startswith("Error:"):
        sys.exit(text)
    return text


def copy_to_clipboard(text: str) -> None:
    try:
        subprocess.run(["pbcopy"], input=text, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"[clipboard copy failed] {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="inline Chinese text")
    source.add_argument("--file", type=Path, help="read Chinese text from a .txt file")
    source.add_argument("--pdf", type=Path, help="PDF to pull a page of Chinese text from (use with --page)")
    parser.add_argument("--page", type=int, help="page number (0-indexed) to extract, required with --pdf")
    parser.add_argument("-o", "--output", type=Path, help="save CSV to this file instead of just printing it")
    parser.add_argument("--no-clipboard", dest="clipboard", action="store_false", help="don't copy the CSV to the clipboard")
    args = parser.parse_args()

    if args.pdf and args.page is None:
        parser.error("--pdf requires --page")

    if args.text is not None:
        chinese_text = args.text.strip()
    elif args.file is not None:
        if not args.file.exists():
            sys.exit(f"File not found: {args.file}")
        chinese_text = args.file.read_text().strip()
    elif args.pdf is not None:
        if not args.pdf.exists():
            sys.exit(f"File not found: {args.pdf}")
        chinese_text = extract_pdf_page_text(args.pdf, args.page).strip()
    else:
        chinese_text = sys.stdin.read().strip()

    if not chinese_text:
        sys.exit("No Chinese text to convert.")

    cedict = load_cedict()
    csv_text = chinese_text_to_csv(chinese_text, cedict)
    if not csv_text:
        sys.exit("No Chinese sentences found in the input.")

    if args.output:
        args.output.write_text(csv_text + "\n")
        print(f"Saved to {args.output}")
    else:
        print(csv_text)

    if args.clipboard:
        copy_to_clipboard(csv_text)
        print("(copied to clipboard)", file=sys.stderr)


if __name__ == "__main__":
    main()
