#!/usr/bin/env python3
"""
Loads topics from a .txt file to use as practice prompts for transcribe_loop.py.

Same file format as read-nonfiction.html (vercel_flashcards) uses for its
notes export, so you can drop in a file exported straight from that tool, or
just write one by hand:

    === TOPICS ===

    #topic name here
    notes / body text for this topic, any length, until the next # line.

    #another topic
    more notes.

    ##a nested sub-point
    still just parsed as its own entry, at level 2.

The "=== TOPICS ===" envelope is optional -- a bare file that just starts
with "#heading" lines works directly too.

Usage:
    python3 topics.py topics/internet_addiction.txt --list
    python3 topics.py topics/internet_addiction.txt --random
    python3 topics.py topics/internet_addiction.txt --index 0
"""

import argparse
import random
import re
import sys
from pathlib import Path

SECTION_RE = re.compile(r"^===\s*(INTRO|TOPICS)\s*===", re.MULTILINE)
HEADING_RE = re.compile(r"^#+\s*\S", re.MULTILINE)
HEADER_LINE_RE = re.compile(r"^(#+)\s*(.*)$")
BLOCK_SPLIT_RE = re.compile(r"(?=^#+\s*\S)", re.MULTILINE)


def parse_entries(text: str) -> list[dict]:
    """Port of read-nonfiction.html's parseEntries(): split on '#' heading lines."""
    entries = []
    for block in BLOCK_SPLIT_RE.split(text):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        header = HEADER_LINE_RE.match(lines[0])
        if header:
            level = min(len(header.group(1)), 3)
            name = header.group(2).strip()
        else:
            level = 1
            name = lines[0].lstrip("#").strip()
        notes = "\n".join(lines[1:]).strip()
        if not name:
            continue
        entries.append({"name": name, "notes": notes, "level": level})
    return entries


def parse_export(text: str) -> dict:
    """Port of read-nonfiction.html's parseExport(): handle the optional envelope."""
    parts = SECTION_RE.split(text)
    section_map = {}
    for i in range(1, len(parts), 2):
        section_map[parts[i]] = parts[i + 1] if i + 1 < len(parts) else ""

    topics_text = section_map.get("TOPICS")
    if topics_text is None:
        # bare paste, no envelope -- accept it directly if it looks like headings
        if HEADING_RE.search(text):
            topics_text = text

    if topics_text is None:
        return {"topics": []}

    return {
        "intro": section_map.get("INTRO", "").strip(),
        "topics": parse_entries(topics_text),
    }


def load_topics(path: Path) -> list[dict]:
    text = path.read_text()
    return parse_export(text)["topics"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", help="path to a topics .txt file")
    parser.add_argument("--list", action="store_true", help="list all topics with their index")
    parser.add_argument("--random", action="store_true", help="pick a random top-level topic")
    parser.add_argument("--index", type=int, default=None, help="pick topic by index (see --list)")
    parser.add_argument("--show-notes", action="store_true", help="also print the topic's notes (spoils the answer -- use for self-checking after you answer)")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"File not found: {path}")

    topics = load_topics(path)
    if not topics:
        sys.exit("No topics found. Expected '#heading' lines -- see --help for the format.")

    if args.list:
        for i, t in enumerate(topics):
            indent = "  " * (t["level"] - 1)
            print(f"[{i}] {indent}{t['name']}")
        return

    # top-level topics only make good standalone prompts; sub-points (level 2/3)
    # are usually fragments of their parent, so default selection skips them.
    top_level = [t for t in topics if t["level"] == 1] or topics

    if args.index is not None:
        chosen = topics[args.index]
    elif args.random:
        chosen = random.choice(top_level)
    else:
        chosen = top_level[0]

    print(f"Topic: {chosen['name']}")
    print("Explain it out loud, in your own words, like you're teaching it to someone.\n")
    if args.show_notes:
        print("--- notes ---")
        print(chosen["notes"])


if __name__ == "__main__":
    main()
