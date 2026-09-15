#!/usr/bin/env python3
"""
Streamlit UI for chinese_csv.py: paste Chinese text and get a
Chinese/Pinyin/English meaning table -- one row per sentence plus a
word-by-word breakdown -- select cells and copy to paste straight into
Google Sheets as a real table (not comma-separated text), or download as a
.csv.

The browser equivalent of `pbpaste | python3 chinese_csv.py`: paste
whatever you copied (straight out of a PDF viewer, a webpage, wherever) into
the text box instead of piping the clipboard. Same local pipeline as
chinese_csv.py (jieba segmentation + pypinyin + CC-CEDICT, no Ollama, no
network -- see that file's module docstring for how the segmentation/pinyin/
meaning picks are made); this just wraps it in a page.

Usage:
    streamlit run streamlit_app.py
"""

import csv
import io

import pandas as pd
import streamlit as st

from chinese_csv import chinese_text_to_rows, load_cedict


def rows_to_csv(rows: list[list[str]]) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    return buf.getvalue().strip()


st.set_page_config(page_title="Chinese -> CSV", page_icon="🀄")


@st.cache_resource
def get_cedict() -> dict:
    return load_cedict()


def main() -> None:
    st.title("🀄 Chinese -> CSV")
    st.caption(
        "Paste Chinese text and get a Chinese, Pinyin, English meaning "
        "table -- word-by-word breakdown, ready to paste into Google "
        "Sheets. Works on Simplified or Traditional text, and handles a "
        "PDF viewer's mid-word line wraps."
    )

    chinese_text = st.text_area(
        "Chinese text",
        height=250,
        placeholder="你好，很高兴认识你。",
        label_visibility="collapsed",
    )

    convert = st.button("Convert", type="primary", disabled=not chinese_text.strip())

    if convert:
        with st.spinner("Converting..."):
            rows = chinese_text_to_rows(chinese_text.strip(), get_cedict())
        if not rows:
            st.error("No Chinese sentences found in that input.")
        else:
            st.session_state.rows = rows

    rows = st.session_state.get("rows")
    if rows:
        st.markdown("### Table")
        df = pd.DataFrame(rows[1:], columns=rows[0])
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption(
            "Click a cell, drag (or shift-click) to select the range you want, "
            "then ⌘C/Ctrl+C -- pastes straight into Google Sheets as a table. "
            "The hover toolbar in the corner also has a CSV download icon."
        )

        st.download_button("Download .csv", data=rows_to_csv(rows), file_name="chinese_vocab.csv", mime="text/csv")


if __name__ == "__main__":
    main()
