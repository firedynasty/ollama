"""
Google Docs auth + fetch helpers for streamlit_app.py's sidebar -- lets you
load topics directly from a Google Doc instead of a local topics/*.txt file.

A loaded doc is parsed with the same '#heading' rules as topics.py (see its
docstring for the format), so anything exported from read-nonfiction.html, or
just typed directly into a Doc, works as a topic source.

Reuses the OAuth client secret from the streamlit_openai_chat project's
gdoc_picker.py if one hasn't been set up here yet, so you don't need to
re-register an OAuth client per project -- drop your own client_secret.json
next to this file to override that.
"""

import json
from pathlib import Path

import streamlit as st
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/documents.readonly",
]
RECENT_LIMIT = 15

_HERE = Path(__file__).parent
_SHARED_CREDS_DIR = Path("/Users/stanleytan/Documents/technical/python/streamlit_openai_chat")

CLIENT_SECRETS_FILE = _HERE / "client_secret.json"
if not CLIENT_SECRETS_FILE.exists() and (_SHARED_CREDS_DIR / "client_secret.json").exists():
    CLIENT_SECRETS_FILE = _SHARED_CREDS_DIR / "client_secret.json"

TOKEN_FILE = _HERE / "gdoc_token.json"  # cached credentials; keep out of git


# ---------- Auth ----------

def load_cached_credentials():
    if not TOKEN_FILE.exists():
        return None
    info = json.loads(TOKEN_FILE.read_text())
    creds = Credentials.from_authorized_user_info(info, SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        save_credentials(creds)
    return creds


def save_credentials(creds) -> None:
    TOKEN_FILE.write_text(creds.to_json())


def get_credentials():
    creds = st.session_state.get("gdoc_creds")
    if creds and creds.valid:
        return creds
    creds = load_cached_credentials()
    if creds and creds.valid:
        st.session_state["gdoc_creds"] = creds
        return creds
    return None


def sign_in():
    """Opens a browser tab for Google consent, blocks until it completes."""
    if not CLIENT_SECRETS_FILE.exists():
        st.error(
            f"Missing client_secret.json. Create an OAuth client (type 'Desktop app') "
            "in Google Cloud Console, enable the Drive API and Docs API, and download it "
            f"as client_secret.json into {CLIENT_SECRETS_FILE.parent}."
        )
        return None
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS_FILE), SCOPES)
    creds = flow.run_local_server(port=0)
    save_credentials(creds)
    st.session_state["gdoc_creds"] = creds
    return creds


def sign_out() -> None:
    st.session_state.pop("gdoc_creds", None)
    st.session_state.pop("gdoc_recent", None)
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()


# ---------- Drive / Docs ----------

def fetch_recent_docs(creds, limit: int = RECENT_LIMIT) -> list[dict]:
    drive = build("drive", "v3", credentials=creds)
    resp = drive.files().list(
        q="mimeType='application/vnd.google-apps.document' and trashed=false",
        orderBy="viewedByMeTime desc",
        pageSize=limit,
        fields="files(id, name, viewedByMeTime)",
    ).execute()
    return resp.get("files", [])


def doc_content_to_text(content) -> str:
    """Flatten a Docs API body.content structure into plain text."""
    chunks = []
    for element in content or []:
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        for el in paragraph.get("elements", []):
            run = el.get("textRun")
            if run:
                chunks.append(run.get("content", ""))
    return "".join(chunks)


def fetch_doc_text(creds, doc_id: str) -> str:
    docs = build("docs", "v1", credentials=creds)
    doc = docs.documents().get(documentId=doc_id).execute()
    return doc_content_to_text(doc.get("body", {}).get("content"))
