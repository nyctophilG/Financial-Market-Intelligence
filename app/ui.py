"""
ui.py – Financial Market Intelligence
Run from project root: streamlit run ui.py
"""

import os
import sqlite3
import uuid
import time

import streamlit as st

st.set_page_config(
    page_title="Financial Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from main import (
    process_query, save_uploaded_file, list_uploaded_files,
    is_financial_query, get_chroma_doc_count
)

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    background-color: #0d0f14;
    color: #d4dbe8;
}

/* Hide only hamburger + footer. Keep header (contains sidebar arrow). */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111318 !important;
    border-right: 1px solid #1e2330 !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 0 14px 24px 14px !important;
}

/* All sidebar buttons — no borders, full width, left-aligned */
section[data-testid="stSidebar"] button {
    width: 100% !important;
    text-align: left !important;
    background: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    color: #6e7b94 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.82rem !important;
    padding: 7px 10px !important;
    margin: 1px 0 !important;
    box-shadow: none !important;
    transition: background 0.1s, color 0.1s !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
section[data-testid="stSidebar"] button:hover {
    background: #1a1f2e !important;
    color: #c0cce0 !important;
}
section[data-testid="stSidebar"] button:focus {
    box-shadow: none !important;
    outline: none !important;
}

/* New chat button — accented */
.btn-new-chat button {
    background: #162038 !important;
    color: #7eaaee !important;
    border: 1px solid #1d3060 !important;
    font-weight: 600 !important;
    margin-bottom: 4px !important;
}
.btn-new-chat button:hover {
    background: #1e2a4a !important;
    color: #a8c8f8 !important;
}

/* Active conversation row */
.conv-row-active button {
    background: #1b2440 !important;
    color: #dce8ff !important;
}

/* Section label */
.slabel {
    display: block;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #334;
    padding: 14px 2px 5px 2px;
}

/* Route badges */
.badge {
    display: inline-block;
    font-size: 0.62rem;
    font-family: 'IBM Plex Mono', monospace;
    border-radius: 3px;
    padding: 1px 6px;
    margin-bottom: 5px;
    letter-spacing: 0.04em;
}
.badge-ollama { background:#152515; color:#5a9a5a; border:1px solid #254525; }
.badge-crewai { background:#151528; color:#7a8aee; border:1px solid #252548; }
.badge-error  { background:#251515; color:#ee6a6a; border:1px solid #452525; }

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border-bottom: 1px solid #111520 !important;
    padding: 10px 0 !important;
}

/* Divider */
hr { border-color: #1a1f2e !important; margin: 6px 0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: #1e2330; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Database
# ══════════════════════════════════════════════════════════════
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(_THIS_DIR, "chat_history.db")

def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    db = _conn()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT 'New conversation',
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            route TEXT DEFAULT '',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    db.commit(); db.close()

def new_conv(title="New conversation"):
    cid = str(uuid.uuid4())
    db = _conn()
    db.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", (cid, title))
    db.commit(); db.close()
    return cid

def rename_conv(cid, title):
    db = _conn()
    db.execute(
        "UPDATE conversations SET title=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (title[:80], cid)
    )
    db.commit(); db.close()

def del_conv(cid):
    db = _conn()
    db.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
    db.execute("DELETE FROM conversations WHERE id=?", (cid,))
    db.commit(); db.close()

def get_convs():
    db = _conn()
    rows = db.execute(
        "SELECT id, title FROM conversations ORDER BY updated_at DESC"
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]

def add_msg(cid, role, content, route=""):
    db = _conn()
    db.execute(
        "INSERT INTO messages (conversation_id, role, content, route) VALUES (?,?,?,?)",
        (cid, role, content, route)
    )
    db.execute(
        "UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (cid,)
    )
    db.commit(); db.close()

def get_msgs(cid):
    db = _conn()
    rows = db.execute(
        "SELECT role, content, route FROM messages "
        "WHERE conversation_id=? ORDER BY timestamp ASC", (cid,)
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]

def total_msgs():
    db = _conn()
    n = db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    db.close()
    return n


# ══════════════════════════════════════════════════════════════
# Session bootstrap
# ══════════════════════════════════════════════════════════════
init_db()

if "cid" not in st.session_state:
    convs = get_convs()
    st.session_state.cid = convs[0]["id"] if convs else new_conv()

if "show_files"  not in st.session_state: st.session_state.show_files  = False
if "last_error"  not in st.session_state: st.session_state.last_error  = ""
if "ingest_msgs" not in st.session_state: st.session_state.ingest_msgs = []


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:

    # Brand header
    st.markdown("""
    <div style="padding:18px 0 10px 0;">
      <span style="font-size:1rem;font-weight:600;color:#dce8ff;">📈 FinIntel</span><br>
      <span style="font-size:0.6rem;color:#334;font-family:'IBM Plex Mono',monospace;
                   letter-spacing:0.1em;">MULTI-AGENT · RAG · OLLAMA</span>
    </div>
    """, unsafe_allow_html=True)

    # New conversation
    st.markdown('<div class="btn-new-chat">', unsafe_allow_html=True)
    if st.button("＋  New conversation", key="btn_new"):
        st.session_state.cid = new_conv()
        st.session_state.last_error = ""
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Conversations ─────────────────────────────────────────
    st.markdown('<span class="slabel">Conversations</span>', unsafe_allow_html=True)

    convs = get_convs()
    if not convs:
        st.caption("No conversations yet.")
    else:
        for conv in convs:
            is_active = conv["id"] == st.session_state.cid
            title     = conv["title"] or "New conversation"
            display   = title[:40] + "…" if len(title) > 41 else title

            row_class = "conv-row-active" if is_active else "conv-row"
            st.markdown(f'<div class="{row_class}" style="display:flex;align-items:center;gap:2px;">', unsafe_allow_html=True)

            c1, c2 = st.columns([10, 1])
            with c1:
                if st.button(display, key=f"go_{conv['id']}"):
                    st.session_state.cid = conv["id"]
                    st.session_state.last_error = ""
                    st.rerun()
            with c2:
                if st.button("✕", key=f"del_{conv['id']}"):
                    was_active = st.session_state.cid == conv["id"]
                    del_conv(conv["id"])
                    if was_active:
                        remaining = [x for x in convs if x["id"] != conv["id"]]
                        st.session_state.cid = remaining[0]["id"] if remaining else new_conv()
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Document upload ───────────────────────────────────────
    st.markdown('<span class="slabel">Document Upload</span>', unsafe_allow_html=True)
    st.caption("PDF → data/raw   ·   CSV/TXT → data/processed\nFiles are auto-ingested into ChromaDB.")

    uploaded = st.file_uploader(
        "files",
        accept_multiple_files=True,
        type=["pdf", "txt", "csv"],
        label_visibility="collapsed",
    )

    if st.button("💾  Save & Ingest files", key="btn_save"):
        if uploaded:
            msgs = []
            for f in uploaded:
                dest, ingest_msg = save_uploaded_file(f.read(), f.name)
                msgs.append(ingest_msg)
            st.session_state.ingest_msgs = msgs
            st.rerun()
        else:
            st.warning("No files selected.")

    # Show ingestion results
    for msg in st.session_state.ingest_msgs:
        if msg.startswith("✅"):
            st.success(msg)
        else:
            st.warning(msg)

    # File browser
    browse_lbl = "📂  Hide /data" if st.session_state.show_files else "📂  Browse /data"
    if st.button(browse_lbl, key="btn_browse"):
        st.session_state.show_files = not st.session_state.show_files

    if st.session_state.show_files:
        idx = list_uploaded_files()
        if idx["raw"] or idx["processed"]:
            if idx["raw"]:
                st.caption("data/raw/")
                for fn in idx["raw"]:
                    st.markdown(
                        f'<div style="font-size:0.73rem;color:#5a8ad4;font-family:monospace;'
                        f'padding:1px 0 1px 10px;">📄 {fn}</div>',
                        unsafe_allow_html=True
                    )
            if idx["processed"]:
                st.caption("data/processed/")
                for fn in idx["processed"]:
                    st.markdown(
                        f'<div style="font-size:0.73rem;color:#7aaad4;font-family:monospace;'
                        f'padding:1px 0 1px 10px;">📄 {fn}</div>',
                        unsafe_allow_html=True
                    )
        else:
            st.caption("No files uploaded yet.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Stats ─────────────────────────────────────────────────
    st.markdown('<span class="slabel">System Status</span>', unsafe_allow_html=True)
    chroma_count = get_chroma_doc_count()
    chroma_status = f"{chroma_count} chunks" if chroma_count >= 0 else "unavailable"
    st.caption(
        f"💬 {len(get_convs())} chats  ·  {total_msgs()} messages\n\n"
        f"🗄️ ChromaDB: {chroma_status}"
    )

    # Show last error if any
    if st.session_state.last_error:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<span class="slabel">Last error</span>', unsafe_allow_html=True)
        st.error(st.session_state.last_error[:300])


# ══════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════
cid = st.session_state.cid

# Header
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;
            padding:4px 0 16px 0;border-bottom:1px solid #1a1f2e;margin-bottom:16px;">
  <span style="font-size:1.4rem;">📈</span>
  <div>
    <div style="font-size:1.05rem;font-weight:600;color:#e2e8f0;">
      Financial Market Intelligence
    </div>
    <div style="font-size:0.67rem;color:#334;font-family:'IBM Plex Mono',monospace;
                letter-spacing:0.06em;">
      CrewAI · ChromaDB · Ollama llama3.1
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Load history
history = get_msgs(cid)

# Empty state
if not history:
    st.markdown("""
    <div style="text-align:center;padding:48px 0 24px 0;">
      <div style="font-size:2rem;margin-bottom:12px;">📊</div>
      <div style="font-size:0.92rem;color:#2a3450;">
        Ask anything — financial analysis or general conversation
      </div>
      <div style="font-size:0.68rem;color:#1e2a3a;margin-top:8px;font-family:monospace;">
        financial keywords → CrewAI pipeline &nbsp;·&nbsp; everything else → Ollama llama3.1
      </div>
    </div>
    """, unsafe_allow_html=True)

# Render history
for msg in history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("route"):
            badge = "badge-crewai" if msg["route"] == "crewai" else "badge-ollama"
            label = "🤖 crewai" if msg["route"] == "crewai" else "💬 ollama"
            st.markdown(f'<span class="badge {badge}">{label}</span>', unsafe_allow_html=True)
        st.markdown(msg["content"])

# ── Input ──────────────────────────────────────────────────────
prompt = st.chat_input("Ask anything — financial analysis or general questions…")

if prompt:
    if not history:
        rename_conv(cid, prompt[:60].strip())

    with st.chat_message("user"):
        st.markdown(prompt)
    add_msg(cid, "user", prompt)

    with st.chat_message("assistant"):
        is_fin = is_financial_query(prompt)

        if is_fin:
            with st.status("🤖 Running CrewAI pipeline…", expanded=True) as status:
                st.write("🔍 **Data Gatherer** — searching ChromaDB…")
                st.write("_(CrewAI verbose output is printing in your terminal)_")
                result = process_query(prompt)
                if result["error"]:
                    status.update(label="⚠️ Pipeline completed with errors", state="error", expanded=True)
                    st.write(f"**Error:** {result['error'][:300]}")
                else:
                    status.update(label="✅ Pipeline complete", state="complete", expanded=False)
        else:
            with st.spinner("Thinking…"):
                result = process_query(prompt)

        # Badge
        badge = "badge-crewai" if result["route"] == "crewai" else "badge-ollama"
        label = "🤖 crewai" if result["route"] == "crewai" else "💬 ollama"
        st.markdown(f'<span class="badge {badge}">{label}</span>', unsafe_allow_html=True)

        if result["response"]:
            st.markdown(result["response"])
        else:
            st.warning("No response returned. Check terminal for details.")

        # Persist error for sidebar display
        if result["error"]:
            st.session_state.last_error = result["error"]

    add_msg(cid, "assistant", result["response"] or "_(no response)_", route=result["route"])
    st.rerun()
