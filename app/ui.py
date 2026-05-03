"""
ui.py  —  lives at <project_root>/app/ui.py
Run from project root:  streamlit run app/ui.py

Upload flow (two-phase, no blocking):
  Phase 1 — button click:  write raw bytes to disk instantly, st.rerun()
  Phase 2 — next render:   embed into ChromaDB with st.spinner (no pending widget reruns)
"""

import os
import sqlite3
import uuid
import threading

import streamlit as st

st.set_page_config(
    page_title="Financial Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from main import (
    process_query,
    ingest_to_chroma,
    list_uploaded_files,
    is_crewai_query,
    get_chroma_doc_count,
    refresh_chroma_count,
    RAW_DIR,
    PROCESSED_DIR,
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
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

section[data-testid="stSidebar"] {
    background-color: #0f1116 !important;
    border-right: 1px solid #1a1f2c !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 0 12px 24px 12px !important;
}
section[data-testid="stSidebar"] button {
    width: 100% !important;
    text-align: left !important;
    background: transparent !important;
    border: none !important;
    border-radius: 5px !important;
    color: #5a6880 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.81rem !important;
    padding: 6px 10px !important;
    margin: 1px 0 !important;
    box-shadow: none !important;
    transition: background 0.1s, color 0.1s !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
section[data-testid="stSidebar"] button:hover {
    background: #181d28 !important;
    color: #b8c8e0 !important;
}
section[data-testid="stSidebar"] button:focus {
    box-shadow: none !important;
    outline: none !important;
}
.btn-newchat button {
    background: #111d30 !important;
    color: #6aa0e8 !important;
    border: 1px solid #1c3050 !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    margin-bottom: 2px !important;
}
.btn-newchat button:hover { background: #182440 !important; color: #90c0f8 !important; }
.conv-active button { background: #171f34 !important; color: #dce8ff !important; }
.slabel {
    display: block;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #2a3040;
    padding: 16px 2px 5px 2px;
}
.btn-ingest button {
    background: #0e1f10 !important;
    color: #4a9a5a !important;
    border: 1px solid #1a3020 !important;
    font-weight: 600 !important;
}
.btn-ingest button:hover { background: #152818 !important; color: #6ab87a !important; }
.badge {
    display: inline-block;
    font-size: 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    border-radius: 3px;
    padding: 1px 6px;
    margin-bottom: 6px;
    letter-spacing: 0.05em;
}
.b-ollama { background:#0e1e0e; color:#4a9a5a; border:1px solid #1a3020; }
.b-crewai { background:#0e0e1e; color:#6a7aee; border:1px solid #1a1a40; }
[data-testid="stChatMessage"] {
    background: transparent !important;
    border-bottom: 1px solid #0f1318 !important;
    padding: 10px 0 !important;
}
[data-testid="stStatusWidget"] {
    background: #0a0d14 !important;
    border: 1px solid #1a1f2c !important;
    border-radius: 6px !important;
    font-size: 0.79rem !important;
}
hr { border-color: #161b24 !important; margin: 8px 0 !important; }
.chroma-pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 10px;
    border-radius: 10px;
    margin-top: 4px;
}
.chroma-ok    { background:#0e1e12; color:#4a9a60; border:1px solid #1a3020; }
.chroma-empty { background:#1e100e; color:#9a5040; border:1px solid #3a1a10; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: #1a1f2c; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Database
# ══════════════════════════════════════════════════════════════
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(_APP_DIR, "chat_history.db")

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
    db.execute("INSERT INTO conversations (id,title) VALUES (?,?)", (cid, title))
    db.commit(); db.close()
    return cid

def rename_conv(cid, title):
    db = _conn()
    db.execute("UPDATE conversations SET title=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (title[:80], cid))
    db.commit(); db.close()

def del_conv(cid):
    db = _conn()
    db.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
    db.execute("DELETE FROM conversations WHERE id=?", (cid,))
    db.commit(); db.close()

def get_convs():
    db = _conn()
    rows = db.execute("SELECT id,title FROM conversations ORDER BY updated_at DESC").fetchall()
    db.close()
    return [dict(r) for r in rows]

def add_msg(cid, role, content, route=""):
    db = _conn()
    db.execute("INSERT INTO messages (conversation_id,role,content,route) VALUES (?,?,?,?)", (cid,role,content,route))
    db.execute("UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (cid,))
    db.commit(); db.close()

def get_msgs(cid):
    db = _conn()
    rows = db.execute(
        "SELECT role,content,route FROM messages WHERE conversation_id=? ORDER BY timestamp ASC", (cid,)
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]

def total_msgs():
    db = _conn()
    n = db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    db.close()
    return n


# ══════════════════════════════════════════════════════════════
# Session state bootstrap
# ══════════════════════════════════════════════════════════════
init_db()

if "cid"             not in st.session_state:
    convs = get_convs()
    st.session_state.cid = convs[0]["id"] if convs else new_conv()

if "show_files"      not in st.session_state: st.session_state.show_files      = False
if "ingest_log"      not in st.session_state: st.session_state.ingest_log      = []
if "staged_paths"    not in st.session_state: st.session_state.staged_paths    = []
if "ingest_running"  not in st.session_state: st.session_state.ingest_running  = False
if "chroma_count"    not in st.session_state:
    st.session_state.chroma_count = get_chroma_doc_count()


# ══════════════════════════════════════════════════════════════
# PHASE 2 — Embedding (runs at top of render BEFORE sidebar)
# Must be outside sidebar block so st.spinner renders in main area
# ══════════════════════════════════════════════════════════════
if st.session_state.ingest_running and st.session_state.staged_paths:
    paths = st.session_state.staged_paths
    names = ", ".join(os.path.basename(p) for p in paths)

    with st.spinner(f"⏳ Embedding into ChromaDB: {names} — please wait, do not navigate away…"):
        log     = {}
        lock    = threading.Lock()

        def _embed_one(path):
            msg = ingest_to_chroma(path)
            with lock:
                log[path] = msg

        threads = [threading.Thread(target=_embed_one, args=(p,), daemon=True) for p in paths]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    result_log = [log.get(p, f"❌ {os.path.basename(p)}: unknown error") for p in paths]
    st.session_state.ingest_log     = result_log
    st.session_state.chroma_count   = refresh_chroma_count()
    st.session_state.ingest_running = False
    st.session_state.staged_paths   = []
    st.rerun()


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:

    st.markdown("""
    <div style="padding:18px 0 10px 0;">
      <span style="font-size:0.98rem;font-weight:600;color:#dce8ff;">📈 FinIntel</span><br>
      <span style="font-size:0.58rem;color:#2a3040;font-family:'IBM Plex Mono',monospace;
                   letter-spacing:0.12em;">CREWAI · CHROMADB · OLLAMA</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="btn-newchat">', unsafe_allow_html=True)
    if st.button("＋  New conversation", key="btn_new"):
        st.session_state.cid        = new_conv()
        st.session_state.ingest_log = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<span class="slabel">Conversations</span>', unsafe_allow_html=True)
    convs = get_convs()
    if not convs:
        st.caption("No conversations yet.")
    else:
        for conv in convs:
            is_active = conv["id"] == st.session_state.cid
            raw_t     = conv["title"] or "New conversation"
            disp      = raw_t[:40] + "…" if len(raw_t) > 41 else raw_t
            if is_active:
                st.markdown('<div class="conv-active">', unsafe_allow_html=True)
            c1, c2 = st.columns([10, 1])
            with c1:
                if st.button(disp, key=f"go_{conv['id']}"):
                    st.session_state.cid = conv["id"]
                    st.rerun()
            with c2:
                if st.button("✕", key=f"d_{conv['id']}"):
                    was = st.session_state.cid == conv["id"]
                    del_conv(conv["id"])
                    if was:
                        rem = [x for x in convs if x["id"] != conv["id"]]
                        st.session_state.cid = rem[0]["id"] if rem else new_conv()
                    st.rerun()
            if is_active:
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Document upload ───────────────────────────────────────
    st.markdown('<span class="slabel">Upload Documents</span>', unsafe_allow_html=True)

    n_docs = st.session_state.chroma_count
    if n_docs > 0:
        st.markdown(
            f'<span class="chroma-pill chroma-ok">🗄 ChromaDB: {n_docs} chunks ready</span>',
            unsafe_allow_html=True,
        )
    elif n_docs == 0:
        st.markdown(
            '<span class="chroma-pill chroma-empty">🗄 ChromaDB: empty — upload files below</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="chroma-pill chroma-empty">🗄 ChromaDB: unavailable</span>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=["pdf", "txt", "csv"],
        label_visibility="collapsed",
        key="file_uploader",
    )

    if uploaded:
        st.markdown(
            f'<div style="font-size:0.74rem;color:#3a5060;padding:3px 0 5px 2px;">'
            f'{len(uploaded)} file(s) staged — click Save & Ingest</div>',
            unsafe_allow_html=True,
        )

    # ── PHASE 1: write bytes to disk, trigger phase 2 ─────────
    st.markdown('<div class="btn-ingest">', unsafe_allow_html=True)
    if st.button("✅  Save & Ingest into ChromaDB", key="btn_ingest"):
        if uploaded:
            staged = []
            for f in uploaded:
                ext      = os.path.splitext(f.name)[1].lower()
                dest_dir = RAW_DIR if ext == ".pdf" else PROCESSED_DIR
                dest     = os.path.join(dest_dir, f.name)
                with open(dest, "wb") as fh:
                    fh.write(f.read())
                staged.append(dest)
                print(f"[Save] Written: {dest}")

            # Hand off to phase 2 (top of next render)
            st.session_state.staged_paths   = staged
            st.session_state.ingest_running = True
            st.rerun()   # rerun immediately — no embedding yet, no blocking
        else:
            st.warning("No files selected.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Ingestion results
    for msg in st.session_state.ingest_log:
        if msg.startswith("✅"):
            st.success(msg, icon="✅")
        else:
            st.warning(msg)

    browse_lbl = "📂 Hide /data" if st.session_state.show_files else "📂 Browse /data"
    if st.button(browse_lbl, key="btn_browse"):
        st.session_state.show_files = not st.session_state.show_files

    if st.session_state.show_files:
        idx = list_uploaded_files()
        if idx["raw"] or idx["processed"]:
            if idx["raw"]:
                st.caption("data/raw/  (PDFs)")
                for fn in idx["raw"]:
                    st.markdown(
                        f'<div style="font-size:0.72rem;color:#4a7aba;font-family:monospace;'
                        f'padding:1px 0 1px 10px;">📄 {fn}</div>',
                        unsafe_allow_html=True,
                    )
            if idx["processed"]:
                st.caption("data/processed/  (CSV/TXT)")
                for fn in idx["processed"]:
                    st.markdown(
                        f'<div style="font-size:0.72rem;color:#6a9ad4;font-family:monospace;'
                        f'padding:1px 0 1px 10px;">📄 {fn}</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No files uploaded yet.")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<span class="slabel">Routing Guide</span>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.68rem;color:#2a3a50;line-height:1.7;">
      <span style="color:#6a7aee;">●</span>
      <b style="color:#3a4a70;">CrewAI</b> — "analyze", "revenue",<br>
      <span style="color:#2a3a60;">&nbsp;&nbsp;"from the file", "net sales",<br>
      &nbsp;&nbsp;SEC terms, Q3 2024, tickers…</span><br><br>
      <span style="color:#4a9a5a;">●</span>
      <b style="color:#3a5a40;">Ollama</b> — everything else
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption(f"💬 {len(get_convs())} chats · {total_msgs()} messages")


# ══════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════
cid = st.session_state.cid

st.markdown("""
<div style="display:flex;align-items:center;gap:12px;
            padding:4px 0 14px 0;border-bottom:1px solid #161b24;margin-bottom:14px;">
  <span style="font-size:1.3rem;">📈</span>
  <div>
    <div style="font-size:1.02rem;font-weight:600;color:#e0e8f8;">
      Financial Market Intelligence
    </div>
    <div style="font-size:0.65rem;color:#2a3040;font-family:'IBM Plex Mono',monospace;
                letter-spacing:0.06em;">
      CrewAI agents · ChromaDB · Ollama llama3.1
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

history = get_msgs(cid)

if not history:
    n    = st.session_state.chroma_count
    hint = f"🗄 {n} chunks in ChromaDB — ready for document analysis" if n > 0 \
           else "📂 Upload & ingest files on the left, then ask about them"
    st.markdown(f"""
    <div style="text-align:center;padding:44px 0 20px 0;">
      <div style="font-size:1.8rem;margin-bottom:12px;">📊</div>
      <div style="font-size:0.9rem;color:#2a3450;margin-bottom:8px;">
        Ask anything — financial document analysis or general conversation
      </div>
      <div style="font-size:0.68rem;color:#1e2840;font-family:monospace;">{hint}</div>
      <div style="font-size:0.65rem;color:#1a2030;margin-top:10px;line-height:1.8;">
        Use <code style="color:#4a5a70;">"analyze"</code>,
        <code style="color:#4a5a70;">"revenue"</code>,
        <code style="color:#4a5a70;">"from the file"</code> → CrewAI pipeline
      </div>
    </div>
    """, unsafe_allow_html=True)

for msg in history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("route"):
            cls = "b-crewai" if msg["route"] == "crewai" else "b-ollama"
            lbl = "🤖 crewai" if msg["route"] == "crewai" else "💬 ollama"
            st.markdown(f'<span class="badge {cls}">{lbl}</span>', unsafe_allow_html=True)
        st.markdown(msg["content"])

prompt = st.chat_input(
    "Ask anything… 'analyze', 'revenue', 'from the file' → CrewAI  |  anything else → Ollama"
)

if prompt:
    if not history:
        rename_conv(cid, prompt[:60].strip())

    with st.chat_message("user"):
        st.markdown(prompt)
    add_msg(cid, "user", prompt)

    with st.chat_message("assistant"):
        use_crewai = is_crewai_query(prompt)

        if use_crewai:
            with st.status("🤖 CrewAI pipeline running…", expanded=True) as status:
                st.write("🔍 **Data Gatherer** — searching ChromaDB…")
                st.write("📊 **Financial Analyst** — structuring report…")
                st.write("🛡️ **Risk Monitor** — fact-checking…")
                st.caption("_Verbose output printing in your terminal._")
                result = process_query(prompt)
                if result.get("error") and result["error"] not in ("empty_output", "no_docs"):
                    status.update(label="⚠️ Completed with errors", state="error", expanded=True)
                else:
                    status.update(label="✅ Analysis complete", state="complete", expanded=False)
        else:
            with st.spinner("Thinking…"):
                result = process_query(prompt)

        cls = "b-crewai" if result["route"] == "crewai" else "b-ollama"
        lbl = "🤖 crewai" if result["route"] == "crewai" else "💬 ollama"
        st.markdown(f'<span class="badge {cls}">{lbl}</span>', unsafe_allow_html=True)
        st.markdown(result["response"] or "_(no response)_")

    add_msg(cid, "assistant", result["response"] or "_(no response)_", route=result["route"])
    st.rerun()
