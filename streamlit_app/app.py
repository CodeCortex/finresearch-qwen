from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st


st.set_page_config(
    page_title="FinResearch-Qwen",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
RAW_DIR = Path(os.getenv("RAW_DIR", "data/raw/company_reports"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "result" not in st.session_state:
    st.session_state.result = None
if "recent" not in st.session_state:
    st.session_state.recent = []

dark = st.session_state.theme == "dark"

C = {
    "bg": "#070b14" if dark else "#f5f8fc",
    "panel": "#0f1726" if dark else "#ffffff",
    "panel2": "#111c2e" if dark else "#f8fafc",
    "text": "#f8fafc" if dark else "#0f172a",
    "muted": "#94a3b8" if dark else "#64748b",
    "line": "rgba(148,163,184,.16)" if dark else "rgba(15,23,42,.10)",
    "accent": "#60a5fa" if dark else "#2563eb",
    "green": "#34d399" if dark else "#059669",
    "red": "#fb7185" if dark else "#e11d48",
    "shadow": "rgba(0,0,0,.30)" if dark else "rgba(15,23,42,.08)",
}



st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: "DM Sans", sans-serif;
    }}

    .stApp {{
        background:
          radial-gradient(circle at 5% 0%, rgba(96,165,250,.10), transparent 28%),
          radial-gradient(circle at 95% 5%, rgba(45,212,191,.08), transparent 25%),
          {C["bg"]};
        color: {C["text"]};
    }}

    .block-container {{
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 4rem;
    }}

    [data-testid="stSidebar"] {{
        background: {C["panel"]};
        border-right: 1px solid {C["line"]};
    }}

    .hero {{
        padding: 38px 42px;
        border-radius: 28px;
        color: white;
        margin-bottom: 22px;
        border: 1px solid rgba(255,255,255,.12);
        background:
          radial-gradient(circle at 88% 15%, rgba(45,212,191,.22), transparent 25%),
          radial-gradient(circle at 65% 90%, rgba(167,139,250,.18), transparent 28%),
          linear-gradient(135deg,#07111f,#111c34 55%,#0b2630);
        box-shadow: 0 28px 80px rgba(0,0,0,.22);
    }}

    .hero-kicker {{
        color: #93c5fd;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }}

    .hero h1 {{
        font-family: Manrope, sans-serif;
        font-size: clamp(38px, 4vw, 58px);
        line-height: 1;
        letter-spacing: -2px;
        margin: 0 0 14px;
    }}

    .hero p {{
        color: #cbd5e1;
        max-width: 900px;
        font-size: 16px;
        line-height: 1.7;
        margin: 0;
    }}

    .badges {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 20px;
    }}

    .badge {{
        padding: 7px 11px;
        border-radius: 999px;
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.14);
        color: #e2e8f0;
        font-size: 11px;
        font-weight: 700;
    }}

    .stat, .card, .source, .status {{
        background: {C["panel"]};
        border: 1px solid {C["line"]};
        box-shadow: 0 10px 34px {C["shadow"]};
    }}

    .stat {{
        border-radius: 18px;
        padding: 16px;
        min-height: 92px;
    }}

    .stat-icon {{ font-size: 18px; }}
    .stat-value {{
        font-family: Manrope, sans-serif;
        font-size: 24px;
        font-weight: 800;
        color: {C["text"]};
    }}
    .stat-label {{
        color: {C["muted"]};
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: .8px;
    }}

    .section-title {{
        font-family: Manrope, sans-serif;
        color: {C["text"]};
        font-size: 21px;
        font-weight: 800;
        margin: 10px 0 5px;
    }}

    .section-subtitle, .muted {{
        color: {C["muted"]};
        font-size: 12px;
    }}

    .card {{
        border-radius: 20px;
        padding: 20px;
    }}

    .answer {{
        background: {C["panel"]};
        border: 1px solid {C["line"]};
        border-left: 4px solid {C["accent"]};
        border-radius: 18px;
        padding: 24px 27px;
        color: {C["text"]};
        font-size: 16px;
        line-height: 1.8;
        box-shadow: 0 12px 36px {C["shadow"]};
    }}

    .source {{
        border-radius: 16px;
        padding: 15px 17px;
        margin: 9px 0;
    }}

    .source-title {{
        color: {C["text"]};
        font-weight: 800;
        font-size: 13px;
    }}

    .source-meta {{
        color: {C["muted"]};
        font-size: 10px;
        margin-top: 5px;
    }}

    .source-snippet {{
        color: {C["muted"]};
        font-size: 12px;
        line-height: 1.6;
        margin-top: 9px;
    }}

    .step {{
        background: {C["panel2"]};
        border: 1px solid {C["line"]};
        border-radius: 13px;
        padding: 11px;
        margin-bottom: 8px;
    }}

    .step b {{ color: {C["text"]}; font-size: 12px; }}
    .step span {{ color: {C["muted"]}; font-size: 10px; }}

    .status {{
        border-radius: 14px;
        padding: 11px 13px;
        display: flex;
        gap: 9px;
        align-items: center;
    }}

    .dot {{
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: {C["green"]};
    }}

    .dot.off {{ background: {C["red"]}; }}

    .profile {{
        background: {C["panel"]};
        border: 1px solid {C["line"]};
        border-radius: 17px;
        padding: 16px;
    }}

    .profile a {{
        color: {C["accent"]};
        text-decoration: none;
        font-size: 11px;
        font-weight: 700;
        margin-right: 10px;
    }}

    .history {{
        background: {C["panel2"]};
        border: 1px solid {C["line"]};
        border-radius: 11px;
        padding: 9px 11px;
        color: {C["muted"]};
        font-size: 11px;
        margin-bottom: 6px;
    }}

    div.stButton > button {{
        border-radius: 12px !important;
        min-height: 42px;
        font-weight: 700 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def esc(value: Any) -> str:
    return html.escape(str(value))


def api_get(path: str):
    return requests.get(f"{API_URL}{path}", timeout=10)


def api_post(path: str, payload: dict, timeout: int = 600):
    return requests.post(
        f"{API_URL}{path}",
        json=payload,
        timeout=timeout,
    )


def health():
    try:
        response = api_get("/health")
        response.raise_for_status()
        return True, response.json()
    except Exception:
        return False, {}


def developer():
    try:
        response = api_get("/v1/developer")
        response.raise_for_status()
        return response.json()
    except Exception:
        return {"developer": "FinResearch-Qwen", "contact": {}}


connected, health_data = health()
indexed = int(health_data.get("indexed_chunks", 0)) if health_data else 0


# ============================================================
# Hero
# ============================================================

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">● AI Financial Research Platform</div>
      <h1>FinResearch-Qwen</h1>
      <p>
        Evidence-grounded financial research powered by Qwen3-4B,
        QLoRA, semantic retrieval and ChromaDB. Analyze company reports,
        financial statements and research documents with traceable evidence.
      </p>
      <div class="badges">
        <span class="badge">Qwen3-4B + QLoRA</span>
        <span class="badge">4-bit NF4</span>
        <span class="badge">RAG + ChromaDB</span>
        <span class="badge">PDF / TXT / MD</span>
        <span class="badge">FastAPI</span>
        <span class="badge">Streamlit</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Metrics
# ============================================================

a, b, c, d = st.columns(4)

with a:
    st.markdown(
        f'<div class="stat"><div class="stat-icon">◉</div>'
        f'<div class="stat-value">{"Online" if connected else "Offline"}</div>'
        f'<div class="stat-label">API Status</div></div>',
        unsafe_allow_html=True,
    )

with b:
    st.markdown(
        f'<div class="stat"><div class="stat-icon">▦</div>'
        f'<div class="stat-value">{indexed:,}</div>'
        f'<div class="stat-label">Indexed Chunks</div></div>',
        unsafe_allow_html=True,
    )

with c:
    st.markdown(
        '<div class="stat"><div class="stat-icon">⌁</div>'
        '<div class="stat-value">Semantic</div>'
        '<div class="stat-label">Retrieval</div></div>',
        unsafe_allow_html=True,
    )

with d:
    st.markdown(
        '<div class="stat"><div class="stat-icon">✦</div>'
        '<div class="stat-value">QLoRA</div>'
        '<div class="stat-label">Financial LLM</div></div>',
        unsafe_allow_html=True,
    )

st.write("")


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## Control Center")
    st.caption("Manage the financial research engine.")

    # new_theme = st.radio(
    #     "Appearance",
    #     ["dark", "light"],
    #     index=0 if dark else 1,
    #     horizontal=True,
    # )
    
    new_theme = st.radio(
            "Appearance",
            ["dark"],
            horizontal=True,
        )

    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.divider()

    st.markdown("### System Health")

    st.markdown(
        f"""
        <div class="status">
          <div class="dot {' ' if connected else 'off'}"></div>
          <div>
            <b style="color:{C['text']};">
              {"API Online" if connected else "API Offline"}
            </b><br>
            <span style="color:{C['muted']};font-size:10px;">
              {esc(API_URL)} · {indexed:,} chunks
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    if st.button("Refresh health", use_container_width=True):
        st.rerun()

    st.divider()

    st.markdown("### Knowledge Base")

    uploaded = st.file_uploader(
        "Upload financial document",
        type=["pdf", "txt", "md"],
        help="Annual reports, filings, investor presentations or research notes.",
    )

    if uploaded:
        target = RAW_DIR / Path(uploaded.name).name
        try:
            target.write_bytes(uploaded.getbuffer())
            st.success(f"Saved {uploaded.name}")
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

    if st.button(
        "Index Knowledge Base",
        type="primary",
        use_container_width=True,
    ):
        if not connected:
            st.error("Start FastAPI before indexing.")
        else:
            with st.spinner("Parsing, chunking and embedding documents..."):
                try:
                    response = api_post(
                        "/v1/index",
                        {
                            "directory": str(RAW_DIR),
                            "chunk_size": 1200,
                            "chunk_overlap": 200,
                            "clear": False,
                        },
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.success(
                        f"Indexed {result.get('chunks', 0):,} chunks "
                        f"from {result.get('documents', 0)} documents."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

    st.caption(f"Directory: {RAW_DIR}")

    st.divider()

    st.markdown("### Retrieval")

    top_k = st.slider(
        "Evidence sources",
        1,
        10,
        5,
    )

    use_rag = st.toggle(
        "Use RAG",
        value=True,
    )

    st.divider()

    dev = developer()
    contact = dev.get("contact", {})

    st.markdown(
        f"""
        <div class="profile">
          <b style="color:{C['text']};">{esc(dev.get("developer", "FinResearch-Qwen"))}</b>
          <div class="muted">GenAI · RAG · LLM Fine-tuning · Backend</div>
          <br>
          <a href="{esc(contact.get("linkedin", "#"))}" target="_blank">LinkedIn</a>
          <a href="{esc(contact.get("github", "#"))}" target="_blank">GitHub</a>
          <a href="{esc(contact.get("instagram", "#"))}" target="_blank">Instagram</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        """
        <div class="muted">
        <b>Research-only system.</b><br>
        Outputs are generated from model knowledge and retrieved documents.
        They are not personalized investment advice and should be independently verified.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Main layout
# ============================================================

left, right = st.columns([1.65, 1], gap="large")


with left:
    st.markdown(
        '<div class="section-title">Research Workspace</div>'
        '<div class="section-subtitle">'
        'Ask a financial question and receive an evidence-grounded answer.'
        '</div>',
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "Research question",
        placeholder=(
            "Example: What was the change in operating margin "
            "between FY2025 and FY2026? Show the calculation and cite evidence."
        ),
        height=145,
        label_visibility="collapsed",
        value=st.session_state.get("prefill", ""),
    )

    run_col, clear_col = st.columns([4, 1])

    with run_col:
        run = st.button(
            "Run Financial Research  →",
            type="primary",
            use_container_width=True,
        )

    with clear_col:
        clear = st.button(
            "Clear",
            use_container_width=True,
        )

    if clear:
        st.session_state.result = None
        st.session_state.prefill = ""
        st.rerun()

    if run:
        if len(question.strip()) < 3:
            st.warning("Enter a more specific research question.")
        elif not connected:
            st.error(f"FastAPI is unavailable at {API_URL}.")
        else:
            with st.spinner(
                "Retrieving evidence and generating financial analysis..."
            ):
                try:
                    response = api_post(
                        "/v1/research",
                        {
                            "question": question.strip(),
                            "use_rag": use_rag,
                            "top_k": top_k,
                        },
                    )
                    response.raise_for_status()
                    st.session_state.result = response.json()

                    q = question.strip()
                    st.session_state.recent = [
                        q,
                        *[
                            x
                            for x in st.session_state.recent
                            if x.lower() != q.lower()
                        ],
                    ][:6]

                except requests.HTTPError:
                    try:
                        detail = response.json().get("detail", str(response.text))
                    except Exception:
                        detail = response.text
                    st.error(f"Research request failed: {detail}")
                except Exception as exc:
                    st.error(f"Research request failed: {exc}")

    result = st.session_state.result

    if result:
        st.markdown("### Answer")

        answer = esc(
            result.get(
                "answer",
                "No answer returned.",
            )
        ).replace("\n", "<br>")

        st.markdown(
            f'<div class="answer">{answer}</div>',
            unsafe_allow_html=True,
        )

        sources = result.get("sources", [])

        st.markdown(
            f"### Retrieved Evidence · {len(sources)} sources"
        )

        if not sources:
            st.info("No RAG sources were returned.")
        else:
            for i, source in enumerate(sources, 1):
                metadata_source = source.get("source", "unknown")
                file_name = source.get(
                    "file_name",
                    Path(metadata_source).name if metadata_source else "unknown",
                )
                page = source.get("page")
                score = source.get("score", "N/A")
                snippet = source.get("snippet", "")

                page_label = (
                    f"Page {page}"
                    if page not in (None, -1, "N/A")
                    else "Document"
                )

                score_label = (
                    f"{float(score):.3f}"
                    if isinstance(score, (int, float))
                    else "N/A"
                )

                st.markdown(
                    f"""
                    <div class="source">
                      <div class="source-title">
                        {i}. {esc(file_name)}
                      </div>
                      <div class="source-meta">
                        {esc(page_label)} · Retrieval score {esc(score_label)}
                        · {esc(metadata_source)}
                      </div>
                      <div class="source-snippet">
                        {esc(snippet)}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            """
            <div class="card">
              <div style="font-size:30px;">⌁</div>
              <h3 style="margin:7px 0;">Start your financial research</h3>
              <div class="muted">
                Ask about financial statements, ratios, company performance,
                risks, growth or information contained in indexed documents.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("### Example Research Questions")

        examples = [
            "What was the revenue growth between FY2023 and FY2024?",
            "What are the major financial risks mentioned in the report?",
            "Calculate the operating margin and show the calculation.",
            "Compare revenue and net profit across the two years.",
        ]

        for i, example in enumerate(examples):
            if st.button(
                example,
                key=f"example_{i}",
                use_container_width=True,
            ):
                st.session_state.prefill = example
                st.rerun()



with right:
    st.markdown(
        '<div class="section-title">AI Research Pipeline</div>',
        unsafe_allow_html=True,
    )

    steps = [
        ("01", "Ingest", "PDF, TXT and Markdown"),
        ("02", "Chunk", "Overlapping retrieval chunks"),
        ("03", "Embed", "Sentence Transformer vectors"),
        ("04", "Retrieve", "ChromaDB semantic search"),
        ("05", "Generate - Finetune Model", "Qwen3-4B + LoRA"),
    ]

    for number, title, description in steps:
        st.markdown(
            f"""
            <div class="step">
              <b>{number} · {title}</b><br>
              <span>{description}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Architecture")

    st.code(
        """User
  ↓
Streamlit
  ↓
FastAPI
  ↓
Query Embedding
  ↓
ChromaDB
  ↓
Top-K Evidence
  ↓
Qwen3-4B + LoRA
  ↓
Answer + Sources""",
        language="text",
    )

    st.markdown("### System Components")

    st.markdown(
        """
        <div class="card">
          <b>LLM</b>
          <div class="muted">Qwen3-4B + QLoRA</div>
          <br>
          <b>Retrieval</b>
          <div class="muted">Sentence Transformers + ChromaDB</div>
          <br>
          <b>Documents</b>
          <div class="muted">PDF · TXT · Markdown</div>
          <br>
          <b>Backend</b>
          <div class="muted">FastAPI + Pydantic</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.recent:
        st.markdown("### Recent Research")
        for item in st.session_state.recent:
            st.markdown(
                f'<div class="history">{esc(item)}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Developer Tools")

    st.markdown(
        f"""
        <div class="card">
          <a href="{esc(API_URL + "/docs")}" target="_blank">Swagger API</a>
          <a href="{esc(API_URL + "/health")}" target="_blank">Health</a>
          <a href="{esc(API_URL + "/openapi.json")}" target="_blank">OpenAPI</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    f"""
    <div style="
      margin-top:35px;
      border-top:1px solid {C['line']};
      padding-top:18px;
      text-align:center;
      color:{C['muted']};
      font-size:10px;">
      FinResearch-Qwen · Qwen3-4B + QLoRA + RAG · Financial Research Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)
