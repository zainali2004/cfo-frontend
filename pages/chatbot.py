"""
FrontEnd/pages/chatbot.py
Data 2 Insight — Chatbot POC (primary entry point).

Users reach this page via FrontEnd/main.py, which immediately redirects here.
The sidebar is hidden with CSS so the app presents as a clean, single-page
application with no Streamlit navigation chrome.

Page structure
--------------
  [Top section]   Upload + Sector hints + Key metrics + Data preview
  [Tab 1]         Sector Detector — runs Agents 1 & 2 on demand
  [Tab 2]         Graphs          — auto-runs full Agent 1→5 chain silently
  [Tab 3]         AI Chatbot      — GPT-4o-mini Q&A, available after Graphs run

Implementation phases
---------------------
  Phase 2.1  imports, set_page_config, API helpers, sidebar hide CSS
  Phase 2.2  session state initialisation + _reset_state()
  Phase 2.3  top section UI
  Phase 2.4  Tab 1 — Sector Detector
  Phase 2.5  Tab 2 — Graphs
  Phase 2.6  Tab 3 — AI Chatbot
"""

# ============================================================
# PHASE 2.1 — Imports, Page Config & Shared API Helpers
# ============================================================

# Standard library
import base64

# Third-party — must be imported before any st.* call so that
# urllib3 warnings are suppressed before requests are made.
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import re

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# st.set_page_config MUST be the very first Streamlit call in the script.
# Any other st.* call before this raises a StreamlitAPIException.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Data 2 Insight — Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Hide sidebar completely ────────────────────────────────────────────────────
# Removes the sidebar panel and the «/» collapse-toggle button so the app
# presents as a clean, full-width single-page application.
# Both selectors are needed: one for the sidebar container, one for the button
# that appears when the sidebar is collapsed.
st.markdown(
    """
    <style>
    [data-testid="stSidebar"]       { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# API configuration
#
# Copied from main.py (not imported) because main.py is a Streamlit app, not
# a module.  Importing it would execute every top-level widget call at import
# time, causing duplicate rendering.
#
# Future improvement: extract to FrontEnd/api_client.py once a third page
# is added, eliminating the duplication.
# ---------------------------------------------------------------------------
API_BASE: str = "https://cfo-backend-de7r.onrender.com" # DEPLOYED PRODUCTION URL
#API_BASE: str = "http://localhost:8000"                 # LOCAL DEVELOPMENT URL

# REQUESTS_VERIFY = False:
# Disables SSL certificate verification for loopback calls.
# Required on corporate proxy networks (e.g. Deloitte) where a TLS inspection
# proxy intercepts localhost traffic and presents a self-signed certificate.
# This is safe for localhost — these are internal loopback requests only.
REQUESTS_VERIFY: bool = False


def api_post(path: str, **kwargs):
    """
    POST to the FastAPI backend and return the parsed JSON response.

    Returns None and calls st.error() on any HTTP or network failure,
    allowing the caller to check `if result:` rather than catching exceptions.

    Parameters
    ----------
    path : str
        API path relative to API_BASE, e.g. "/upload" or "/chat".
    **kwargs
        Additional keyword arguments forwarded to requests.post()
        (e.g. json=payload, files=files).

    Returns
    -------
    dict | None
        Parsed JSON response on success, None on any failure.
    """
    kwargs.setdefault("verify", REQUESTS_VERIFY)
    try:
        resp = requests.post(f"{API_BASE}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = exc.response.text or str(exc)
        st.error(f"API error ({path}): {detail}")
        return None
    except Exception as exc:
        st.error(f"Request failed ({path}): {exc}")
        return None


# ============================================================
# PHASE 2.2 — Session State Initialisation
# ============================================================

# ---------------------------------------------------------------------------
# Default values for all session state keys.
#
# Using a mapping (key → default) instead of a scattered series of
# `if key not in st.session_state` checks means:
#   • Adding a new key requires one entry here (and in _reset_state below)
#   • The full state surface area is visible in one place
# ---------------------------------------------------------------------------
_STATE_DEFAULTS: dict = {
    # Internal — file identity for change detection ("filename|size")
    # Prefixed with _ to signal it is infrastructure, not UI data.
    "_file_id": None,

    # Upload result from POST /upload — set by Phase 2.3
    "upload_result": None,

    # Agent outputs — populated during pipeline execution (Phases 2.4, 2.5)
    "profiles":       None,   # Agent 1: data profile list
    "domain_info":    None,   # Agent 2: sector/domain dict
    "concepts":       None,   # Agent 3: KPI list with calculated values
    "insights":       None,   # Agent 4: multi-category insight dict
    "rendered_visuals": None, # Agent 5: list of {insight_text, image_b64, ...}

    # Pipeline completion flag — gates the Chatbot tab (Phase 2.6)
    "pipeline_done": False,

    # Chat conversation history — grows with each turn in Phase 2.6
    # Each element: {"role": "user"|"assistant", "content": str}
    "chat_history": [],

    # Pending suggested question — set by the suggested-question button click
    # and consumed (and cleared) on the very next script re-run at the main
    # tab level, avoiding rendering issues from calling _process_chat() inside
    # a column-scoped button handler.
    "sq_pending": None,
}


def _init_state() -> None:
    """
    Ensure every session state key exists with its default value.
    Called once at module level on every script re-run.
    Skips keys that are already set so existing values survive re-runs.
    """
    for key, default in _STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _reset_state() -> None:
    """
    Reset all pipeline and chat state when a new file is detected.

    Called by the file-change detection block in Phase 2.3.
    `_file_id` itself is NOT reset here — the caller updates it after
    calling this function with the new file's identity.

    The defaults are sourced from _STATE_DEFAULTS to keep a single
    source of truth.  Any future state key added to _STATE_DEFAULTS is
    automatically covered here.
    """
    for key, default in _STATE_DEFAULTS.items():
        if key == "_file_id":
            continue                   # caller manages this
        # Use a fresh copy of mutable defaults (e.g. [] for chat_history)
        st.session_state[key] = list(default) if isinstance(default, list) else default


# Run state initialisation at module level.
# This executes on every Streamlit re-run and is idempotent — it only
# sets keys that do not yet exist.
_init_state()


# ============================================================
# PHASE 2.3 — Helper: Data Preview Renderer
# ============================================================

def _render_data_preview(result: dict) -> None:
    """
    Render a contextual data preview based on the uploaded file type.

    Covers all 6 file types supported by the backend:
      CSV / Excel     — tabular dataframe preview
      PDF             — table count banner + detected financial tables
      TXT             — table count banner + raw text preview
      MSG + structured attachment  — email body + attachment dataframes
      MSG + text attachment        — email body + attachment text
      MSG body only                — email body + any detected tables

    Parameters
    ----------
    result : dict
        Parsed response from POST /upload (UploadResponse JSON).
        Must contain at least a "file_type" key.

    Notes
    -----
    Extracted from main.py's inline preview block (lines 91-155) into a
    named function so the top-section flow remains compact and readable,
    and so this function can be moved to utils_frontend.py if a third
    page needs it.
    """
    file_type = result.get("file_type")

    # ── CSV / Excel ────────────────────────────────────────────────────────
    if file_type in ("csv", "excel"):
        st.subheader(f"Data Preview ({file_type.upper()})")
        records = result.get("records", [])
        if records:
            df_preview = pd.DataFrame(records[:10])
            # use_container_width=True exploits the wide layout setting
            st.dataframe(df_preview, use_container_width=True)

    # ── PDF / MSG / TXT ────────────────────────────────────────────────────
    elif file_type in ("pdf", "msg", "txt"):
        label = {
            "pdf": "PDF",
            "msg": "Outlook email (.msg)",
            "txt": "Plain text (.txt)",
        }.get(file_type, file_type.upper())

        tables_json = result.get("tables_json", []) or []
        atts        = result.get("attachments_preview") or []

        if file_type == "msg":
            att_table_total = sum(a.get("table_count", 0) for a in atts)
            total_tables    = len(tables_json) + att_table_total
            st.success(
                f"✅ Processed {label}  |  "
                f"{len(atts)} attachment(s)  |  "
                f"{total_tables} table(s) detected."
            )

            # Email body preview
            body_text = result.get("extracted_text", "")
            with st.expander("Email Body Preview", expanded=False):
                preview = body_text[:2000] + ("..." if len(body_text) > 2000 else "")
                # Use a code block so whitespace/newlines are preserved without
                # needing unsafe_allow_html — clean text arrives from the backend.
                st.markdown(f"```\n{preview}\n```")

            # Attachment previews — one entry per attachment
            if atts:
                with st.expander(f"Attachments ({len(atts)})", expanded=True):
                    for att in atts:
                        primary_badge = (
                            " 🔑 **(primary data source)**" if att.get("is_primary") else ""
                        )
                        st.markdown(
                            f"**{att['name']}** — `{att['type']}`{primary_badge}"
                        )
                        if att["type"] in ("csv", "excel"):
                            recs     = att.get("records", [])
                            cols_att = att.get("columns", [])
                            st.caption(
                                f"{att.get('rows', '?')} rows × "
                                f"{att.get('cols', '?')} columns"
                            )
                            if recs:
                                st.dataframe(
                                    pd.DataFrame(recs, columns=cols_att or None),
                                    use_container_width=True,
                                )
                        else:
                            preview = att.get("text_preview", "")
                            if att.get("table_count"):
                                st.caption(f"Tables detected: {att['table_count']}")
                            st.text(preview)
                        st.divider()

        else:
            # PDF or TXT
            st.success(f"✅ Detected {len(tables_json)} table(s) from {label}.")
            if file_type == "txt":
                extracted = result.get("extracted_text", "")
                with st.expander("Text File Preview", expanded=False):
                    st.text(extracted[:3000] + ("..." if len(extracted) > 3000 else ""))

        # Detected financial tables — shown for PDF, TXT, and MSG with tables
        dfs_records = result.get("dfs_records", {}) or {}
        if dfs_records:
            with st.expander("Detected Financial Tables Preview", expanded=False):
                for tname, rows in dfs_records.items():
                    st.subheader(tname)
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # MSG with a structured (CSV/Excel) attachment — also show its data rows
        if file_type == "msg" and result.get("records"):
            with st.expander("Primary Attachment Data Preview", expanded=False):
                df_att = pd.DataFrame(result["records"][:10])
                st.dataframe(df_att, use_container_width=True)


# ============================================================
# PHASE 2.3 — Top Section: Title, Inputs, Upload & Preview
# ============================================================

st.title("💬 Data 2 Insight — Chatbot")

# ── User inputs (widget state — no session state needed) ──────────────────
# These local variables are in scope throughout the rest of the script,
# accessible inside all three tab blocks below.  Streamlit's own widget
# state system persists their values across re-runs without any explicit
# session_state entries.
user_hints  = st.text_area("Sector Knowledge / Hints (Optional)")
key_metrics = st.text_area("Key Metrics / Metrics to calculate (Optional)")

uploaded_file = st.file_uploader(
    "Upload CSV, Excel, PDF, Outlook Email (.msg) or Plain Text (.txt)",
    type=["csv", "xlsx", "xls", "pdf", "msg", "txt"],
)

# ── File-change detection ──────────────────────────────────────────────────
# Uses the _file_id strategy from Phase 2.2.
# Computes a lightweight identity string (name + byte size) for the current
# upload.  If it differs from the stored identity, all pipeline state is
# reset before processing the new file — correctly handles the case where the
# user swaps files mid-session (which main.py's simpler check does not catch).
if uploaded_file is not None:
    file_id = f"{uploaded_file.name}|{uploaded_file.size}"

    if st.session_state["_file_id"] != file_id:
        # New or changed file — purge all agent outputs and chat history
        _reset_state()
        st.session_state["_file_id"] = file_id

    # ── POST /upload — idempotent: fires once per unique file ──────────────
    # After the first successful call the result is stored in session state.
    # Subsequent re-runs (tab clicks, button presses) skip this block and
    # read the cached result instead.
    if st.session_state["upload_result"] is None:
        with st.spinner("Processing file..."):
            upload_data = api_post(
                "/upload",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                },
            )
        if upload_data:
            st.session_state["upload_result"] = upload_data

    # ── Render data preview from session state ─────────────────────────────
    # Always read from session state (not the local upload_data variable) so
    # that the preview is shown even on re-runs where the API was not called.
    result = st.session_state.get("upload_result")
    if result:
        _render_data_preview(result)

else:
    # No file uploaded yet
    st.info(
        "Upload a file to begin. "
        "Use the **Sector Detector** and **Graphs** tabs to analyse your data, "
        "then ask questions in the **AI Chatbot** tab."
    )


# ============================================================
# PHASE 2.4 — Helper: Sector Result Renderer
# ============================================================

def _render_sector_result(domain_info: dict) -> None:
    """
    Render the Agent 2 sector/domain detection result.

    Extracted to a named function because the same display is needed in
    two places within Tab 1:
      • "Already detected" state — shown on tab open when domain_info is cached
      • "Just completed" state  — shown immediately after the API call returns

    Parameters
    ----------
    domain_info : dict
        The 'final' sub-dict from the POST /sector response.
        Expected keys: domain, definition, subdomain (optional),
        wiki_url (optional), confidence (optional).
    """
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown(f"**🏢 Sector:** {domain_info.get('domain', 'N/A')}")
        if domain_info.get("subdomain"):
            st.markdown(f"**🔍 Subsector:** {domain_info['subdomain']}")
        defn = domain_info.get("definition", "")
        if defn:
            st.markdown(f"**📖 Definition:** {defn}")

    with col_right:
        conf = domain_info.get("confidence", "")
        if conf:
            st.metric("Confidence", conf)
        wiki = domain_info.get("wiki_url", "")
        if wiki:
            st.markdown(f"[📚 Wikipedia]({wiki})")


def _render_kpi_table(concepts: list) -> None:
    """Render the calculated KPIs in a cleanly formatted Pandas DataFrame table."""
    if not concepts:
        return

    table_data = []
    for c in concepts:
        name = c.get("concept_phrase", "")
        if not name:
            continue
        
        val = c.get("calculated_value")
        if val is None or str(val).strip().lower() in ("", "none", "null", "needs_data"):
            val = "N/A"
            
        meaning = c.get("why_it_matters", c.get("business_relevance", ""))
        
        table_data.append({
            "KPI Name": name,
            "Value": val,
            "Meaning": meaning
        })
        
    if table_data:
        st.markdown("### 🔢 Calculated KPIs")
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)


# ============================================================
# PHASE 2.4 — Tabs Scaffold + Tab 1: Sector & KPIs
# ============================================================

# Create all three tab headers at once.
# The content for each tab is filled in by the three `with tab:` blocks
# below (and in Phases 2.5 and 2.6).  Streamlit renders all tab headers
# here and lazily renders only the active tab's content.
tab1, tab2, tab3 = st.tabs(["📊 Sector & KPIs", "📈 Graphs", "💬 AI Chatbot"])


with tab1:
    st.header("Sector & KPIs")
    st.caption(
        "Automatically identifies the financial sector and calculates Key Performance Indicators (KPIs) "
        "for your uploaded data."
    )

    upload_result = st.session_state.get("upload_result")

    if not upload_result:
        # Guard: top section hasn't processed a file yet
        st.info("⬆️ Upload a file first using the uploader above.")

    else:
        # ── Show cached result if sector/KPI detection was already run ───
        cached_domain = st.session_state.get("domain_info")
        cached_concepts = st.session_state.get("concepts")

        if cached_domain:
            st.success("✅ Sector detected")
            _render_sector_result(cached_domain)
            
        if cached_concepts:
            st.success("✅ KPIs calculated")
            _render_kpi_table(cached_concepts)
            
        if cached_domain or cached_concepts:
            st.divider()

            # Warn the user if re-running won't regenerate the graphs
            if st.session_state.get("pipeline_done"):
                st.warning(
                    "⚠️ The Graphs pipeline has already run using the previous "
                    "results. Re-running will update the context for the **AI Chatbot** but **won't regenerate the charts**. "
                    "Switch to the **Graphs** tab and click Generate Graphs to refresh."
                )

        # ── Button (label changes when a result already exists) ───────────
        btn_label = (
            "🔄 Re-run Analysis" if (cached_domain and cached_concepts) else "🔍 Analyze Sector & Calculate KPIs"
        )

        if st.button(btn_label, key="btn_sector_kpi"):

            # ── Step 1: Agent 1 — Data Profiler ───────────────────────────
            # Profiles are a prerequisite for the sector detector.
            # We always re-profile here (rather than reusing cached profiles)
            # so that a re-run reflects any changes to user_hints that might
            # affect the profiler's context clues.
            with st.spinner("Loading..."):
                profile_data = api_post(
                    "/profile",
                    json={
                        "raw_preview":    upload_result.get("raw_preview"),
                        "extracted_text": upload_result.get("extracted_text"),
                    },
                )

            if profile_data:
                st.session_state["profiles"] = profile_data.get("profiles", [])

                # ── Step 2: Agent 2 — Sector Detector ─────────────────────
                with st.spinner("Loading..."):
                    sector_data = api_post(
                        "/sector",
                        json={
                            # Agents 3/4/5 expect the "final" sub-dict, so we
                            # store only that — same approach as main.py line 267.
                            "data_profile":   st.session_state["profiles"],
                            "memory":         {},
                            "extracted_text": upload_result.get("extracted_text"),
                            "user_hints":     user_hints,   # widget var from Phase 2.3
                        },
                    )

                if sector_data:
                    # Store only the "final" sub-dict (not candidates / follow-up qs)
                    st.session_state["domain_info"] = sector_data.get(
                        "final", sector_data
                    )
                    
                    # ── Step 3: Agent 3 — KPI Calculator ──────────────────────
                    with st.spinner("Loading..."):
                        user_metrics_list = (
                            [m.strip() for m in key_metrics.split(",") if m.strip()]
                            if key_metrics else []
                        )
                        metrics_payload: dict = {
                            "domain_info":  st.session_state["domain_info"],
                            "data_profile": st.session_state["profiles"],
                            "memory":       {},
                            "user_metrics": user_metrics_list,
                            "key_metrics":  key_metrics,
                        }
                        
                        # Apply identical mapping logic to pass files appropriately into Agent 3 
                        if upload_result.get("dfs_records"):
                            metrics_payload["pdf_dfs_records"] = upload_result["dfs_records"]
                        elif upload_result.get("records"):
                            metrics_payload["df_records"]  = upload_result["records"]
                            metrics_payload["df_columns"]  = upload_result.get("columns", [])
                        if upload_result.get("extracted_text"):
                            metrics_payload["extracted_text"] = upload_result["extracted_text"]

                        metrics_data = api_post("/metrics", json=metrics_payload)
                        
                    if metrics_data:
                        st.session_state["concepts"] = metrics_data.get("concepts", [])
                        
                        st.success("✅ Analysis complete")
                        _render_sector_result(st.session_state["domain_info"])
                        _render_kpi_table(st.session_state["concepts"])


# ============================================================
# PHASE 2.5 — Helpers: Pipeline Runner & Visual Renderer
# ============================================================

def _run_pipeline(upload_result: dict, user_hints: str, key_metrics: str) -> bool:
    """
    Execute all 5 agents sequentially with a live progress bar.

    Each step calls the corresponding FastAPI endpoint and stores the result
    in session state.  Returns False immediately on any step failure,
    short-circuiting the chain without using st.stop() (which would stop
    Tab 3 from rendering too).

    Parameters
    ----------
    upload_result : dict
        Parsed POST /upload response — source of raw_preview, extracted_text,
        records, dfs_records, etc.  Passed in rather than read from session
        state directly so the function is easier to test in isolation.
    user_hints : str
        Sector / domain hints entered by the user above the tabs.
    key_metrics : str
        Comma-separated KPIs the user wants calculated.

    Returns
    -------
    bool
        True if all 5 steps completed successfully; False otherwise.
        On False, at least one st.error() will already have been rendered
        by api_post(), so no additional error message is needed here.
    """
    bar = st.progress(0, text="Loading...")

    # ── Step 1: Agent 1 — Data Profiler ───────────────────────────────────
    bar.progress(5, text="Loading...")
    profile_data = api_post(
        "/profile",
        json={
            "raw_preview":    upload_result.get("raw_preview"),
            "extracted_text": upload_result.get("extracted_text"),
        },
    )
    if not profile_data:
        return False
    profiles = profile_data.get("profiles", [])
    st.session_state["profiles"] = profiles
    bar.progress(20, text="Loading...")

    # ── Step 2: Agent 2 — Sector Detector ─────────────────────────────────
    sector_data = api_post(
        "/sector",
        json={
            "data_profile":   profiles,
            "memory":         {},
            "extracted_text": upload_result.get("extracted_text"),
            "user_hints":     user_hints,
        },
    )
    if not sector_data:
        return False
    domain_info = sector_data.get("final", sector_data)
    st.session_state["domain_info"] = domain_info
    bar.progress(40, text="Loading...")

    # ── Step 3: Agent 3 — KPI / Metrics Calculator ────────────────────────
    # Parse the comma-separated key_metrics string into a list
    user_metrics_list = (
        [m.strip() for m in key_metrics.split(",") if m.strip()]
        if key_metrics else []
    )
    metrics_payload: dict = {
        "domain_info":  domain_info,
        "data_profile": profiles,
        "memory":       {},
        "user_metrics": user_metrics_list,
        "key_metrics":  key_metrics,
    }
    # File-type-specific data fields (mirrors main.py lines 342-349 exactly)
    if upload_result.get("dfs_records"):          # PDF / MSG with parsed tables
        metrics_payload["pdf_dfs_records"] = upload_result["dfs_records"]
    elif upload_result.get("records"):            # CSV / Excel
        metrics_payload["df_records"]  = upload_result["records"]
        metrics_payload["df_columns"]  = upload_result.get("columns", [])
    if upload_result.get("extracted_text"):       # Any file with text content
        metrics_payload["extracted_text"] = upload_result["extracted_text"]

    metrics_data = api_post("/metrics", json=metrics_payload)
    if not metrics_data:
        return False
    concepts = metrics_data.get("concepts", [])
    st.session_state["concepts"] = concepts
    bar.progress(60, text="Loading...")

    # ── Step 4: Agent 4 — Insight Generator ───────────────────────────────
    insights_data = api_post(
        "/insights",
        json={
            "data_profile":       profiles,
            "domain_info":        domain_info,
            "extracted_concepts": concepts,
            "memory":             {},
            "extracted_text":     upload_result.get("extracted_text"),
            "user_hints":         user_hints,
            "key_metrics":        key_metrics,
        },
    )
    if not insights_data:
        return False
    st.session_state["insights"] = insights_data
    bar.progress(80, text="Loading...")

    # ── Step 5: Agent 5 — Visual Generator ────────────────────────────────
    # Pass data fields so Agent 5 uses the actual DataFrame for CSV/Excel
    # and structured tables for PDF/MSG — previously only extracted_text=""
    # was sent for CSV uploads, causing every chart to fail silently.
    visuals_data = api_post(
        "/visuals",
        json={
            "extracted_text": upload_result.get("extracted_text"),
            "insights_json":  insights_data,
            "domain_info":    domain_info,
            # Agent 3 KPIs — verified calculated values to ground chart generation
            "kpis":           concepts,
            # CSV / Excel route
            "df_records":     upload_result.get("records"),
            "df_columns":     upload_result.get("columns"),
            # PDF / TXT / MSG route (structured tables from Agent 0)
            "dfs_records":    upload_result.get("dfs_records"),
        },
    )
    if not visuals_data:
        return False
    st.session_state["rendered_visuals"] = visuals_data.get("visuals", [])
    bar.progress(100, text="Loading complete!")
    return True


def _render_visuals(
    visuals: list,
    upload_result: dict,
    insights: dict,
    domain_info: dict,
) -> None:
    """
    Render Agent 5 chart images with captions and a PowerPoint export button.

    Extracted to a named function because the same rendering is needed in
    two places within Tab 2:
      • "Just completed" state  — shown immediately after _run_pipeline() returns True
      • "Already done" state    — shown when the user revisits Tab 2 later

    Parameters
    ----------
    visuals : list
        Agent 5 output — list of dicts with keys:
        insight_text, image_b64 (base64 PNG), derived_signal,
        why_this_chart, error (if chart generation failed).
    upload_result : dict
        Needed for PowerPoint export (extracted_text field).
    insights : dict
        Agent 4 output — needed for PowerPoint export payload.
    domain_info : dict
        Agent 2 output — needed for PowerPoint export payload.
    """
    if not visuals:
        st.info(
            "No charts were generated.  This typically happens when the uploaded "
            "file has no extracted text (CSV/Excel-only uploads).  "
            "Try uploading a PDF or MSG file for richer visual insights."
        )
        return

    st.markdown("## 📈 Generated Visual Insights")

    for i, viz in enumerate(visuals, 1):
        st.markdown(f"### {i}. {viz.get('insight_text', f'Chart {i}')}")
        derived = viz.get("derived_signal", "")
        if derived:
            st.caption(f"Derived Signal: {derived}")
        why = viz.get("why_this_chart", "")
        if why:
            st.caption(why)

        if viz.get("image_b64"):
            # Decode base64 PNG and display at full container width
            # use_container_width replaces deprecated use_column_width
            img_bytes = base64.b64decode(viz["image_b64"])
            st.image(img_bytes, use_container_width=True)
        elif viz.get("error"):
            st.warning(f"⚠️ Visualization skipped: {viz['error']}")

        st.divider()

    # ── PowerPoint export ──────────────────────────────────────────────────
    # Mirrors main.py lines 503-520 faithfully.
    # Using requests directly (not api_post) because we need the raw bytes,
    # not a JSON response — api_post always calls resp.json().
    if st.button("📤 Export All Visuals to PowerPoint", key="btn_pptx"):
        pptx_payload = {
            "extracted_text": upload_result.get("extracted_text"),
            "insights_json":  insights,
            "domain_info":    domain_info,
            "kpis":           st.session_state.get("concepts", []),
            "df_records":     upload_result.get("records"),
            "df_columns":     upload_result.get("columns"),
            "dfs_records":    upload_result.get("dfs_records"),
            # Pass the already-rendered images so the backend uses the EXACT
            # same charts shown on screen — no LLM re-generation, no color drift.
            "pre_rendered_visuals": visuals,
        }
        with st.spinner("Building PowerPoint presentation..."):
            resp = requests.post(
                f"{API_BASE}/export/pptx",
                json=pptx_payload,
                verify=REQUESTS_VERIFY,
            )
        if resp.status_code == 200:
            st.download_button(
                label="⬇️ Download PowerPoint",
                data=resp.content,
                file_name="visual_insights.pptx",
                mime=(
                    "application/vnd.openxmlformats-officedocument"
                    ".presentationml.presentation"
                ),
            )
        else:
            st.error(f"Export failed: {resp.text}")


# ============================================================
# PHASE 2.5 — Tab 2: Graphs
# ============================================================

with tab2:
    st.header("📈 Graphs")
    st.caption(
        "Runs the full AI analysis pipeline (all 5 agents) and generates visual insights. "
        "Completing this step unlocks the AI Chatbot tab."
    )

    upload_result = st.session_state.get("upload_result")

    if not upload_result:
        st.info("⬆️ Upload a file first using the uploader above.")

    else:
        # ── Show cached charts if pipeline already completed ───────────────
        if st.session_state.get("pipeline_done") and st.session_state.get("rendered_visuals"):
            st.success("✅ Pipeline complete — charts generated")
            _render_visuals(
                st.session_state["rendered_visuals"],
                upload_result,
                st.session_state.get("insights", {}),
                st.session_state.get("domain_info", {}),
            )
            st.divider()

        # ── Generate (or regenerate) button ───────────────────────────────
        btn_label = (
            "🔄 Regenerate Graphs"
            if st.session_state.get("pipeline_done")
            else "🚀 Generate Graphs"
        )

        if st.button(btn_label, key="btn_graphs"):
            success = _run_pipeline(upload_result, user_hints, key_metrics)

            if success:
                st.session_state["pipeline_done"] = True
                st.success(
                    "✅ Done! Switch to the **💬 AI Chatbot** tab to ask questions "
                    "about your data."
                )
                _render_visuals(
                    st.session_state["rendered_visuals"],
                    upload_result,
                    st.session_state.get("insights", {}),
                    st.session_state.get("domain_info", {}),
                )
            else:
                st.error(
                    "The pipeline did not complete successfully. "
                    "Check the error message above and try again."
                )


# ============================================================
# PHASE 2.6 — Helper: Markdown / LaTeX Message Renderer
# ============================================================

def _render_chat_message(content: str) -> None:
    """
    Render a chat message with correct Markdown AND LaTeX math support.

    Streamlit's st.markdown() supports KaTeX math but ONLY with the
    dollar-sign delimiters ($...$ and $$...$$).  GPT frequently returns
    LaTeX using bracket notation:
      \\[ ... \\]   — display (block) math
      \\( ... \\)   — inline math

    This helper converts those notations to dollar-sign delimiters before
    passing the content to st.markdown(), so all formulas, fractions,
    Greek letters etc. render correctly rather than appearing as raw text.
    """
    # Convert display math: \[ ... \] → $$ ... $$
    content = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', content, flags=re.DOTALL)
    # Convert inline math: \( ... \) → $ ... $
    content = re.sub(r'\\\((.+?)\\\)', r'$\1$', content, flags=re.DOTALL)
    st.markdown(content)


# ============================================================
# PHASE 2.6 — Helper: Chat Message Processor
# ============================================================

def _process_chat(user_input: str, context: dict) -> None:
    """
    Submit a chat message to POST /chat and render both the user and
    assistant bubbles immediately using st.chat_message().

    This function handles the 'immediate render' pattern:
      1. Render the user bubble at once (before the API call)
      2. Render the assistant bubble with a spinner during the API call
      3. Save both turns to chat_history AFTER rendering

    Called from two places:
      • st.chat_input() handler  — user typed and pressed Enter
      • Suggested question button — user clicked a pre-written question

    Parameters
    ----------
    user_input : str
        The user's message text.
    context : dict
        ChatContext payload — pipeline outputs from session state.
        Rebuilt fresh on every script run (see Phase 2.6 design notes).
    """
    # ── Render user message immediately ───────────────────────────────────
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── Call the backend and render assistant reply ────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            payload = {
                "message": user_input,
                # Shallow copy of history at THIS moment — excludes current turn.
                # The backend appends the current message itself.
                "history": list(st.session_state["chat_history"]),
                "context": context,
            }
            result = api_post("/chat", json=payload)

        if result:
            reply = result.get("reply") or (
                "I was unable to generate a response. Please try rephrasing your question."
            )
        else:
            # api_post already called st.error() — give a safe fallback reply
            reply = (
                "Sorry, I encountered an error connecting to the AI backend. "
                "Please check that the backend is running and try again."
            )

        _render_chat_message(reply)

    # ── Save both turns to history ─────────────────────────────────────────
    # Saved AFTER rendering so that the current run shows the bubbles
    # inline, and subsequent re-runs show them via the history loop.
    st.session_state["chat_history"].append({"role": "user",      "content": user_input})
    st.session_state["chat_history"].append({"role": "assistant",  "content": reply})


# ============================================================
# PHASE 2.6 — Tab 3: AI Chatbot
# ============================================================

with tab3:
    st.header("💬 AI Chatbot")

    # ── Gate: pipeline must complete before chatbot is available ───────────
    # The chatbot's quality depends on having all 5 agent outputs in context.
    # Without them, ChatContext is nearly empty and answers are generic.
    if not st.session_state.get("pipeline_done"):
        st.warning(
            "⚠️ The AI Chatbot is not yet available for this file."
        )
        st.info(
            "Switch to the **📈 Graphs** tab and click **🚀 Generate Graphs** "
            "to run the analysis pipeline.  "
            "The chatbot unlocks automatically when all 5 agents complete."
        )

    else:
        # ── Build ChatContext from all pipeline outputs ─────────────────────
        # Rebuilt fresh on every re-run so it always reflects the latest
        # session state (e.g. after a Sector Detector re-run or pipeline
        # regeneration).  This is cheap — just dict key lookups.
        upload_result = st.session_state.get("upload_result", {})
        context: dict = {
            "profiles":       st.session_state.get("profiles")    or [],
            "domain_info":    st.session_state.get("domain_info")  or {},
            "concepts":       st.session_state.get("concepts")     or [],
            "insights":       st.session_state.get("insights")     or {},
            # raw_preview: first N rows as text (CSV/Excel); None for PDF/MSG
            "raw_preview":    upload_result.get("raw_preview"),
            # extracted_text: full document text (PDF/MSG/TXT); None for CSV/Excel
            "extracted_text": upload_result.get("extracted_text"),
            # df_records / df_columns: FULL structured dataset (CSV/Excel).
            # Passed on every chat turn so the chatbot can answer record-level
            # questions (e.g. "show me rows where revenue > 1000"), not just
            # the aggregated KPIs / insights from the pipeline.
            "df_records":     upload_result.get("records"),
            "df_columns":     upload_result.get("columns"),
            # pdf_table_records: structured tables extracted by Agent 0 from
            # PDF / TXT / MSG files (dfs_records from the upload result).
            # These are the same tables that Agents 3 & 4 use for KPI
            # computation — passing them here gives the chatbot the same
            # structured data access for answering record-level questions
            # about non-CSV file types.
            "pdf_table_records": upload_result.get("dfs_records"),
        }

        # ── Caption + Clear Chat button ────────────────────────────────────
        col_caption, col_clear = st.columns([5, 1])
        with col_caption:
            st.caption(
                "Ask any question about your uploaded data. "
                "The chatbot has full knowledge of the detected sector, "
                "calculated KPIs, and all analysis insights."
            )
        with col_clear:
            if st.button("🗑️ Clear", key="btn_clear_chat", help="Clear conversation history"):
                st.session_state["chat_history"] = []
                st.rerun()

        # ── Container for Chat Content ─────────────────────────────────────
        # Render history and new messages into a container defined before
        # the chat input so the input box always stays at the bottom.
        chat_container = st.container()

        # ── Chat input + pending suggested question ────────────────────────
        # st.chat_input() returns typed text on the Enter re-run, else None.
        # This renders sequentially below the chat_container.
        user_input = st.chat_input("Ask a question about your data...")

        pending_q = None
        if st.session_state.get("sq_pending"):
            pending_q = st.session_state["sq_pending"]
            del st.session_state["sq_pending"]

        active_input = pending_q or user_input

        # ── Render all chat content into the container ─────────────────────
        with chat_container:
            # 1. Render existing conversation history
            # Each entry is {"role": "user"|"assistant", "content": str}
            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"]):
                    _render_chat_message(msg["content"])

            # 2. Suggested questions (shown only on first open, before any chat)
            # Generic questions valid for any financial document.
            # Disappear automatically once the conversation starts.
            # Hide while a pending question is being processed (sq_pending or active_input is set).
            if not st.session_state["chat_history"] and not st.session_state.get("sq_pending") and not active_input:
                st.markdown("**💡 Not sure where to start? Try one of these:**")
                suggested_questions = [
                    "What is the overall financial performance of this data?",
                    "What were the top KPIs calculated and what do they mean?",
                    "What are the most important insights from the analysis?",
                    "What financial sector does this data belong to, and why?",
                ]
                # Render in a 2-column grid.
                sq_cols = st.columns(2)
                for i, question in enumerate(suggested_questions):
                    if sq_cols[i % 2].button(question, key=f"sq_{i}", use_container_width=True):
                        st.session_state["sq_pending"] = question
                        st.rerun()

            # 3. Process and render new user input
            if active_input:
                _process_chat(active_input, context)


