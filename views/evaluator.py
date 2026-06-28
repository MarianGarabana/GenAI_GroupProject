"""
views/evaluator.py — main evaluator page.

Upload a pitch deck → watch the 6 LangGraph agents run live, stage by stage →
get scores and an investment memo.

The progress animation is driven by REAL events: we consume `graph.stream(...)`
and re-render an animated pipeline every time a node finishes, so each stage
lights up exactly when its agent is working. State is kept in st.session_state so
the human-in-the-loop interrupt can pause and resume across reruns.
"""

import os
import tempfile
import uuid

import streamlit as st
from langgraph.types import Command

# Heavy import (LangGraph) lives here, not in the nav shell, so the app boots fast.
from graph import graph

SCORE_BAR = 6  # any dimension below this triggers human review

# Stage icons. A real <svg> element gets stripped by Streamlit's HTML sanitizer,
# so we render minimalist white line-icons as CSS mask-images inside <style>
# (CSS survives sanitization, stays crisp, and the color is just background-color).
ICON_PATHS = {
    "ingest": "<path d='M7 3h7l4 4v14H7z'/><path d='M14 3v4h4'/><path d='M9.5 12h6'/><path d='M9.5 15.5h5'/>",
    "extract": "<path d='M9 7h9'/><path d='M9 12h9'/><path d='M9 17h6'/><circle cx='5' cy='7' r='1.1' fill='%23000'/><circle cx='5' cy='12' r='1.1' fill='%23000'/><circle cx='5' cy='17' r='1.1' fill='%23000'/>",
    "validate": "<circle cx='11' cy='11' r='6'/><path d='M20.5 20.5L16.5 16.5'/><path d='M8.6 11l1.9 1.9 3.4-3.6'/>",
    "score": "<path d='M6 20v-7'/><path d='M12 20V5'/><path d='M18 20v-4'/>",
    "human_review": "<circle cx='10' cy='8' r='3.1'/><path d='M4.6 19.5c0-3.1 2.6-5.2 5.4-5.2 1 0 1.9.3 2.7.7'/><path d='M14.8 16.8l2 2 4-4.2'/>",
    "write_memo": "<rect x='4' y='6' width='16' height='12' rx='2'/><path d='M5 8l7 5 7-5'/>",
}


def _mask_url(body: str) -> str:
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
        "stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
        + body + "</svg>"
    )
    return f'url("data:image/svg+xml,{svg}")'


ICON_STYLE = (
    "<style>"
    ".lp-ic{width:30px;height:30px;display:block;background-color:#fff;"
    "-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;"
    "-webkit-mask-position:center;mask-position:center;"
    "-webkit-mask-size:contain;mask-size:contain;"
    "filter:drop-shadow(0 1px 1.5px rgba(6,16,31,.5));}"
    + "".join(
        f".ic-{k}{{-webkit-mask-image:{_mask_url(v)};mask-image:{_mask_url(v)};}}"
        for k, v in ICON_PATHS.items()
    )
    + "</style>"
)

# key, label, stage color  — order matches the graph
PIPELINE = [
    ("ingest", "Ingest", "#22D3EE"),
    ("extract", "Extract", "#3B82F6"),
    ("validate", "Validate", "#8B5CF6"),
    ("score", "Score", "#FBBF24"),
    ("human_review", "Review", "#FB7185"),
    ("write_memo", "Memo", "#34D399"),
]
NODE_KEYS = {p[0] for p in PIPELINE}

DIMENSIONS = [
    (
        "Market", "market", ":material/public:",
        (
            "Market size & opportunity (30% of composite score)\n\n"
            "Measures how large and credible the addressable market is. "
            "Scores above 6 require a verified, sizable TAM with a realistic path to capturing it. "
            "Below 6 flags an overstated, understated, or unverifiable market — a major concern "
            "because market size is the ceiling on potential returns.\n\n"
            f"Human review triggers if this falls below {SCORE_BAR}/10."
        ),
    ),
    (
        "Team", "team", ":material/groups:",
        (
            "Founding team quality (30% of composite score)\n\n"
            "Evaluates whether the founders have the credentials, domain expertise, and track record "
            "to execute. At seed stage this is the single most important signal — investors bet on "
            "the team as much as the idea. Scores above 6 require independently verifiable "
            "backgrounds. Below 6 means credentials are unverified, contradicted by web searches, "
            "or key roles are missing.\n\n"
            f"Human review triggers if this falls below {SCORE_BAR}/10."
        ),
    ),
    (
        "Product", "product", ":material/widgets:",
        (
            "Product differentiation & defensibility (25% of composite score)\n\n"
            "Assesses whether the product solves a real problem in a differentiated way and whether "
            "the competitive advantage is defensible (IP, network effects, switching costs). "
            "Scores above 6 require specific, verifiable claims about what makes the product "
            "unique. Below 6 flags vague or generic claims without supporting evidence.\n\n"
            f"Human review triggers if this falls below {SCORE_BAR}/10."
        ),
    ),
    (
        "Traction", "traction", ":material/trending_up:",
        (
            "Revenue, growth & customer traction (15% of composite score)\n\n"
            "Looks at real-world evidence that people are using and paying for the product — "
            "revenue, active users, growth rate, signed contracts. Carries the least weight at "
            "seed stage because early-stage companies may not have much yet. However, any "
            "reported metrics that are internally inconsistent or appear to misrepresent "
            "performance (e.g. GMV reported as revenue) will lower this score significantly.\n\n"
            f"Human review triggers if this falls below {SCORE_BAR}/10."
        ),
    ),
]

PIPE_STYLE = """
<style>
  .lp-wrap { padding: 46px 8px 16px; }
  .lp-row { display: flex; align-items: flex-start; gap: 0; overflow-x: auto; padding-bottom: 4px; }
  .lp-stage { flex: 0 0 112px; display: flex; flex-direction: column; align-items: center; text-align: center; }
  .lp-node {
    position: relative; width: 64px; height: 64px; border-radius: 19px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(255,255,255,.05); border: 1.5px solid rgba(255,255,255,.14);
    transition: all .45s ease;
  }
  .lp-name { margin-top: 12px; font-size: 12.5px; font-weight: 600; color: #aebbd4; transition: color .4s; }
  .lp-status { margin-top: 3px; font-size: 9.5px; font-weight: 700; letter-spacing: .7px; text-transform: uppercase; color: #56627d; }

  .lp-stage.pending .lp-node { opacity: .62; }

  .lp-stage.active .lp-node {
    background: var(--c); border-color: transparent; color: #06101f;
    animation: lpGlow 1.5s ease-in-out infinite;
  }
  .lp-stage.active .lp-node::after {
    content: ""; position: absolute; inset: -7px; border-radius: 24px;
    border: 2.5px solid transparent; border-top-color: var(--c); border-right-color: var(--c);
    animation: lpSpin .85s linear infinite;
  }
  .lp-stage.active .lp-name { color: #fff; }
  .lp-stage.active .lp-status { color: var(--c); }
  @keyframes lpGlow { 0%,100% { box-shadow: 0 0 16px 1px var(--c); } 50% { box-shadow: 0 0 30px 7px var(--c); } }
  @keyframes lpSpin { to { transform: rotate(360deg); } }

  .lp-stage.done .lp-node { background: var(--c); border-color: transparent; color: #06101f; }
  .lp-stage.done .lp-status { color: #8290a8; }
  .lp-check {
    position: absolute; right: -5px; bottom: -5px; width: 19px; height: 19px; border-radius: 50%;
    background: #0a1326; color: var(--c); border: 2px solid var(--c);
    font-size: 11px; font-weight: 800; display: flex; align-items: center; justify-content: center;
  }

  .lp-stage.skip .lp-node { border-style: dashed; border-color: rgba(255,255,255,.18); color: #3f4b63; opacity: .45; }
  .lp-stage.skip .lp-name { color: #5a6680; }

  .lp-link { flex: 0 0 26px; height: 64px; position: relative; }
  .lp-link::before {
    content: ""; position: absolute; left: -2px; right: -2px; top: 32px; height: 3px; transform: translateY(-50%);
    background: rgba(255,255,255,.10); border-radius: 3px; transition: background .4s;
  }
  .lp-link.fill::before { background: linear-gradient(90deg, var(--ca), var(--cb)); box-shadow: 0 0 10px -1px var(--cb); }
</style>
"""


def pipeline_html(completed, scores) -> str:
    """Render the animated pipeline for a given set of finished nodes."""
    completed = set(completed)
    nums = [scores.get(k) for k in ("market", "team", "traction", "product")] if scores else []

    review_skip = False
    if "score" in completed and nums and all(isinstance(n, (int, float)) and n >= SCORE_BAR for n in nums):
        review_skip = True
    if "write_memo" in completed and "human_review" not in completed:
        review_skip = True

    status = {}
    for key, _, _ in PIPELINE:
        if key in completed:
            status[key] = "done"
        elif key == "human_review" and review_skip:
            status[key] = "skip"
        else:
            status[key] = "pending"
    for key, _, _ in PIPELINE:  # first remaining node is the one currently working
        if status[key] == "pending":
            status[key] = "active"
            break

    labels = {"pending": "Queued", "active": "Working", "done": "Done", "skip": "Skipped"}
    parts = []
    for i, (key, name, color) in enumerate(PIPELINE):
        if i > 0:
            prev_key, _, prev_color = PIPELINE[i - 1]
            filled = "fill" if status[prev_key] in ("done", "skip") else ""
            parts.append(f'<div class="lp-link {filled}" style="--ca:{prev_color};--cb:{color}"></div>')
        stt = status[key]
        check = '<span class="lp-check">&#10003;</span>' if stt == "done" else ""
        parts.append(
            f'<div class="lp-stage {stt}" style="--c:{color}">'
            f'<div class="lp-node"><i class="lp-ic ic-{key}"></i>{check}</div>'
            f'<div class="lp-name">{name}</div>'
            f'<div class="lp-status">{labels[stt]}</div>'
            f"</div>"
        )
    return PIPE_STYLE + ICON_STYLE + '<div class="lp-wrap"><div class="lp-row">' + "".join(parts) + "</div></div>"


# ------------------------------------------------------------------
# Result renderers
# ------------------------------------------------------------------

def render_scores(scores: dict) -> None:
    st.markdown("##### :material/scoreboard: Scores")
    reasoning = scores.get("reasoning") or {}
    confidence = scores.get("confidence", "")
    composite = scores.get("composite_score")

    cols = st.columns(4)
    for col, (label, key, icon, help_text) in zip(cols, DIMENSIONS):
        value = scores.get(key, "—")
        with col:
            if isinstance(value, (int, float)):
                st.metric(
                    f"{icon} {label}",
                    f"{value:g}/10",
                    delta=round(value - SCORE_BAR, 1),
                    border=True,
                    help=help_text,
                )
                reason = reasoning.get(key, "") if isinstance(reasoning, dict) else ""
                if reason:
                    st.caption(reason)
            else:
                st.metric(f"{icon} {label}", "—", border=True)

    badge_parts = []
    if composite is not None:
        if composite >= 7.5:
            badge_parts.append(f":green-badge[:material/verified: Strong — composite {composite:.2f}/10]")
        elif composite >= SCORE_BAR:
            badge_parts.append(f":blue-badge[:material/thumb_up: Solid — composite {composite:.2f}/10]")
        else:
            badge_parts.append(f":orange-badge[:material/warning: Below bar — composite {composite:.2f}/10]")
    else:
        numeric = [scores.get(k) for _, k, *_ in DIMENSIONS if isinstance(scores.get(k), (int, float))]
        if numeric:
            avg = sum(numeric) / len(numeric)
            if avg >= 7.5:
                badge_parts.append(f":green-badge[:material/verified: Strong — avg {avg:.1f}/10]")
            elif avg >= SCORE_BAR:
                badge_parts.append(f":blue-badge[:material/thumb_up: Solid — avg {avg:.1f}/10]")
            else:
                badge_parts.append(f":orange-badge[:material/warning: Below bar — avg {avg:.1f}/10]")

    conf_badge = {
        "high":   ":green-badge[:material/shield: Evidence confidence: High]",
        "medium": ":blue-badge[:material/shield: Evidence confidence: Medium]",
        "low":    ":orange-badge[:material/error: Evidence confidence: Low]",
    }.get(confidence, "")
    if conf_badge:
        badge_parts.append(conf_badge)

    if badge_parts:
        st.markdown("  &nbsp;  ".join(badge_parts))


def render_memo(memo: str, *, key: str) -> None:
    st.markdown("##### :material/draft: Investment memo")
    with st.container(border=True):
        st.markdown(memo)
    st.download_button(
        "Download memo",
        data=memo,
        file_name="investment_memo.txt",
        mime="text/plain",
        icon=":material/download:",
        key=key,
    )


def stream_pipeline(graph_input, config, placeholder, completed):
    """Drive the graph and animate the pipeline. Returns (final_state, interrupt_payload)."""
    final_state = {}
    interrupt_payload = None
    for chunk in graph.stream(graph_input, config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            interrupt_payload = chunk["__interrupt__"][0].value
            placeholder.html(pipeline_html(completed, interrupt_payload.get("scores", {})))
            break
        for node_name, delta in chunk.items():
            if node_name in NODE_KEYS:
                completed.append(node_name)
                if isinstance(delta, dict):
                    final_state.update(delta)
                placeholder.html(pipeline_html(completed, final_state.get("scores", {})))
    return final_state, interrupt_payload


# ==================================================================
# Page
# ==================================================================

ss = st.session_state

st.title(":material/query_stats: VC Pitch Evaluator")
st.markdown(
    "Upload a startup pitch deck and a team of AI agents will extract the key claims, "
    "fact-check them against the live web, score the opportunity, and write a "
    "partner-ready investment memo — watch each agent work in real time below."
)
st.markdown(
    ":blue-badge[LangGraph orchestration] &nbsp; "
    ":violet-badge[Gemini 2.5 Flash] &nbsp; "
    ":green-badge[RAG · PyPDF + Chroma] &nbsp; "
    ":gray-badge[Web fact-checking] &nbsp; "
    ":orange-badge[Human-in-the-loop]"
)
st.markdown("")

# ------------------------------------------------------------------
# Upload
# ------------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    with st.container(border=True):
        st.markdown("##### :material/upload_file: Upload pitch deck")
        uploaded_file = st.file_uploader("Drop a text-based PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded_file:
            st.success(f"Ready: **{uploaded_file.name}**", icon=":material/check_circle:")
        else:
            st.caption("PDF only · text-based decks work best (scanned images can't be read).")

with right:
    with st.container(border=True):
        st.markdown("##### :material/route: Pipeline")
        st.markdown(
            "1. **Ingest** — PDF → text\n"
            "2. **Extract** — Gemini finds 4 claims\n"
            "3. **Validate** — web fact-check\n"
            "4. **Score** — 0–10 per dimension\n"
            "5. **Review** — human gate if any score < 6\n"
            "6. **Memo** — investment recommendation"
        )
        st.caption("See it animated on the **How it works** page.")

# ------------------------------------------------------------------
# Run (initial pass)
# ------------------------------------------------------------------
if uploaded_file:
    if st.button("Analyze pitch deck", type="primary", width="stretch", icon=":material/rocket_launch:"):
        # fresh run — clear any previous result and use a new graph thread
        for k in ("completed", "scores", "interrupt", "memo", "config"):
            ss.pop(k, None)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            pdf_path = tmp.name

        config = {"configurable": {"thread_id": uuid.uuid4().hex}}

        st.markdown("##### :material/graphic_eq: Live agent pipeline")
        placeholder = st.empty()
        completed = []
        placeholder.html(pipeline_html(completed, {}))

        ok = False
        try:
            final_state, interrupt_payload = stream_pipeline(
                {"pdf_path": pdf_path}, config, placeholder, completed
            )
            ok = True
        except Exception as e:
            st.error(f"Analysis failed: {e}", icon=":material/error:")
            import traceback
            with st.expander("Technical details"):
                st.code(traceback.format_exc())
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

        if ok:
            ss.completed = completed
            ss.config = config
            if interrupt_payload is not None:
                ss.interrupt = interrupt_payload
                ss.scores = interrupt_payload.get("scores", {})
            else:
                ss.scores = final_state.get("scores", {})
                ss.memo = final_state.get("investment_memo", "No memo generated")
            st.rerun()

# ------------------------------------------------------------------
# Persistent result view (survives reruns, incl. the human-in-the-loop)
# ------------------------------------------------------------------
if ss.get("memo"):
    st.markdown("##### :material/graphic_eq: Pipeline")
    st.html(pipeline_html(ss.get("completed", []), ss.get("scores", {})))
    st.success("Analysis complete", icon=":material/check_circle:")
    render_scores(ss.get("scores", {}))
    render_memo(ss.memo, key="dl_final")

elif ss.get("interrupt"):
    st.markdown("##### :material/graphic_eq: Pipeline — paused for review")
    st.html(pipeline_html(ss.get("completed", []), ss.get("scores", {})))
    st.warning("One or more scores fell below the bar — analyst sign-off required.", icon=":material/gavel:")
    render_scores(ss.get("scores", {}))

    scores = ss.get("scores", {})
    flagged = [
        (label, key, scores[key])
        for label, key, *_ in DIMENSIONS
        if isinstance(scores.get(key), (int, float)) and scores[key] < SCORE_BAR
    ]
    if flagged:
        with st.expander(f":material/flag: Why review was triggered ({len(flagged)} dimension{'s' if len(flagged) > 1 else ''} below {SCORE_BAR}/10)", expanded=True):
            reasoning = scores.get("reasoning") or {}
            for label, key, val in flagged:
                reason = reasoning.get(key, "") if isinstance(reasoning, dict) else ""
                st.markdown(f"**{label}: {val:g}/10** — {reason}" if reason else f"**{label}: {val:g}/10**")

    st.markdown("##### :material/rate_review: Analyst review")
    feedback = st.text_area(
        "Your assessment",
        placeholder="e.g. 'Team is exceptional; traction just needs time to validate. Recommend proceeding with a smaller check.'",
        height=120,
        key="review_feedback",
    )
    if st.button("Submit review & generate memo", type="primary", width="stretch", icon=":material/send:"):
        st.markdown("##### :material/graphic_eq: Resuming pipeline")
        placeholder = st.empty()
        completed = list(ss.get("completed", []))
        try:
            final_state, _ = stream_pipeline(
                Command(resume=feedback), ss.config, placeholder, completed
            )
            ss.completed = completed
            ss.memo = final_state.get("investment_memo", "No memo generated")
            ss.pop("interrupt", None)
            st.rerun()
        except Exception as e:
            st.error(f"Could not generate the memo: {e}", icon=":material/error:")

elif not uploaded_file:
    # Empty state
    st.markdown("")
    with st.container(border=True):
        st.markdown(
            "##### :material/auto_awesome: Waiting for a deck\n"
            "Upload a pitch deck above to start. The six agents will light up one by one as they "
            "read it, verify the claims, score it, and write the memo."
        )
