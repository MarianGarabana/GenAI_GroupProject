"""
views/how_it_works.py — animated explainer for the agent pipeline.

A CSS-animated walkthrough of how the 6 LangGraph agents turn a raw PDF into an
investment memo. Animations are pure CSS keyframes (no JS), injected via st.html.
Below the animation, native Streamlit cards give the readable, course-mapped detail.
"""

import streamlit as st

from graph import graph, get_graph_image

# ------------------------------------------------------------------
# Stage icons — minimalist white line-icons rendered as CSS mask-images
# (a real <svg> element gets stripped by the HTML sanitizer; CSS survives).
# ------------------------------------------------------------------
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
    ".hiw-ic{width:32px;height:32px;display:block;background-color:#fff;"
    "-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;"
    "-webkit-mask-position:center;mask-position:center;"
    "-webkit-mask-size:contain;mask-size:contain;"
    "filter:drop-shadow(0 1px 2px rgba(6,16,31,.55));}"
    + "".join(
        f".ic-{k}{{-webkit-mask-image:{_mask_url(v)};mask-image:{_mask_url(v)};}}"
        for k, v in ICON_PATHS.items()
    )
    + "</style>"
)

STAGES = [
    ("analysis", "01", "Ingest", "PDF &rarr; raw text", "ingest"),
    ("analysis", "02", "Extract", "4 investor claims", "extract"),
    ("analysis", "03", "Validate", "Web fact-check", "validate"),
    ("analysis", "04", "Score", "0&ndash;10 per axis", "score"),
    ("human", "05", "Review", "Human gate if &lt; 6", "human_review"),
    ("memo", "06", "Memo", "Investment memo", "write_memo"),
]

# ------------------------------------------------------------------
# Build the animated hero + pipeline (pure CSS, no JS)
# ------------------------------------------------------------------
_nodes = []
for i, (cls, num, name, desc, ikey) in enumerate(STAGES):
    if i > 0:
        link_delay = f"{i * 0.4:.2f}s"
        _nodes.append(
            f'<div class="hiw-link"><span class="hiw-dot" style="animation-delay:{link_delay}"></span></div>'
        )
    _nodes.append(
        f'<div class="hiw-stage hiw-{cls}" style="--d:{i * 0.12:.2f}s">'
        f'<div class="hiw-node"><i class="hiw-ic ic-{ikey}"></i></div>'
        f'<div class="hiw-step">STEP {num}</div>'
        f'<div class="hiw-name">{name}</div>'
        f'<div class="hiw-desc">{desc}</div>'
        f"</div>"
    )
_pipeline = "".join(_nodes)

st.html(
    ICON_STYLE
    + f"""
<style>
  .hiw-hero {{
    position: relative; overflow: hidden; border-radius: 20px;
    padding: 34px 34px 30px; margin: 2px 0 6px; color: #fff;
    background:
      radial-gradient(820px 300px at 10% -20%, rgba(91,141,239,.40), transparent 60%),
      radial-gradient(680px 320px at 102% 130%, rgba(21,128,61,.26), transparent 55%),
      linear-gradient(125deg, #1d2942, #16213b 55%, #10182b);
    box-shadow: 0 22px 48px -26px rgba(15, 23, 42, .65);
  }}
  .hiw-hero::after {{
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background: linear-gradient(115deg, transparent 32%, rgba(255,255,255,.07) 48%, transparent 64%);
    background-size: 250% 250%; animation: hiwSheen 7s ease-in-out infinite;
  }}
  @keyframes hiwSheen {{ 0% {{ background-position: 130% 0; }} 100% {{ background-position: -30% 0; }} }}
  .hiw-eyebrow {{ font-size: 12px; font-weight: 700; letter-spacing: 2.4px; text-transform: uppercase; color: #9db8ff; }}
  .hiw-h1 {{ font-size: 30px; font-weight: 800; letter-spacing: -.6px; margin: 9px 0 9px; line-height: 1.12; }}
  .hiw-sub {{ font-size: 15px; line-height: 1.6; color: #cdd7ea; max-width: 700px; margin: 0; }}

  .hiw-pipe {{ display: flex; align-items: flex-start; gap: 0; overflow-x: auto; padding: 30px 6px 6px; }}
  .hiw-stage {{
    flex: 0 0 120px; display: flex; flex-direction: column; align-items: center; text-align: center;
    transform: translateY(16px); opacity: 0;
    animation: hiwUp .65s cubic-bezier(.2,.7,.2,1) forwards; animation-delay: var(--d);
  }}
  @keyframes hiwUp {{ to {{ transform: translateY(0); opacity: 1; }} }}

  .hiw-node {{
    width: 74px; height: 74px; border-radius: 20px; position: relative;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 16px 28px -14px rgba(30,42,68,.6);
    animation: hiwFloat 4s ease-in-out infinite; animation-delay: var(--d);
  }}
  .hiw-node::before {{
    content: ""; position: absolute; inset: -4px; border-radius: 24px;
    border: 2px solid currentColor; opacity: 0;
    animation: hiwPulse 3s ease-out infinite; animation-delay: var(--d);
  }}
  @keyframes hiwFloat {{ 0%,100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-5px); }} }}
  @keyframes hiwPulse {{ 0% {{ transform: scale(.82); opacity: .5; }} 70% {{ transform: scale(1.2); opacity: 0; }} 100% {{ opacity: 0; }} }}

  .hiw-analysis .hiw-node {{ background: linear-gradient(150deg, #3a5ea8, #1e2a44); color: #8fb0f6; }}
  .hiw-human .hiw-node {{ background: linear-gradient(150deg, #c79a3e, #94670f); color: #ffe1a3; }}
  .hiw-memo .hiw-node {{ background: linear-gradient(150deg, #2f9e6b, #14603e); color: #9eeec5; }}

  .hiw-step {{ margin-top: 13px; font-size: 9.5px; font-weight: 700; letter-spacing: 1.4px; color: #8090ad; }}
  .hiw-name {{ font-size: 13.5px; font-weight: 700; color: #E6ECF7; margin-top: 2px; }}
  .hiw-desc {{ font-size: 11.5px; color: #9AAAC6; line-height: 1.35; margin-top: 3px; padding: 0 2px; }}

  .hiw-link {{ flex: 0 0 42px; height: 74px; position: relative; }}
  .hiw-link::before {{
    content: ""; position: absolute; left: -3px; right: -3px; top: 37px; height: 3px; transform: translateY(-50%);
    background: repeating-linear-gradient(90deg, #c4d0e6 0 7px, transparent 7px 13px);
    -webkit-mask: linear-gradient(90deg, transparent, #000 16%, #000 84%, transparent);
            mask: linear-gradient(90deg, transparent, #000 16%, #000 84%, transparent);
    animation: hiwDash 1.05s linear infinite;
  }}
  @keyframes hiwDash {{ to {{ background-position: 13px 0; }} }}
  .hiw-dot {{
    position: absolute; top: 37px; left: 0; width: 9px; height: 9px; border-radius: 50%;
    transform: translate(-50%, -50%); background: #5b8def;
    box-shadow: 0 0 11px 2px rgba(91,141,239,.7);
    animation: hiwTravel 2.4s ease-in-out infinite;
  }}
  @keyframes hiwTravel {{ 0% {{ left: 0; opacity: 0; }} 12% {{ opacity: 1; }} 88% {{ opacity: 1; }} 100% {{ left: 100%; opacity: 0; }} }}

  .hiw-legend {{ display: flex; gap: 18px; flex-wrap: wrap; font-size: 12px; color: #9AAAC6; padding: 4px 8px 2px; }}
  .hiw-legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
  .hiw-chip {{ width: 11px; height: 11px; border-radius: 4px; }}
  .hiw-chip.b {{ background: linear-gradient(150deg, #3a5ea8, #1e2a44); }}
  .hiw-chip.a {{ background: linear-gradient(150deg, #c79a3e, #94670f); }}
  .hiw-chip.g {{ background: linear-gradient(150deg, #2f9e6b, #14603e); }}
</style>

<div class="hiw-hero">
  <div class="hiw-eyebrow">Inside the pipeline</div>
  <div class="hiw-h1">Six AI agents, one investment decision</div>
  <p class="hiw-sub">A pitch deck enters as a PDF and leaves as a partner-ready memo. Each agent does one
  job, hands its work to the next, and the whole chain pauses for a human whenever the numbers look shaky &mdash;
  orchestrated end-to-end with LangGraph.</p>
</div>

<div class="hiw-pipe">{_pipeline}</div>
<div class="hiw-legend">
  <span><i class="hiw-chip b"></i> AI analysis</span>
  <span><i class="hiw-chip a"></i> Human gate</span>
  <span><i class="hiw-chip g"></i> Output</span>
</div>
"""
)

st.markdown("")

# ------------------------------------------------------------------
# The six agents — readable, course-mapped detail
# ------------------------------------------------------------------
st.subheader("The six agents, in detail")

CARDS = [
    (":material/description:", "1 · Ingest",
     "PyPDF reads every page of the deck and concatenates it into one raw text blob — the document-loading step of a RAG pipeline.",
     ":blue-badge[RAG · Session 6]"),
    (":material/lightbulb:", "2 · Extract",
     "A Gemini LCEL chain (`prompt | llm | parser`) returns exactly four investor claims as JSON: market size, team, traction, and product.",
     ":violet-badge[LCEL chains · Sessions 8–9]"),
    (":material/travel_explore:", "3 · Validate",
     "For each claim the agent runs a DuckDuckGo search, then Gemini labels the evidence **VERIFIED**, **PLAUSIBLE**, or **UNVERIFIED**.",
     ":green-badge[Agents + tools · Session 7]"),
    (":material/scoreboard:", "4 · Score",
     "Gemini weighs the claims against the web evidence and scores Market, Team, Product, and Traction from 0–10, each with one line of reasoning.",
     ":blue-badge[Structured output · Sessions 10–11]"),
    (":material/rate_review:", "5 · Human review",
     "If any score lands below 6, `interrupt()` freezes the graph (state saved by MemorySaver) until an analyst submits feedback — then it resumes.",
     ":orange-badge[Human-in-the-loop · Sessions 9–10]"),
    (":material/draft:", "6 · Memo",
     "A final Gemini chain synthesizes everything — claims, evidence, scores, analyst notes — into an **INVEST / PASS / CONDITIONAL** memo.",
     ":gray-badge[Generation · Session 3]"),
]

for row_start in range(0, len(CARDS), 3):
    cols = st.columns(3, gap="medium")
    for col, (icon, title, body, badge) in zip(cols, CARDS[row_start:row_start + 3]):
        with col:
            with st.container(border=True):
                st.markdown(f"##### {icon} {title}")
                st.markdown(body)
                st.markdown(badge)

st.markdown("")

# ------------------------------------------------------------------
# Under the hood + the live graph
# ------------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    with st.container(border=True):
        st.markdown("##### :material/hub: Why LangGraph")
        st.markdown(
            "- **Typed shared state** — every agent reads and writes one `PitchState` `TypedDict`.\n"
            "- **Conditional routing** — a single edge function inspects the scores and forks to "
            "*human review* or straight to *memo*.\n"
            "- **Checkpointed memory** — `MemorySaver` snapshots state after every node, which is what "
            "lets the human-in-the-loop interrupt pause and resume without losing work.\n"
            "- **Streaming-ready** — the same compiled graph can stream node-by-node updates."
        )

with right:
    with st.container(border=True):
        st.markdown("##### :material/account_tree: The live graph")
        st.caption("Rendered straight from the compiled LangGraph — this *is* the running pipeline, not a drawing.")
        try:
            st.image(get_graph_image(graph), width="stretch")
        except Exception as e:
            st.info(
                "Live diagram needs network access to the Mermaid renderer — "
                "the animated pipeline above is always available.",
                icon=":material/cloud_off:",
            )
            st.caption(f"({e})")
