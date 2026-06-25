"""
PATTERN:
  1. create_agent()     → prompt | llm.bind_tools(tools)
  2. agent_node()       → calls agent, returns {"messages": [result]}
  3. ToolNode(tools)    → executes whichever tool the LLM chose to call
  4. should_continue()  → conditional edge: if tool_calls → tools, else → END
  5. Sub-graph compiled → reusable validation pipeline
This pattern is the same as the one seen in class.
"""

import functools
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agents.tools import VALIDATOR_TOOLS


# ── Agent sub-graph state ─────────────────────────────────────────────────────
# Separate from PitchState — this is the internal state of the validation loop.
# Uses add_messages so each new message is appended, not replaced.
# CLASS CONCEPT: Annotated + add_messages — Session 9-10

class ValidatorState(TypedDict):
    messages: Annotated[list, add_messages]


# ── Lazy-loaded LLM ───────────────────────────────────────────────────────────

_llm = None

def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    return _llm


# ── create_agent — identical to professor's pattern ──────────────────────────
# CLASS CONCEPT: Sessions 8-9, 10-11
# From Conchita_News_Writer_Agent_in_LangGraph_june26.ipynb:
#   def create_agent(llm, tools, system_message):
#       prompt = ChatPromptTemplate.from_messages([...])
#       return prompt | llm.bind_tools(tools)

def create_agent(llm, tools, system_message: str):
    """Creates an agent as a prompt | llm.bind_tools(tools) chain."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_message}"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm


MAX_CHARS_PER_CLAIM = 600

VALIDATOR_SYSTEM_MESSAGE = f"""You are a research analyst gathering evidence to support a venture capital evaluation.

For each claim provided:
1. Search the web for current market data, benchmarks, or news
2. Search Wikipedia for industry background or founder context
3. Write exactly 3-4 sentences of factual evidence (max {MAX_CHARS_PER_CLAIM} characters per claim) including:
   - Specific numbers, statistics, or facts from your search results
   - How the market or industry actually looks based on public data
   - Any relevant context about the founders, competitors, or product category

Do NOT score or judge the claims — just report what you found.
Structure your response with a clear heading for each claim."""


# ── agent_node — identical to professor's pattern ────────────────────────────
# From notebook: def agent_node(state, agent, name): result = agent.invoke(state)

def agent_node(state: ValidatorState, agent, name: str) -> dict:
    result = agent.invoke(state)
    return {"messages": [result]}


# ── should_continue — conditional edge ───────────────────────────────────────
# From notebook: if last_message.tool_calls → "tools" else → "outliner"
# CLASS CONCEPT: Conditional edges — Session 9-10

def should_continue(state: ValidatorState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# ── Build the validation sub-graph ───────────────────────────────────────────
# Mirrors the professor's graph exactly, adapted for validation instead of news writing.

def _build_validator_graph():
    llm = get_llm()
    validator_agent = create_agent(llm, VALIDATOR_TOOLS, VALIDATOR_SYSTEM_MESSAGE)

    # Wrap agent_node with functools.partial — same as notebook
    validator_node = functools.partial(agent_node, agent=validator_agent, name="Validator")

    tool_node = ToolNode(VALIDATOR_TOOLS)

    workflow = StateGraph(ValidatorState)
    workflow.add_node("validator", validator_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("validator")
    workflow.add_conditional_edges("validator", should_continue)
    workflow.add_edge("tools", "validator")

    return workflow.compile()


_validator_graph = None

def _get_validator_graph():
    global _validator_graph
    if _validator_graph is None:
        _validator_graph = _build_validator_graph()
    return _validator_graph


# ── Public interface for Role 1 ───────────────────────────────────────────────

def validate_claims(state: dict) -> dict:
    """
    LangGraph node function.

    Reads:  state["extracted_claims"]  — dict with market_size, team_background,
                                          traction, product_description
    Writes: state["validation_results"] — dict with one assessment string per claim
    """
    claims = state.get("extracted_claims", {})

    if not claims:
        return {"validation_giresults": {}, "error": "No claims to validate."}

    claims_text = "\n".join(
        f"- {key.replace('_', ' ').title()}: {value}"
        for key, value in claims.items()
        if value and "Not mentioned" not in str(value)
    )

    prompt = (
        f"Please research the following claims from a startup pitch deck "
        f"and return factual evidence for each one:\n\n"
        f"{claims_text}"
    )

    try:
        result = _get_validator_graph().invoke(
            {"messages": [HumanMessage(content=prompt)]}
        )

        # The final message is the agent's written assessment.
        # Gemini 2.5 Flash returns content as a list of parts, not a plain string.
        final_message = result["messages"][-1]
        content = final_message.content if hasattr(final_message, "content") else ""
        if isinstance(content, list):
            summary = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        else:
            summary = str(content)

        # Parse the agent's summary back into per-claim results.
        # Find each heading's position, then slice between consecutive headings
        # so sections don't bleed into each other.
        lower = summary.lower()
        positions = {}
        for key in claims:
            label = key.replace("_", " ").title()
            idx = lower.find(label.lower())
            if idx != -1:
                positions[key] = idx

        sorted_keys = sorted(positions, key=lambda k: positions[k])
        validation_results = {}
        for key in sorted_keys:
            start = positions[key]
            validation_results[key] = summary[start : start + MAX_CHARS_PER_CLAIM].strip()

        # Any claim whose heading wasn't found gets the first MAX_CHARS_PER_CLAIM of the summary
        for key in claims:
            if key not in validation_results:
                validation_results[key] = summary[:MAX_CHARS_PER_CLAIM]

        return {"validation_results": validation_results, "error": None}

    except Exception as e:
        return {
            "validation_results": {
                key: f"UNVERIFIED — Validation failed: {str(e)[:100]}"
                for key in claims
            },
            "error": f"Validation agent failed: {str(e)}",
        }
