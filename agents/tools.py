"""
TOOLS DEFINED:
  search_web       — DuckDuckGo for current market data, traction benchmarks, news
  search_wikipedia — Wikipedia for industry background, founder context, tech categories
"""

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

_ddg = DuckDuckGoSearchRun()
_wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1500)
)


@tool
def search_web(query: str) -> str:
    """Search the web for current information about a topic.
    Use this to validate market size figures, traction benchmarks, funding rounds, and competitor data."""
    try:
        return _ddg.run(query)[:2000]
    except Exception as e:
        return f"Web search unavailable: {str(e)}"


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for background context on an industry, technology, or person.
    Use this to validate founder credentials, understand an industry's history, or get context on a product category."""
    try:
        return _wiki.run(query)
    except Exception as e:
        return f"Wikipedia unavailable: {str(e)}"


VALIDATOR_TOOLS = [search_web, search_wikipedia]
