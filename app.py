# SSL fix for IE University / corporate networks — must come before all other imports.
# The university network runs an SSL inspection proxy that replaces server certificates
# with its own. The google-genai SDK uses httpx, which rejects those certificates.
# verify=False disables verification for httpx connections only. Acceptable for a
# local class project; do not use in production.
import httpx
_orig_httpx_init = httpx.Client.__init__
def _ssl_fixed_httpx_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _orig_httpx_init(self, *args, **kwargs)
httpx.Client.__init__ = _ssl_fixed_httpx_init

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# Navigation shell
# ------------------------------------------------------------------
# This file is the single entry point (`streamlit run app.py`). It sets
# global config, brand, and routes between the two pages in `views/`.
# Each page is a self-contained script; shared state lives in st.session_state.
# ------------------------------------------------------------------

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(
    page_title="VC Pitch Evaluator",
    page_icon=str(ASSETS / "icon.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"))

evaluator = st.Page(
    "views/evaluator.py",
    title="Evaluator",
    icon=":material/query_stats:",
    default=True,
)
how_it_works = st.Page(
    "views/how_it_works.py",
    title="How it works",
    icon=":material/account_tree:",
)

nav = st.navigation([evaluator, how_it_works])
nav.run()

# ------------------------------------------------------------------
# Shared sidebar footer — renders under the nav on every page
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(" ")
    st.caption(
        "Powered by **LangGraph** · **Gemini 2.5 Flash** · **RAG**  \n"
        "Generative AI — IE University MBDS"
    )
