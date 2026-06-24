import streamlit as st
from pathlib import Path
import sys
import os

st.set_page_config(page_title="VC Pitch Evaluator", layout="wide")
st.title("💼 VC Startup Pitch Evaluator")

st.markdown("""
This system evaluates startup pitch decks using AI agents and LangGraph orchestration.
- **Upload** a PDF pitch deck
- **Extract** claims about market size, team, traction, product
- **Validate** claims against live market data using web search
- **Score** each dimension 0–10
- **Review** human analyst if any score < 6
- **Get** a final investment recommendation memo
""")

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Upload pitch deck")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

with col2:
    st.subheader("Status")
    st.info("🟡 Waiting for PDF...", icon="ℹ️")

if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    if st.button("🚀 Analyze pitch deck"):
        st.info("⏳ Processing... (backend integration coming Thursday)")
        
        with st.expander("Expected output structure", expanded=True):
            st.markdown("""
            **Once the backend is ready, you'll see:**
            - Extracted claims from the pitch deck
            - Evidence from web search validation
            - Scores for Team, Market, Product, Traction
            - Investment memo with recommendation
            - (Optional) Human review form if any score < 6
            """)

st.divider()

with st.sidebar:
    st.markdown("### 📋 Team roles")
    st.markdown("""
    - R1: Graph architect
    - R2: RAG engineer
    - R3: Agent engineer
    - R4: Output engineer
    - R5: Integration (you!)
    - R6: Presentation
    """)
    
    st.markdown("### 🗓️ Timeline")
    st.markdown("""
    - **Tue–Wed**: Setup
    - **Thu–Fri**: Integration
    - **Sat–Sun**: Demo rehearsal
    - **Mon**: Presentation
    """)
