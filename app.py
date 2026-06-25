import streamlit as st
import tempfile
import os
from pathlib import Path
from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

# Import the real graph
from graph import graph, get_graph_image, PitchState

st.set_page_config(page_title="VC Pitch Evaluator", layout="wide")
st.title("💼 VC Startup Pitch Evaluator")

st.markdown("""
This system evaluates startup pitch decks using AI agents and LangGraph orchestration.
Upload a PDF pitch deck and get an investment recommendation memo within seconds.
""")

st.divider()

# Sidebar: Graph visualization
with st.sidebar:
    st.subheader("📊 Agent Pipeline")
    try:
        st.image(get_graph_image(graph), caption="Live LangGraph Architecture")
    except Exception as e:
        st.warning(f"Graph visualization unavailable: {e}")
    
    st.markdown("### 🔍 How it works")
    st.markdown("""
    1. **Ingest** — PDF → Extract text
    2. **Extract** — Gemini finds 4 key claims
    3. **Validate** — DuckDuckGo searches, Gemini assesses
    4. **Score** — Each dimension 0-10
    5. **Review** — Human analyst (if needed)
    6. **Memo** — Investment recommendation
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📄 Upload Pitch Deck")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

with col2:
    st.subheader("Status")
    status_placeholder = st.empty()

if uploaded_file:
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name
    
    st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    if st.button("🚀 Analyze Pitch Deck", use_container_width=True):
        status_placeholder.info("⏳ Processing... Running agent pipeline")
        
        # Initialize session state for tracking graph execution
        if "thread_id" not in st.session_state:
            st.session_state.thread_id = str(hash(uploaded_file.name))
        
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        initial_state = {"pdf_path": pdf_path}
        
        # Create containers for live updates
        progress_container = st.container()
        results_container = st.container()
        
        try:
            # Run the graph
            with progress_container:
                st.info("⏳ Starting graph execution...")
                result = graph.invoke(initial_state, config)
            
            status_placeholder.success("✅ Analysis complete!")
            
            # Check if graph paused for human review
            if "__interrupt__" in result:
                with results_container:
                    st.warning("⚠️ Human Review Required")
                    interrupt_data = result["__interrupt__"][0].value
                    
                    st.subheader("📊 Scores")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Market", interrupt_data.get("scores", {}).get("market", "—"))
                    with col2:
                        st.metric("Team", interrupt_data.get("scores", {}).get("team", "—"))
                    with col3:
                        st.metric("Product", interrupt_data.get("scores", {}).get("product", "—"))
                    with col4:
                        st.metric("Traction", interrupt_data.get("scores", {}).get("traction", "—"))
                    
                    st.subheader("📝 Analyst Review Form")
                    st.markdown("**One or more scores fell below 6. Review and provide feedback:**")
                    
                    feedback = st.text_area(
                        "Your feedback on the pitch:",
                        placeholder="E.g., 'Team is exceptional but traction needs more time to validate. Recommend proceeding.'",
                        height=100
                    )
                    
                    if st.button("✅ Submit Review", use_container_width=True):
                        # Resume graph with human feedback
                        final_result = graph.invoke(Command(resume=feedback), config)
                        
                        st.success("✅ Analysis complete!")
                        
                        st.subheader("📧 Investment Memo")
                        memo = final_result.get("investment_memo", "No memo generated")
                        st.text(memo)
                        
                        # Offer download
                        st.download_button(
                            label="📥 Download Memo",
                            data=memo,
                            file_name="investment_memo.txt",
                            mime="text/plain"
                        )
            
            else:
                # No human review needed — show results directly
                with results_container:
                    st.subheader("📊 Scores")
                    scores = result.get("scores", {})
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Market", scores.get("market", "—"))
                    with col2:
                        st.metric("Team", scores.get("team", "—"))
                    with col3:
                        st.metric("Product", scores.get("product", "—"))
                    with col4:
                        st.metric("Traction", scores.get("traction", "—"))
                    
                    st.subheader("📧 Investment Memo")
                    memo = result.get("investment_memo", "No memo generated")
                    st.text(memo)
                    
                    # Offer download
                    st.download_button(
                        label="📥 Download Memo",
                        data=memo,
                        file_name="investment_memo.txt",
                        mime="text/plain"
                    )
        
        except Exception as e:
            status_placeholder.error(f"❌ Error during analysis: {str(e)}")
            st.error(f"**Error Details:**\n{str(e)}")
            import traceback
            with st.expander("Technical Details"):
                st.code(traceback.format_exc())
        
        finally:
            # Clean up temp file
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

st.divider()

with st.expander("ℹ️ About this project"):
    st.markdown("""
    **VC Startup Pitch Evaluator**
    
    Built with:
    - LangGraph StateGraph with 6 agents
    - LangChain RAG pipeline (PyPDF + Chroma)
    - Gemini 2.5 Flash as the LLM
    - DuckDuckGo for web search validation
    - Human-in-the-loop interrupts for oversight
    
    Course: Generative AI — IE University MBDS  
    Professor: Conchita Diaz Cantarero (Head of AI Education, Google Cloud EMEA)
    """)
