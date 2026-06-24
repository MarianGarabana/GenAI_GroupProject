# VC Startup Pitch Evaluator

An AI system that evaluates startup pitch decks using LangGraph orchestration, multi-agent reasoning, and RAG-based claim extraction.

## What it does

1. **Ingestion**: Upload a startup pitch deck (PDF)
2. **Extraction**: AI extracts key claims (market size, team, traction, product)
3. **Validation**: An agent searches the web to validate each claim
4. **Scoring**: Each dimension scored 0–10 based on evidence
5. **Review**: If any score < 6, human analyst reviews before final output
6. **Memo**: Investment recommendation memo generated

## Quick start

```bash
# Clone and enter the repo
git clone https://github.com/MarianGarabana/GenAI_GroupProject.git
cd GenAI_GroupProject

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# Run the Streamlit app
streamlit run app.py
```

## Architecture

LangGraph StateGraph with 7 nodes:
- **Ingestion**: PyPDF → text splitter → Chroma embeddings
- **Extractor**: RAG chain — retrieve from Chroma → prompt → Gemini → structured JSON
- **Validator**: AgentExecutor with DuckDuckGo + Wikipedia tools
- **Scorer**: LCEL chain — state → Gemini → JSON scores
- **Writer**: LCEL chain — state → Gemini → investment memo
- **Conditional edge**: if any score < 6 → human review; else → finish
- **Human review**: LangGraph interrupt — analyst reviews, approves, graph resumes

## Team roles

- **Role 1**: Graph architect (LangGraph, StateGraph, edges, human interrupt)
- **Role 2**: RAG engineer (PyPDF, text splitter, Chroma, claim extraction)
- **Role 3**: Agent engineer (AgentExecutor, DuckDuckGo, Wikipedia, memory)
- **Role 4**: Output engineer (scorer + memo writer LCEL chains)
- **Role 5**: Integration lead (repo, Streamlit, end-to-end tests, demo)
- **Role 6**: Presentation lead (slides, business case, architecture story)

## Development timeline

- **Tue–Wed**: Roles 1, 2, 5 set up scaffold
- **Thu–Sun**: All roles work in parallel, integrate Friday, demo dry-runs Sunday
- **Mon**: Final push, demo, presentation
