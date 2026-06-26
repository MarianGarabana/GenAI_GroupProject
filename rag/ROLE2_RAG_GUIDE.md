# Role 2 — RAG Pipeline Engineer Guide

Your role is the PDF ingestion and Chroma retrieval layer for the VC Startup Pitch Evaluator.
The professor's role card asks you to own:

- `rag/ingest.py`
- `rag/retriever.py`
- `rag/extractor_chain.py`
- `data/sample_pitch.pdf`

## What the files do

### `rag/ingest.py`
Loads the pitch deck PDF with `pypdf.PdfReader`, cleans page text, chunks it with `RecursiveCharacterTextSplitter`, embeds the chunks, and persists them into a Chroma collection.

### `rag/retriever.py`
Opens the persisted Chroma vector store and returns a retriever. It uses Gemini embeddings when `GOOGLE_API_KEY` is set. It also includes `HashEmbeddings`, a deterministic local fallback so tests and demos can run without an API key.

### `rag/extractor_chain.py`
Builds the LCEL chain:

```text
Chroma retriever -> context formatter -> ChatPromptTemplate -> Gemini -> PydanticOutputParser
```

It returns a `StartupClaims` object with:

- `market_size`
- `team_background`
- `traction`
- `product_description`

### `data/sample_pitch.pdf`
A small text-based pitch deck used for smoke tests and live demo rehearsal.

## How to run your part

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your Gemini key to `.env`:

```bash
GOOGLE_API_KEY=your_key_here
```

For offline tests without Gemini, set:

```bash
USE_LOCAL_EMBEDDINGS=true
```

Run tests:

```bash
pytest tests/test_rag_pipeline.py -q
```

Try extraction manually:

```bash
python - <<'PY'
from rag.extractor_chain import extract_claims_from_pdf
claims = extract_claims_from_pdf('data/sample_pitch.pdf')
print(claims.model_dump_json(indent=2))
PY
```

## What to say in the presentation

"My section is the private knowledge layer. The uploaded pitch deck is loaded with pypdf directly — no deprecated community wrappers — split into overlapping chunks, embedded, and persisted in Chroma. Then the extractor chain retrieves only the most relevant chunks and asks Gemini to return structured JSON claims. This reduces hallucination because the model is grounded in the founder's actual deck instead of relying on general training knowledge."

## Integration handoff to Role 1 / Role 5

The easiest graph integration is:

```python
from rag.extractor_chain import extract_claims_dict

claims = extract_claims_dict(state['pdf_path'])
return {'extracted_claims': claims, 'error': None}
```

If the team wants separate ingest and extract nodes, use:

```python
from rag.ingest import ingest_pdf_to_chroma
from rag.retriever import get_retriever
from rag.extractor_chain import build_extraction_chain
```

## Quality checklist

- PDF is text-based, not scanned images.
- `data/sample_pitch.pdf` extracts text correctly.
- Chroma folder is ignored by git.
- `GOOGLE_API_KEY` is never committed.
- The output is structured JSON/Pydantic, not free text.


## Claude review fixes applied

- `retrieve_context(max_chars=6000)` now always returns at least a truncated first chunk when documents exist. This prevents an empty context if the first retrieved chunk is large.
- `load_pdf_documents()` preserves meaningful newlines so `RecursiveCharacterTextSplitter` can split on slide/bullet structure.
- `ingest_and_get_retriever()` now correctly passes `chunk_size` and `chunk_overlap`.
- `extract_claims_dict` is exported from `rag/__init__.py` for easy graph integration.
- Tests now use an absolute sample PDF path and include a regression test for context truncation.

## Integration warning for the team

The running LangGraph/Streamlit demo must call Role 2's RAG handoff function, ideally:

```python
from rag import extract_claims_dict

claims = extract_claims_dict(pdf_path, persist_directory="chroma_db")
```

If `graph/nodes.py` directly reads the whole PDF and sends all text to Gemini, then Chroma retrieval and the structured `StartupClaims` schema are bypassed. Raise this in group sync so Role 1/5 can wire Role 2 into the graph before the final demo.
