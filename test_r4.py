"""
Quick R4 chain smoke test — run with: python test_r4.py

Wrapped in __main__ guard so pytest does not collect and execute this
file during test discovery (it would crash on SystemExit).
"""
import httpx

_o = httpx.Client.__init__


def _f(self, *a, **kw):
    kw["verify"] = False
    _o(self, *a, **kw)


httpx.Client.__init__ = _f


if __name__ == "__main__":
    from dotenv import load_dotenv  # noqa: E402
    load_dotenv()

    from graph.nodes import (  # noqa: E402
        ingest_node, extract_node, score_node, write_memo_node,
    )

    print("--- INGEST ---")
    s = ingest_node({"pdf_path": "data/sample_pitch.pdf"})
    print(f"Chars extracted: {len(s.get('raw_text', ''))}")
    if s.get("error"):
        print(f"ERROR: {s['error']}")
        raise SystemExit(1)

    print("\n--- EXTRACT ---")
    s.update(extract_node(s))
    if s.get("error"):
        print(f"ERROR: {s['error']}")
        raise SystemExit(1)
    for k, v in s["extracted_claims"].items():
        print(f"  {k}: {str(v)[:90]}")

    print("\n--- SCORE ---")
    s["validation_results"] = {k: "PLAUSIBLE" for k in s["extracted_claims"]}
    s.update(score_node(s))
    if s.get("error"):
        print(f"ERROR: {s['error']}")
        raise SystemExit(1)
    sc = s["scores"]
    print(f"  market={sc['market']}  team={sc['team']}"
          f"  traction={sc['traction']}  product={sc['product']}")
    print(f"  composite={sc['composite_score']}  confidence={sc['confidence']}")
    print(f"  HITL needed: {s['human_review_required']}")

    print("\n--- MEMO ---")
    s.update(write_memo_node(s))
    if s.get("error"):
        print(f"ERROR: {s['error']}")
        raise SystemExit(1)
    print(s["investment_memo"])
