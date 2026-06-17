"""
search.py — Art Search Core Functions + CLI Tester
----------------------------------------------------
Two purposes:
  1. Contains all search functions imported by app.py
  2. Can be run directly to test search from the terminal

Usage as CLI tester:
    python search.py

Usage as module (imported by app.py):
    from search import semantic_search, style_search, image_similarity_search

Commands in CLI mode:
    semantic: woman in blue dress      — find by visual description
    style: impressionism               — find all paintings of a style
    refine: warm sunset colors         — narrow down current results
    new                                — clear and start fresh
    quit                               — exit
"""

from pathlib import Path
import chromadb

from models import embed_text, embed_image_pil, predict_style

BASE_DIR   = Path(__file__).parent
CHROMA_DIR = str(BASE_DIR / "chroma_db")

MAX_RESULTS = 20
MIN_SCORE   = 0.0   # lower default than photo organizer — art scores tend to be lower


# ── ChromaDB connection ───────────────────────────────────────────────────────

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection("artworks")


# ── Core search functions (imported by app.py) ────────────────────────────────

def semantic_search(query: str, min_score=MIN_SCORE, n=MAX_RESULTS,
                    limit_to=None, style_filter=None) -> list:
    """
    Find artworks that match a natural language description.

    Examples:
      semantic_search("woman in a garden with soft light")
      semantic_search("dark dramatic scene with candlelight")
      semantic_search("geometric shapes and primary colors")
      semantic_search("stormy seascape with dramatic sky")

    Args:
      query        : natural language description
      min_score    : minimum cosine similarity (0.0 - 1.0)
      n            : max results to return
      limit_to     : list of artwork IDs to search within (for refine)
      style_filter : only return artworks of this style (e.g. "baroque")

    Returns list of dicts with keys: id, score, style, title, path, ocr_text
    """
    collection = get_collection()
    total      = max(collection.count(), 1)
    embedding  = embed_text(query)

    raw = collection.query(
        query_embeddings=[embedding],
        n_results=total
    )

    ids, distances, metas = (
        raw["ids"][0], raw["distances"][0], raw["metadatas"][0]
    )

    # Filter to a subset if refining
    if limit_to:
        limit_set = set(limit_to)
        triples   = [(i, d, m) for i, d, m in zip(ids, distances, metas)
                     if i in limit_set]
        if not triples:
            return []
        ids, distances, metas = zip(*triples)

    results = [
        {
            "id":       i,
            "score":    round(1 - d, 3),
            "style":    m.get("style", "unknown"),
            "title":    m.get("title", ""),
            "path":     m.get("path", ""),
            "ocr_text": m.get("ocr_text", ""),
            "filename": m.get("filename", ""),
        }
        for i, d, m in zip(ids, distances, metas)
        if (1 - d) >= min_score
    ]

    # Filter by style if specified
    if style_filter:
        results = [r for r in results if r["style"] == style_filter]

    return results[:n]


def style_search(style: str, n=MAX_RESULTS) -> list:
    """
    Find all indexed artworks belonging to a specific style.
    Uses ChromaDB metadata filtering — no embedding needed.

    Example:
      style_search("baroque")
      style_search("surrealism")
    """
    collection = get_collection()
    total      = max(collection.count(), 1)

    raw = collection.get(
        where={"style": style},
        limit=min(n, total),
        include=["metadatas"]
    )

    return [
        {
            "id":       i,
            "score":    None,   # no similarity score for pure style browse
            "style":    m.get("style", "unknown"),
            "title":    m.get("title", ""),
            "path":     m.get("path", ""),
            "ocr_text": m.get("ocr_text", ""),
            "filename": m.get("filename", ""),
        }
        for i, m in zip(raw["ids"], raw["metadatas"])
    ]


def image_similarity_search(pil_image, min_score=MIN_SCORE,
                             n=MAX_RESULTS, style_filter=None) -> list:
    """
    Find artworks visually similar to an uploaded image.
    Works for both artwork uploads AND real photos — SigLIP 2 handles both.

    Example use case: upload a photo of a sunset → find paintings with similar mood/colors

    Args:
      pil_image    : PIL Image object
      min_score    : minimum cosine similarity
      n            : max results
      style_filter : optionally limit to one style
    """
    collection = get_collection()
    total      = max(collection.count(), 1)
    embedding  = embed_image_pil(pil_image)

    raw = collection.query(
        query_embeddings=[embedding],
        n_results=total
    )

    ids, distances, metas = (
        raw["ids"][0], raw["distances"][0], raw["metadatas"][0]
    )

    results = [
        {
            "id":       i,
            "score":    round(1 - d, 3),
            "style":    m.get("style", "unknown"),
            "title":    m.get("title", ""),
            "path":     m.get("path", ""),
            "ocr_text": m.get("ocr_text", ""),
            "filename": m.get("filename", ""),
        }
        for i, d, m in zip(ids, distances, metas)
        if (1 - d) >= min_score
    ]

    if style_filter:
        results = [r for r in results if r["style"] == style_filter]

    return results[:n]


def combined_search(pil_image, text_modifier: str,
                    image_weight: float = 0.7,
                    min_score: float = MIN_SCORE,
                    n: int = MAX_RESULTS,
                    style_filter: str = None) -> list:
    """
    Multimodal query — combines an uploaded image with a text modifier
    into a single search vector. This is the core "multimodal search"
    architecture: upload a painting + describe a change, get artworks
    that satisfy both.

    Example:
      combined_search(uploaded_painting, "but with warmer autumn colors")
      combined_search(uploaded_painting, "similar composition but baroque style", image_weight=0.5)
      combined_search(uploaded_painting, "more dramatic lighting", image_weight=0.85)

    Args:
      pil_image     : PIL Image object (the visual anchor of the query)
      text_modifier : natural language description of the desired change
      image_weight  : 0.0-1.0, how much the image should dominate vs text.
                      0.7 = mostly visual match, nudged by the text.
                      0.3 = mostly the text concept, loosely anchored to the image.
      min_score     : minimum cosine similarity
      n             : max results
      style_filter  : optionally limit to one style

    Returns same result format as other search functions, plus a
    'combo_weight' field showing the weights used (useful for the UI).
    """
    import numpy as np

    text_weight = 1.0 - image_weight

    img_emb  = np.array(embed_image_pil(pil_image), dtype="float32")
    text_emb = np.array(embed_text(text_modifier), dtype="float32")

    # Weighted sum, then renormalize to unit length so cosine similarity
    # in ChromaDB still behaves correctly.
    combined = image_weight * img_emb + text_weight * text_emb
    norm     = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm

    collection = get_collection()
    total      = max(collection.count(), 1)

    raw = collection.query(
        query_embeddings=[combined.tolist()],
        n_results=total
    )

    ids, distances, metas = (
        raw["ids"][0], raw["distances"][0], raw["metadatas"][0]
    )

    results = [
        {
            "id":           i,
            "score":        round(1 - d, 3),
            "style":        m.get("style", "unknown"),
            "title":        m.get("title", ""),
            "path":         m.get("path", ""),
            "ocr_text":     m.get("ocr_text", ""),
            "filename":     m.get("filename", ""),
            "combo_weight": f"{image_weight:.0%} image / {text_weight:.0%} text",
        }
        for i, d, m in zip(ids, distances, metas)
        if (1 - d) >= min_score
    ]

    if style_filter:
        results = [r for r in results if r["style"] == style_filter]

    return results[:n]


def predict_artwork_style(image_path: str) -> list:
    """
    Zero-shot style prediction for a single artwork.
    Returns top 3 predicted styles with confidence scores.

    Example output:
      [('impressionism', 0.84), ('romanticism', 0.62), ('baroque', 0.41)]
    """
    from models import embed_image
    embedding = embed_image(image_path)
    return predict_style(embedding, top_n=3)


# ── CLI tester ────────────────────────────────────────────────────────────────

def print_results(results: list, label: str):
    if not results:
        print("  No results found.")
        return
    print(f"\n  {label} — {len(results)} results:")
    print(f"  {'#':<3} {'Score':<7} {'Style':<16} {'Title'}")
    print(f"  {'─'*60}")
    for i, r in enumerate(results[:10], 1):
        score = f"{r['score']:.3f}" if r["score"] is not None else "  —  "
        title = r["title"][:35] if r["title"] else r["filename"][:35]
        print(f"  {i:<3} {score:<7} {r['style']:<16} {title}")


def main():
    print("=" * 55)
    print("  Art Search — CLI Tester")
    print("=" * 55)

    try:
        col = get_collection()
        print(f"  {col.count()} artworks indexed\n")
    except Exception:
        print("  ERROR: No index found. Run python indexer.py first.")
        return

    print("Commands:")
    print("  semantic: woman reading by window")
    print("  style: baroque")
    print("  refine: warm colors")
    print("  new    — clear results")
    print("  quit   — exit\n")

    current_results = None

    while True:
        prompt = "Search> " if current_results is None \
            else f"Refine ({len(current_results)} results)> "

        try:
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "quit":
            break

        elif cmd == "new":
            current_results = None
            print("Cleared.\n")

        elif cmd.startswith("semantic:"):
            query = user_input[len("semantic:"):].strip()
            current_results = semantic_search(query, n=10)
            print_results(current_results, f'Semantic: "{query}"')

        elif cmd.startswith("style:"):
            style = user_input[len("style:"):].strip().lower()
            current_results = style_search(style, n=10)
            print_results(current_results, f'Style: "{style}"')

        elif cmd.startswith("refine:"):
            if current_results is None:
                print("  Nothing to refine — search first.\n")
                continue
            query   = user_input[len("refine:"):].strip()
            ids     = [r["id"] for r in current_results]
            current_results = semantic_search(query, n=10, limit_to=ids)
            print_results(current_results, f'Refined: "{query}"')

        else:
            print("  Unknown command. Try: semantic: <query>  |  style: <style>  |  refine: <query>  |  new  |  quit\n")

    print("Bye!")


if __name__ == "__main__":
    main()