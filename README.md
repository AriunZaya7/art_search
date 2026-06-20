# Semantic Art Search

A cross-modal retrieval system for searching a collection of paintings using natural language — built for an NLP & Computer Vision course project.

Type a description like *"woman reading by a window with soft light"* or *"dark dramatic scene with candlelight"* and retrieve matching paintings from an indexed collection, without any manual tagging.

---

## How it works

Every image and every text query is encoded into the **same 1152-dimensional vector space** using [SigLIP 2](https://huggingface.co/timm/ViT-SO400M-16-SigLIP2-384) (`ViT-SO400M-16-SigLIP2-384`). Search is just nearest-neighbor lookup in that shared space via cosine similarity, powered by [ChromaDB](https://www.trychroma.com/).

```
Text query  ──► SigLIP 2 text encoder   ──┐
                                            ├──► shared 1152-dim space ──► ChromaDB (cosine similarity) ──► ranked results
Image       ──► SigLIP 2 image encoder  ──┘
```

No fine-tuning is used — SigLIP 2 is applied zero-shot as a pretrained feature extractor.

---

## Dataset

**416 paintings** across six art movements, sourced from [Wikimedia Commons](https://commons.wikimedia.org/) via its public API (stable URLs, public-domain licensing):

| Style | Example artists |
|---|---|
| Impressionism | Monet, Renoir, Degas |
| Expressionism | Munch, Kirchner, Schiele |
| Surrealism | Dalí, Magritte, Kahlo |
| Baroque | Caravaggio, Rembrandt, Vermeer |
| Abstract | Pollock, Rothko, Mondrian |
| Romanticism | Friedrich, Turner, Goya |

Each style lives in its own subfolder under `gallery/`, which also doubles as a ground-truth label for evaluation.

---

## Features

- **🔍 Search** — natural language → matching artworks, with chained/refined queries
- **🖼 Browse Style** — browse the collection filtered by movement
- **🎨 Similar Art** — upload any image → find visually similar paintings, plus a zero-shot style prediction
- **🌌 Embedding Space** — 2D UMAP visualization of the real embedding space, color-coded by style
- **📊 Evaluate** — interactive golden-set builder with live Precision@K / Recall@K

---

## Project structure

```
art_search/
├── gallery/              # dataset, organized by style subfolder
├── chroma_db/            # vector index (auto-created by indexer.py)
├── models.py             # SigLIP 2 loading + embedding functions (single source of truth)
├── indexer.py            # builds the ChromaDB index from gallery/
├── search.py             # retrieval functions used by the app
├── app.py                # Streamlit UI
├── evaluate.py           # Precision@K / Recall@K / MAP evaluation
├── silhouette_analysis.py# cluster-separation analysis on the real embeddings
├── download_wikiart.py   # dataset downloader (Wikimedia Commons API)
└── requirements.txt
```

---

## Setup

```bash
git clone <this-repo-url>
cd art_search
pip install -r requirements.txt
```

> First run downloads SigLIP 2's weights (~3GB) — this only happens once.

### 1. Get the dataset

```bash
python download_wikiart.py
```

Downloads 416 images into `gallery/<style>/`. Safe to re-run — already-downloaded images are skipped.

### 2. Build the index

```bash
python indexer.py
```

Embeds every image with SigLIP 2 and stores the vectors + metadata in `chroma_db/`. Safe to re-run — already-indexed images are skipped, so an interrupted run can simply be restarted.

### 3. Run the app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Try a search like:

```
woman reading by a window with soft light
dark dramatic scene with candlelight chiaroscuro
stormy seascape with turbulent waves
```

---

## Evaluation

To measure retrieval quality on your own golden set:

1. Open the app's **📊 Evaluate** tab
2. Run a query, check the boxes on genuinely relevant results, click "Add to golden set"
3. Repeat for at least 5 different queries
4. Copy the generated `GOLDEN_SET` block into `evaluate.py`
5. Run:

```bash
python evaluate.py
```

This prints Precision@K, Recall@K, and MAP (Mean Average Precision) for each query plus the aggregate.

To check how well-separated each art style is in the embedding space (independent of any 2D-plotting artifacts):

```bash
python silhouette_analysis.py
```

---

## Notes & limitations

- Text→image similarity scores in this embedding space are typically low in absolute terms (often under 0.15) even for correct matches — this is expected behavior for cross-modal models, not a bug. Relative ranking matters more than the absolute score.
- The embedding space separates some styles (e.g. baroque, surrealism) more cleanly than others (e.g. abstract) — see `silhouette_analysis.py` for a quantified breakdown.
- This architecture is domain-agnostic: nothing in `models.py`, `indexer.py`, or `search.py` is art-specific. Swapping the `gallery/` folder and style labels would adapt the same pipeline to other domains (e.g. industrial parts search, product catalogs).

---

## Stack

`SigLIP 2` · `ChromaDB` · `Streamlit` · `UMAP` · `Wikimedia Commons API`