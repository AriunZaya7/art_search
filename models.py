"""
models.py — Art Search Project
--------------------------------
Single source of truth for all ML models.
Both indexer.py and app.py import from here — guarantees the same
vector space is used for indexing AND querying.

Model: SigLIP 2 ViT-SO400M-16-384
  - Released by Google, February 2025
  - Outperforms SigLIP 1 and CLIP on retrieval benchmarks
  - Handles abstract aesthetic descriptions well (ideal for art search)
  - First run downloads ~3GB, loads from cache instantly after

Requirements:
    pip install open-clip-torch timm torch Pillow easyocr
"""

import os
import warnings
import torch
import open_clip
from PIL import Image

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*symlinks.*")
warnings.filterwarnings("ignore", message=".*QuickGELU.*")

# ── Model ─────────────────────────────────────────────────────────────────────
_model      = None
_preprocess = None
_tokenizer  = None


def _load():
    """Load SigLIP 2 once — reused for all subsequent calls."""
    global _model, _preprocess, _tokenizer
    if _model is None:
        print("Loading SigLIP 2 (first run downloads ~3GB)...")
        _model, _preprocess = open_clip.create_model_from_pretrained(
            "hf-hub:timm/ViT-SO400M-16-SigLIP2-384"
        )
        _tokenizer = open_clip.get_tokenizer(
            "hf-hub:timm/ViT-SO400M-16-SigLIP2-384"
        )
        _model.eval()
        print("SigLIP 2 ready.")


# ── Image embedding ───────────────────────────────────────────────────────────

def embed_image(image_path: str) -> list:
    """
    Returns a normalized SigLIP 2 embedding for an image file.
    Used by indexer.py to embed artwork images.
    """
    _load()
    img = _preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        feat = _model.encode_image(img)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat[0].tolist()


def embed_image_pil(pil_image) -> list:
    """
    Returns a normalized SigLIP 2 embedding for a PIL Image object.
    Used by app.py for image-to-image search (upload a painting, find similar).
    """
    _load()
    img = _preprocess(pil_image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        feat = _model.encode_image(img)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat[0].tolist()


# ── Text embedding ────────────────────────────────────────────────────────────

def embed_text(text: str) -> list:
    """
    Returns a normalized SigLIP 2 embedding for a text query.
    Used by app.py to embed search queries like:
      'impressionist painting with warm sunset colors'
      'dark dramatic baroque chiaroscuro'
      'abstract geometric shapes primary colors'
    """
    _load()
    tokens = _tokenizer([text])
    with torch.no_grad():
        feat = _model.encode_text(tokens)
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat[0].tolist()


# ── Style classification ──────────────────────────────────────────────────────
# Zero-shot style detection using SigLIP 2.
# Compares an image embedding against text descriptions of each art style.
# No training needed — SigLIP 2 already understands these concepts.

ART_STYLES = {
    "impressionism":  "an impressionist painting with loose brushstrokes, soft light, and everyday scenes",
    "expressionism":  "an expressionist painting with bold distorted colors conveying intense emotion",
    "surrealism":     "a surrealist painting with dreamlike impossible scenes and strange symbolic imagery",
    "baroque":        "a baroque painting with dramatic lighting, rich colors, and religious or mythological themes",
    "abstract":       "an abstract painting with geometric shapes, pure colors, and no recognizable objects",
    "romanticism":    "a romantic painting with dramatic landscapes, heroic figures, and intense emotional atmosphere",
}

_style_embeddings = None


def get_style_embeddings() -> dict:
    """Pre-compute and cache style text embeddings (called once at startup)."""
    global _style_embeddings
    if _style_embeddings is None:
        _load()
        _style_embeddings = {
            style: embed_text(description)
            for style, description in ART_STYLES.items()
        }
    return _style_embeddings


def predict_style(image_embedding: list, top_n: int = 3) -> list:
    """
    Given an image embedding, returns the top N most likely art styles
    as a list of (style_name, confidence_score) tuples.

    Uses cosine similarity between the image and style text descriptions.
    This is zero-shot classification — no training required.

    Example output:
      [('impressionism', 0.82), ('romanticism', 0.61), ('baroque', 0.45)]
    """
    style_embs = get_style_embeddings()
    img_tensor = torch.tensor(image_embedding)

    scores = {}
    for style, style_emb in style_embs.items():
        style_tensor = torch.tensor(style_emb)
        # Cosine similarity = dot product of normalized vectors
        score = float(torch.dot(img_tensor, style_tensor).item())
        scores[style] = round(score, 3)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return ranked[:top_n]


# ── OCR ───────────────────────────────────────────────────────────────────────
# Useful for extracting text from museum labels, signatures, or inscriptions
# visible in artwork images.

_ocr_reader = None


def extract_text(image_path: str) -> str:
    """
    Extracts any visible text from an artwork image using EasyOCR.
    Catches signatures, museum labels, inscriptions, or dates.
    Returns empty string if no text found or on error.
    """
    global _ocr_reader
    if _ocr_reader is None:
        print("Loading OCR model...")
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        print("OCR ready.")
    try:
        results = _ocr_reader.readtext(image_path, detail=0)
        return " ".join(results).strip()
    except Exception:
        return ""