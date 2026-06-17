"""
indexer.py — Art Search Indexer
Indexes all artwork images in gallery/ into ChromaDB.

Differences from photo organizer version:
  - Reads style from the subfolder name (gallery/impressionism/, etc.)
  - Stores artist name and title parsed from filename
  - Stores style as metadata so you can filter by style in the app
  - Safe to re-run — already indexed images are skipped

Run:  python indexer.py
"""

import re
import datetime
from pathlib import Path

import chromadb

from models import embed_image

BASE_DIR = Path(__file__).parent
GALLERY_DIR = BASE_DIR / "gallery"
CHROMA_DIR = str(BASE_DIR / "chroma_db")
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_filename(fname: str) -> dict:
    """
    Extract artist and title from WikiArt filename convention.
    Example: 'impressionism_001_the-water-lilies-1906.jpg'
             → artist guessed from style, title = 'the water lilies 1906'
    Also handles: 'claude-monet_water-lilies-1906.jpg'
    """
    stem = Path(fname).stem  # remove .jpg

    # Remove style prefix and index if present (e.g. 'impressionism_001_')
    stem = re.sub(r'^[a-z]+_\d+_', '', stem)

    # Replace hyphens with spaces, clean up
    readable = stem.replace("-", " ").replace("_", " ").strip()

    return {"title": readable}


def get_style_from_path(img_path: Path) -> str:
    """
    Returns the parent folder name as the style label.
    gallery/impressionism/monet.jpg → 'impressionism'
    If image is directly in gallery/ with no subfolder, returns 'unknown'.
    """
    parent = img_path.parent
    if parent.name == "gallery" or parent == GALLERY_DIR:
        return "unknown"
    return parent.name


# ── ChromaDB setup ────────────────────────────────────────────────────────────

def setup_chroma():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name="artworks",           # different collection name from photo organizer
        metadata={"hnsw:space": "cosine"}
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Collect all images recursively (includes style subfolders)
    all_files = sorted([
        f for f in GALLERY_DIR.rglob("*")
        if f.suffix.lower() in SUPPORTED_EXT
    ])

    if not all_files:
        print(f"No images found in {GALLERY_DIR}")
        print("Run download_wikiart.py first.")
        return

    # Count per style for the summary
    style_counts = {}
    for f in all_files:
        style = get_style_from_path(f)
        style_counts[style] = style_counts.get(style, 0) + 1

    print(f"Found {len(all_files)} images across {len(style_counts)} styles:")
    for style, count in sorted(style_counts.items()):
        print(f"  {style:<22} {count} images")
    print()

    collection = setup_chroma()

    # Check what's already indexed
    existing = set()
    if collection.count() > 0:
        existing = set(collection.get(include=[])["ids"])
    print(f"Already indexed : {len(existing)}")
    print(f"To index        : {len(all_files) - len(existing)}\n")

    indexed = skipped = errors = 0

    for i, img_path in enumerate(all_files, 1):
        # Use relative path as ID so it's unique across styles
        # e.g. "impressionism/impressionism_001_water-lilies.jpg"
        relative = img_path.relative_to(GALLERY_DIR)
        img_id   = str(relative).replace("\\", "/")  # normalize Windows paths

        if img_id in existing:
            print(f"[{i}/{len(all_files)}] skip  {img_id}")
            skipped += 1
            continue

        print(f"[{i}/{len(all_files)}] index {img_id} ...", end=" ", flush=True)

        try:
            # Visual embedding via SigLIP 2
            embedding = embed_image(str(img_path))
            ocr_text = ""  # OCR not needed for paintings

        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Done ─────────────────────────────────────")
    print(f"Indexed  : {indexed}")
    print(f"Skipped  : {skipped}")
    print(f"Errors   : {errors}")
    print(f"Total in DB : {collection.count()}")
    print(f"\nNext step: streamlit run app.py")


if __name__ == "__main__":
    main()