"""
download_wikiart.py — Art Dataset Downloader (Wikimedia Commons)
----------------------------------------------------------------
Downloads 100 paintings per style from Wikimedia Commons API.
Much more reliable than direct WikiArt URLs — images are served
from stable Wikimedia CDN with proper public domain licensing.

Styles: impressionism, expressionism, surrealism,
        baroque, abstract, romanticism

Output: gallery/<style>/<filename>.jpg  (600 images total)

Requirements:
    pip install requests Pillow tqdm

Usage:
    python download_wikiart.py
"""

import time
import random
import requests
from pathlib import Path
from io import BytesIO
from PIL import Image
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
GALLERY_DIR      = Path("gallery")
IMAGES_PER_STYLE = 100
MIN_IMAGE_PX     = 200
REQUEST_TIMEOUT  = 15
DELAY_BETWEEN    = 0.5      # be polite to Wikimedia servers
RANDOM_SEED      = 42

HEADERS = {
    "User-Agent": "ArtSearchProject/1.0 (student research project; contact: student@university.edu)"
}

# ── Wikimedia Commons search categories per style ─────────────────────────────
# Each entry is a list of Wikimedia Commons categories to pull images from.
# Multiple categories per style gives us more variety.

STYLE_CATEGORIES = {
    "impressionism": [
        "Paintings_by_Claude_Monet",
        "Paintings_by_Pierre-Auguste_Renoir",
        "Paintings_by_Edgar_Degas",
        "Paintings_by_Camille_Pissarro",
        "Paintings_by_Alfred_Sisley",
        "Paintings_by_Berthe_Morisot",
        "Paintings_by_Gustave_Caillebotte",
    ],
    "expressionism": [
        "Paintings_by_Edvard_Munch",
        "Paintings_by_Ernst_Ludwig_Kirchner",
        "Paintings_by_Egon_Schiele",
        "Paintings_by_Oskar_Kokoschka",
        "Paintings_by_Emil_Nolde",
        "Paintings_by_Max_Beckmann",
        "Paintings_by_Franz_Marc",
    ],
    "surrealism": [
        "Paintings_by_Salvador_Dalí",
        "Paintings_by_René_Magritte",
        "Paintings_by_Frida_Kahlo",
        "Paintings_by_Max_Ernst",
        "Paintings_by_Joan_Miró",
        "Paintings_by_Giorgio_de_Chirico",
        "Paintings_by_Yves_Tanguy",
    ],
    "baroque": [
        "Paintings_by_Caravaggio",
        "Paintings_by_Rembrandt_van_Rijn",
        "Paintings_by_Johannes_Vermeer",
        "Paintings_by_Peter_Paul_Rubens",
        "Paintings_by_Diego_Velázquez",
        "Paintings_by_Artemisia_Gentileschi",
        "Paintings_by_Nicolas_Poussin",
    ],
    "abstract": [
        "Paintings_by_Wassily_Kandinsky",
        "Paintings_by_Piet_Mondrian",
        "Paintings_by_Kazimir_Malevich",
        "Paintings_by_Jackson_Pollock",
        "Paintings_by_Mark_Rothko",
        "Paintings_by_Paul_Klee",
        "Paintings_by_Franz_Kline",
    ],
    "romanticism": [
        "Paintings_by_Caspar_David_Friedrich",
        "Paintings_by_Eugène_Delacroix",
        "Paintings_by_J._M._W._Turner",
        "Paintings_by_Francisco_Goya",
        "Paintings_by_Théodore_Géricault",
        "Paintings_by_John_Constable",
        "Paintings_by_William_Blake",
    ],
}

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"


# ── Wikimedia API helpers ─────────────────────────────────────────────────────

def get_image_urls_from_category(category: str, limit: int = 50) -> list:
    """
    Fetches image file names from a Wikimedia Commons category.
    Returns list of (title, url) tuples.
    """
    params = {
        "action":      "query",
        "list":        "categorymembers",
        "cmtitle":     f"Category:{category}",
        "cmtype":      "file",
        "cmlimit":     limit,
        "format":      "json",
    }
    try:
        r = requests.get(WIKIMEDIA_API, params=params,
                         headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        members = r.json().get("query", {}).get("categorymembers", [])
        titles  = [m["title"] for m in members if m["title"].lower().endswith(
            (".jpg", ".jpeg", ".png")
        )]
        return titles
    except Exception as e:
        print(f"  API error for {category}: {e}")
        return []


def get_image_direct_url(file_title: str) -> str | None:
    """
    Given a Wikimedia file title like 'File:Monet_waterlilies.jpg',
    returns the direct image URL via imageinfo API.
    """
    params = {
        "action":  "query",
        "titles":  file_title,
        "prop":    "imageinfo",
        "iiprop":  "url",
        "iiurlwidth": 800,     # request 800px wide thumbnail — fast to download
        "format":  "json",
    }
    try:
        r = requests.get(WIKIMEDIA_API, params=params,
                         headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            info = page.get("imageinfo", [])
            if info:
                return info[0].get("thumburl") or info[0].get("url")
    except Exception:
        pass
    return None


def download_image(url: str) -> Image.Image | None:
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        if img.width < MIN_IMAGE_PX or img.height < MIN_IMAGE_PX:
            return None
        return img
    except Exception:
        return None


def safe_filename(file_title: str, style: str, idx: int) -> str:
    """Convert 'File:Claude Monet - Water Lilies.jpg' to a safe filename."""
    name = file_title.replace("File:", "").replace(" ", "_")
    # Remove characters that cause issues on Windows
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(ch, "_")
    return f"{style}_{idx:03d}_{name}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)

    print("Art Dataset Downloader — Wikimedia Commons")
    print(f"Target: {len(STYLE_CATEGORIES)} styles × {IMAGES_PER_STYLE} = "
          f"{len(STYLE_CATEGORIES) * IMAGES_PER_STYLE} images\n")

    style_counts = {}

    for style, categories in STYLE_CATEGORIES.items():
        style_dir = GALLERY_DIR / style
        style_dir.mkdir(parents=True, exist_ok=True)

        existing      = list(style_dir.glob("*.jpg"))
        already_count = len(existing)

        if already_count >= IMAGES_PER_STYLE:
            print(f"✓ {style:<22} already has {already_count} images, skipping")
            style_counts[style] = already_count
            continue

        needed = IMAGES_PER_STYLE - already_count
        print(f"\n{style.upper()} — need {needed} more images")

        # Collect file titles from all categories for this style
        all_titles = []
        for cat in categories:
            print(f"  Fetching list from: {cat} ...", end=" ")
            titles = get_image_urls_from_category(cat, limit=30)
            print(f"{len(titles)} files found")
            all_titles.extend(titles)
            time.sleep(0.3)

        # Deduplicate and shuffle
        all_titles = list(set(all_titles))
        random.shuffle(all_titles)

        if not all_titles:
            print(f"  ⚠ No images found for {style}")
            style_counts[style] = already_count
            continue

        downloaded  = 0
        start_idx   = already_count
        pbar        = tqdm(total=needed, desc=f"  Downloading", unit="img")

        for file_title in all_titles:
            if downloaded >= needed:
                break

            # Get direct download URL
            url = get_image_direct_url(file_title)
            time.sleep(DELAY_BETWEEN)

            if not url:
                continue

            img = download_image(url)
            if img is None:
                continue

            fname = safe_filename(file_title, style, start_idx + downloaded)
            dest  = style_dir / fname
            img.save(dest, "JPEG", quality=90)
            downloaded += 1
            pbar.update(1)

        pbar.close()

        actual = len(list(style_dir.glob("*.jpg")))
        style_counts[style] = actual

        if downloaded < needed:
            print(f"  ⚠ Got {actual}/{IMAGES_PER_STYLE} "
                  f"({needed - downloaded} short — try adding more categories)")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = sum(style_counts.values())
    print(f"\n{'─'*45}")
    print(f"{'Style':<22} {'Images':>8}")
    print(f"{'─'*45}")
    for style, count in style_counts.items():
        status = "✓" if count >= IMAGES_PER_STYLE else f"⚠ {count}/{IMAGES_PER_STYLE}"
        print(f"{style:<22} {count:>8}  {status}")
    print(f"{'─'*45}")
    print(f"{'TOTAL':<22} {total:>8}")
    print(f"\nImages saved to: {GALLERY_DIR.resolve()}")

    if total >= 300:
        print("\nNext step: python indexer.py")
    else:
        print("\n⚠ Dataset is small. Re-run to retry failed downloads.")


if __name__ == "__main__":
    main()