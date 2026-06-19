"""
test_combined_search.py
Quick sanity test for combined_search() — run this BEFORE wiring it into app.py.

Tests three things:
  1. Pure image weight (1.0) should match image_similarity_search() results closely
  2. Pure text weight (0.0) should match semantic_search() results closely
  3. A mid-weight (0.5) should produce a DIFFERENT ranking than either extreme

If all three checks pass, the function is working correctly and safe to wire into the UI.

Usage:
    python test_combined_search.py
"""

from pathlib import Path
from PIL import Image

from search import combined_search, semantic_search, image_similarity_search

# ── Pick any indexed image as your test reference ─────────────────────────────
# Change this to any real path in your gallery/ folder
TEST_IMAGE_PATH = "gallery/baroque/baroque_000_Peter_Paul_Rubens_(1577-1640)_(after)_-_Battle_of_the_Amazons_-_1941.18-94_-_Calderdale_Metropolitan_Borough_Council.jpg"
TEST_MODIFIER    = "but with bright cheerful colors"


def print_top5(label, results):
    print(f"\n{label}")
    print("-" * 60)
    if not results:
        print("  (no results)")
        return
    for r in results[:5]:
        print(f"  {r['score']:.3f}  [{r['style']:<14}]  {r['filename']}")


def main():
    test_path = Path(TEST_IMAGE_PATH)
    if not test_path.exists():
        print(f"ERROR: {TEST_IMAGE_PATH} not found.")
        print("Edit TEST_IMAGE_PATH at the top of this script to point at a real image in your gallery/.")
        return

    pil = Image.open(test_path).convert("RGB")
    print(f"Test image : {test_path.name}")
    print(f"Modifier   : \"{TEST_MODIFIER}\"")

    # ── Test 1: pure image weight (1.0) ───────────────────────────────────────
    pure_image = combined_search(pil, TEST_MODIFIER, image_weight=1.0, min_score=0.0, n=10)
    baseline_image = image_similarity_search(pil, min_score=0.0, n=10)

    pure_image_ids = [r["id"] for r in pure_image[:5]]
    baseline_ids   = [r["id"] for r in baseline_image[:5]]
    overlap_1 = len(set(pure_image_ids) & set(baseline_ids))

    print_top5("TEST 1 — combined_search(image_weight=1.0)", pure_image)
    print_top5("BASELINE — image_similarity_search()", baseline_image)
    print(f"\n>>> Top-5 overlap with pure image search: {overlap_1}/5 "
          f"({'PASS' if overlap_1 >= 4 else 'CHECK THIS'})")

    # ── Test 2: pure text weight (0.0) ────────────────────────────────────────
    pure_text = combined_search(pil, TEST_MODIFIER, image_weight=0.0, min_score=0.0, n=10)
    baseline_text = semantic_search(TEST_MODIFIER, min_score=0.0, n=10)

    pure_text_ids = [r["id"] for r in pure_text[:5]]
    text_baseline_ids = [r["id"] for r in baseline_text[:5]]
    overlap_2 = len(set(pure_text_ids) & set(text_baseline_ids))

    print_top5("TEST 2 — combined_search(image_weight=0.0)", pure_text)
    print_top5("BASELINE — semantic_search(modifier only)", baseline_text)
    print(f"\n>>> Top-5 overlap with pure text search: {overlap_2}/5 "
          f"({'PASS' if overlap_2 >= 4 else 'CHECK THIS'})")

    # ── Test 3: mid-weight should differ from both extremes ──────────────────
    mid = combined_search(pil, TEST_MODIFIER, image_weight=0.5, min_score=0.0, n=10)
    mid_ids = set(r["id"] for r in mid[:5])

    diff_from_image = len(mid_ids - set(pure_image_ids))
    diff_from_text  = len(mid_ids - set(pure_text_ids))

    print_top5("TEST 3 — combined_search(image_weight=0.5)", mid)
    print(f"\n>>> Mid-weight differs from pure-image top-5 by: {diff_from_image}/5 entries")
    print(f">>> Mid-weight differs from pure-text top-5 by:  {diff_from_text}/5 entries")
    print(f">>> ({'PASS — blending is working' if (diff_from_image > 0 and diff_from_text > 0) else 'CHECK THIS — mid-weight identical to an extreme, blending may not be working'})")

    print("\n" + "=" * 60)
    print("If all three tests show PASS, combined_search is verified correct.")
    print("Safe to wire into the Multimodal Query tab in app.py.")


if __name__ == "__main__":
    main()
