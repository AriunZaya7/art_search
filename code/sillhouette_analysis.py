"""
silhouette_analysis.py
Computes per-style silhouette scores on the REAL 1152-dim SigLIP 2 embeddings
(not the 2D UMAP projection — this avoids the rescaling/zoom confusion from
comparing isolated vs combined scatter plots).

Produces:
  1. Console printout of overall + per-style scores
  2. A horizontal bar chart saved as silhouette_scores.png

Requirements:
    pip install scikit-learn matplotlib numpy chromadb

Usage:
    python silhouette_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
import chromadb
from sklearn.metrics import silhouette_score, silhouette_samples
from collections import defaultdict

CHROMA_DIR = "chroma_db"

# ── Style colors matching your app's theme ────────────────────────────────────
STYLE_COLORS = {
    "impressionism":  "#7BA3C4",
    "expressionism":  "#B5562F",
    "surrealism":     "#9B7BB5",
    "baroque":        "#C49A3C",
    "abstract":       "#6B7F5E",
    "romanticism":    "#C2A572",
}


def main():
    print("Loading embeddings from ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection("artworks")
    data = col.get(include=["embeddings", "metadatas"])

    embeddings = np.array(data["embeddings"])
    styles = [m.get("style", "unknown") for m in data["metadatas"]]

    print(f"Loaded {len(embeddings)} embeddings across {len(set(styles))} styles\n")

    # ── Overall silhouette score ────────────────────────────────────────────────
    overall_score = silhouette_score(embeddings, styles, metric="cosine")
    print(f"Overall silhouette score: {overall_score:.3f}")
    print("(-1 = overlapping, 0 = ambiguous, +1 = perfectly separated)\n")

    # ── Per-style average ────────────────────────────────────────────────────────
    per_sample = silhouette_samples(embeddings, styles, metric="cosine")
    by_style = defaultdict(list)
    for score, style in zip(per_sample, styles):
        by_style[style].append(score)

    style_avgs = {style: np.mean(scores) for style, scores in by_style.items()}
    sorted_styles = sorted(style_avgs.items(), key=lambda x: -x[1])

    print("Per-style average silhouette score:")
    for style, avg in sorted_styles:
        print(f"  {style:<15} {avg:.3f}  (n={len(by_style[style])})")

    # ── Plot ──────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("#F4EEE2")
    ax.set_facecolor("#FFFCF6")

    labels = [s[0] for s in sorted_styles]
    values = [s[1] for s in sorted_styles]
    colors = [STYLE_COLORS.get(l, "#888888") for l in labels]

    bars = ax.barh(labels, values, color=colors, edgecolor="#3A2E22", linewidth=0.6, height=0.6)

    # Compute a margin proportional to the data range so labels never collide
    # with the zero-axis line, regardless of how small or negative a bar is.
    data_range = max(values) - min(values)
    margin = max(data_range * 0.03, 0.008)

    for bar, val in zip(bars, values):
        if val >= 0:
            x_pos, ha = val + margin, "left"
        else:
            x_pos, ha = val - margin, "right"
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
                va="center", ha=ha, fontsize=11, color="#3A2E22", fontweight="bold")

    # Give the plot extra horizontal room so edge labels never clip
    pad = data_range * 0.18
    ax.set_xlim(min(values) - pad, max(values) + pad)

    ax.axvline(0, color="#3A2E22", linewidth=0.8)
    ax.axvline(overall_score, color="#8C3F1F", linewidth=1.2, linestyle="--",
               label=f"Overall avg: {overall_score:.3f}")

    ax.set_xlabel("Silhouette score (cosine distance)", fontsize=11, color="#3A2E22")
    ax.set_title("How well-separated is each art style in the embedding space?",
                 fontsize=13, color="#3A2E22", pad=15, fontfamily="serif")

    ax.tick_params(colors="#3A2E22", labelsize=11)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#3A2E22")

    ax.legend(loc="upper right", fontsize=10, frameon=False, labelcolor="#3A2E22")

    plt.tight_layout()
    plt.savefig("silhouette_scores.png", dpi=150, facecolor="#F4EEE2")
    print("\nSaved silhouette_scores.png")


if __name__ == "__main__":
    main()