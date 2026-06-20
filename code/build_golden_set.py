"""
build_golden_set.py — Interactive Golden Set Builder
------------------------------------------------------
Run a query, see the top results with filenames, and mark which ones are
actually relevant. Generates a ready-to-paste GOLDEN_SET entry for evaluate.py.

This solves the chicken-and-egg problem: you can't write expected results
in evaluate.py until you know what your system actually returns and which
of those results are correct.

Usage:
    python build_golden_set.py
"""

from search import semantic_search

N_RESULTS = 10


def show_results(query: str, results: list):
    print(f"\nQuery: \"{query}\"")
    print("-" * 70)
    if not results:
        print("  (no results)")
        return
    for i, r in enumerate(results, 1):
        title = r.get("title") or r.get("filename", "")
        print(f"  {i:2}. [{r['score']:.3f}] [{r['style']:<14}] {r['filename']}")
        if title and title != r.get("filename"):
            print(f"       title: {title}")


def main():
    print("=" * 70)
    print("  Golden Set Builder")
    print("=" * 70)
    print(
        "\nRun a query, look at the results, then type the numbers of the\n"
        "ones that are ACTUALLY relevant (e.g. '1,3,5' or '1 3 5').\n"
        "Type 'skip' to discard this query, or 'quit' to finish.\n"
    )

    golden_entries = []

    while True:
        query = input("\nEnter a query (or 'done' to finish): ").strip()
        if query.lower() == "done":
            break
        if not query:
            continue

        results = semantic_search(query, min_score=0.0, n=N_RESULTS)
        show_results(query, results)

        if not results:
            continue

        mark = input(
            f"\nWhich result numbers (1-{len(results)}) are relevant? "
            f"(comma/space separated, 'skip', or 'all'): "
        ).strip()

        if mark.lower() == "skip":
            continue

        if mark.lower() == "all":
            relevant_indices = list(range(1, len(results) + 1))
        else:
            try:
                relevant_indices = [int(x) for x in mark.replace(",", " ").split()]
            except ValueError:
                print("  Couldn't parse that, skipping this query.")
                continue

        relevant_filenames = [
            results[i - 1]["filename"]
            for i in relevant_indices
            if 1 <= i <= len(results)
        ]

        if not relevant_filenames:
            print("  No valid relevant results marked, skipping.")
            continue

        golden_entries.append({
            "query": query,
            "expected": relevant_filenames,
            "k": 5,
        })
        print(f"  Saved: {len(relevant_filenames)} relevant result(s) for this query.")

    if not golden_entries:
        print("\nNo golden set entries created.")
        return

    # ── Print ready-to-paste Python code ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Paste this into the GOLDEN_SET list in evaluate.py:")
    print("=" * 70 + "\n")

    print("GOLDEN_SET = [")
    for entry in golden_entries:
        print("    {")
        print(f'        "query":    {entry["query"]!r},')
        print(f'        "expected": {entry["expected"]!r},')
        print(f'        "k":        {entry["k"]},')
        print("    },")
    print("]")

    print(f"\n{len(golden_entries)} queries captured. ", end="")
    if len(golden_entries) < 5:
        print(f"Assignment requires at least 5 — run again to add {5 - len(golden_entries)} more.")
    else:
        print("Meets the assignment's 5-query minimum.")


if __name__ == "__main__":
    main()