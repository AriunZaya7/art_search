"""
evaluate.py — Art Search Evaluation
--------------------------------------
Measures retrieval quality using Precision@K, Recall@K, and MAP
(Mean Average Precision) against a hand-verified golden set.

SETUP:
  1. Run build_golden_set.py first to generate real GOLDEN_SET entries
     based on what your system actually returns (don't guess filenames).
  2. Paste those entries below, replacing the placeholder GOLDEN_SET.
  3. Run:  python evaluate.py

This is required by the assignment spec: "define at least 5 sample
queries and their expected results" — that's exactly what GOLDEN_SET is.
"""

from search import semantic_search

# ── Golden Set ────────────────────────────────────────────────────────────────
# Each entry: a query, and the filenames that SHOULD appear in the results.
# Generate these with build_golden_set.py — don't hand-write filenames,
# since you need to see real results first to know what's actually correct.

GOLDEN_SET = [
    {
        "query":    'woman by the window',
        "expected": ['impressionism/impressionism_008_Berthe_Morisot_-_Le_thé.jpg', 'impressionism/impressionism_079_Berthe_Morisot_-_Jeune_fille_au_manteau_vert_(Marthe).jpg', 'impressionism/impressionism_054_Caillebotte_-_Jeune_homme_à_la_fenêtre.jpg', 'expressionism/expressionism_052_Kirchner_-_Otto_and_Maschka_Mueller_in_the_Studio,_1911,_67568.jpg', 'expressionism/expressionism_024_Andreas_Reading_(79).jpg'],
        "k":        5,
    },
    {
        "query":    'portrait of a man with a beard',
        "expected": ["baroque/baroque_011_Diego_Velázquez_(1599-1660)_(school_of)_-_Head_of_a_Man,_The_Conde_De_Tilly,_Johan_'t_Serclaes_(1559–1632)_-_PC.67_-_Pollok_House.jpg", 'baroque/baroque_065_Diego_Velázquez_-_Gaspar_de_Guzmán,_Count-Duke_of_Olivares_-_A104_-_Hispanic_Society_of_America.jpg', 'baroque/baroque_034_Peter_Paul_Rubens_(1577-1640)_(after)_-_Paracelsus_(1493–1541)_(Theophrastus_Bombastus_von_Hohenheim)_-_LP_28_-_Bodleian_Libraries.jpg', 'baroque/baroque_064_Ritratti_di_Giovan_Francesco_Tomassoni.jpg', 'baroque/baroque_047_Diego_Velázquez_(1599-1660)_(copy_after)_-_The_Buffoon,_Pablo_de_Valladolid_-_TH.0206_-_Barrow-in-Furness_Town_Hall.jpg', 'baroque/baroque_052_Diego_Velázquez_(1599-1660)_(after)_-_Philip_IV_of_Spain_(1605–1665)_-_CP-TR_298_-_Cooper_Gallery.jpg'],
        "k":        5,
    },
    {
        "query":    'dark dramatic scene with candlelight',
        "expected": ['expressionism/expressionism_018_Blick_bei_Nacht_auf_die_Rue_des_Maronniers.png', 'baroque/baroque_014_The_Beheading_of_Saint_John_the_Baptist,_Caravaggio_(48256483017).jpg', 'romanticism/romanticism_066_CD_Friedrich_Klosterruine_Oybin.jpg'],
        "k":        5,
    },
    {
        "query":    'cool blue and green color palette',
        "expected": ['impressionism/impressionism_060_Impressionism_monet.jpg', 'abstract/abstract_009_Wassily_kandinsky,_kallmünz,_montagne_verde_chiaro,_1903,_04_firma.jpg', 'abstract/abstract_010_Wassily_kandinsky,_kallmünz,_montagne_verde_chiaro,_1903,_02.jpg', 'impressionism/impressionism_092_Degas_-_COIN_DE_VILLAGE,_circa_1895-98.jpg', 'expressionism/expressionism_067_Ernst_Ludwig_Kirchner_-_Garten_Graef_in_Jena.jpg'],
        "k":        5,
    },
    {
        "query":    'stormy seascape with turbulent waves',
        "expected": ['romanticism/romanticism_087_Theodore_Gericault_024_(27956584309).jpg', 'expressionism/expressionism_036_Beckmann_-_Tiedemann,_0409.jpg', 'surrealism/surrealism_004_Yves_tanguy,_paesaggio_con_nuvole_osa,_1928.jpg', 'surrealism/surrealism_068_Yves_tanguy,_senza_titolo_o_composizione_surrealista,_1927(berlino,_coll._pietzsch).jpg'],
        "k":        5,
    },
]

K_DEFAULT = 5
N_FETCH   = 20  # how many results to retrieve per query before scoring


# ── Metrics ───────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list, relevant: list, k: int) -> float:
    """Of the top-K retrieved, what fraction are relevant?"""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    hits  = sum(1 for f in relevant if f in top_k)
    return hits / k


def recall_at_k(retrieved: list, relevant: list, k: int) -> float:
    """Of all relevant items, what fraction appear in the top-K?"""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    hits  = sum(1 for f in relevant if f in top_k)
    return hits / len(relevant)


def average_precision(retrieved: list, relevant: list) -> float:
    """
    Average Precision — rewards ranking relevant items higher, not just
    including them somewhere in the list. Standard IR evaluation metric.
    """
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = 0
    precision_sum = 0.0
    for i, fname in enumerate(retrieved, 1):
        if fname in relevant_set:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(relevant) if hits > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Art Search Evaluation")
    print("=" * 60)

    if not GOLDEN_SET:
        print(
            "\n⚠  GOLDEN_SET is empty.\n"
            "   Run:  python build_golden_set.py\n"
            "   Then paste the generated entries into GOLDEN_SET above.\n"
        )
        return

    if len(GOLDEN_SET) < 5:
        print(
            f"\n⚠  Only {len(GOLDEN_SET)} queries defined — the assignment "
            f"requires at least 5.\n"
        )

    all_p, all_r, all_ap = [], [], []

    for entry in GOLDEN_SET:
        query    = entry["query"]
        expected = entry["expected"]
        k        = entry.get("k", K_DEFAULT)

        print(f"\nQuery : \"{query}\"  (K={k})")
        print(f"Expected ({len(expected)}): {expected}")

        results   = semantic_search(query, min_score=0.0, n=N_FETCH)
        retrieved = [r["id"] for r in results]

        print(f"Top {k} retrieved: {retrieved[:k]}")

        p  = precision_at_k(retrieved, expected, k)
        r  = recall_at_k(retrieved, expected, k)
        ap = average_precision(retrieved, expected)

        all_p.append(p)
        all_r.append(r)
        all_ap.append(ap)

        print(f"Precision@{k} : {p:.3f}")
        print(f"Recall@{k}    : {r:.3f}")
        print(f"Avg Precision : {ap:.3f}")

    print("\n" + "=" * 60)
    print("  Aggregate Metrics")
    print("=" * 60)
    print(f"Mean Precision@K : {sum(all_p)/len(all_p):.3f}")
    print(f"Mean Recall@K    : {sum(all_r)/len(all_r):.3f}")
    print(f"MAP (Mean Avg P) : {sum(all_ap)/len(all_ap):.3f}")
    print()
    print("MAP interpretation (rough guide, not a hard rule):")
    print("  > 0.5  — strong retrieval")
    print("  > 0.3  — acceptable")
    print("  < 0.3  — investigate query phrasing, threshold, or model fit")


if __name__ == "__main__":
    main()