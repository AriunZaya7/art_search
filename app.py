"""
app.py — Art Style Search Engine
Run with:  streamlit run app.py

Tabs:
  🔍 Search        — semantic text search across all artworks
  🖼 Browse Style  — browse all paintings by art style
  🎨 Similar Art   — upload an image, find visually similar artworks
  📊 Evaluate      — interactive golden set evaluation (Precision@K, Recall@K)
"""

import io
import zipfile
import sqlite3
import datetime
from pathlib import Path

import chromadb
import streamlit as st
from PIL import Image

from search import (
    semantic_search,
    style_search,
    image_similarity_search,
    combined_search,
)
from models import predict_style, embed_image_pil

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
GALLERY_DIR = BASE_DIR / "gallery"
CHROMA_DIR  = str(BASE_DIR / "chroma_db")
SQLITE_FILE = BASE_DIR / "groups.db"

COLS        = 4
MAX_RESULTS = 40
ALL_STYLES  = [
    "impressionism", "expressionism", "surrealism",
    "baroque", "abstract", "romanticism",
]

STYLE_COLORS = {
    "impressionism":  "#7bb3d4",
    "expressionism":  "#d47b7b",
    "surrealism":     "#a67bd4",
    "baroque":        "#d4a67b",
    "abstract":       "#7bd4a6",
    "romanticism":    "#d4c87b",
    "unknown":        "#888888",
}

EXAMPLE_QUERIES = [
    "woman reading by a window with soft light",
    "dark dramatic scene with candlelight chiaroscuro",
    "stormy seascape with turbulent waves",
    "dreamlike landscape with melting objects",
    "bold geometric shapes and primary colors",
    "figures dancing in dappled sunlight",
]

# ── CSS ───────────────────────────────────────────────────────────────────────

GALLERY_CSS = """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #0c0c0d !important;
    color: #f0ead6 !important;
}
[data-testid="stSidebar"] {
    background: #111114 !important;
    border-right: 1px solid #222228;
}
.block-container {
    padding-top: 1.2rem !important;
    max-width: 1400px;
}
h1 {
    font-family: Georgia, 'Times New Roman', serif !important;
    font-size: 1.8rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.04em !important;
    color: #f0ead6 !important;
}
h2, h3 {
    font-family: Georgia, serif !important;
    font-weight: 400 !important;
    color: #c9a84c !important;
    letter-spacing: 0.03em;
}
input[type="text"], textarea {
    background: #18181c !important;
    color: #f0ead6 !important;
    border: 1px solid #333340 !important;
    border-radius: 3px !important;
    font-size: 1rem !important;
}
input[type="text"]:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 1px #c9a84c33 !important;
}
.stButton > button {
    background: transparent !important;
    color: #c9a84c !important;
    border: 1px solid #c9a84c55 !important;
    border-radius: 3px !important;
    font-size: 11px !important;
    letter-spacing: 0.06em !important;
    padding: 3px 10px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #c9a84c !important;
    color: #0c0c0d !important;
    border-color: #c9a84c !important;
}
[data-testid="stFormSubmitButton"] > button {
    background: #c9a84c !important;
    color: #0c0c0d !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.06em !important;
}
.style-chip {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 2px;
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 600;
    margin-bottom: 4px;
}
.score-badge {
    font-size: 11px;
    color: #888;
    font-family: monospace;
}
.artwork-title {
    font-family: Georgia, serif;
    font-size: 12px;
    color: #c4b99a;
    margin-top: 4px;
    line-height: 1.4;
}
.sidebar-label {
    font-size: 10px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 4px;
    margin-top: 8px;
}
hr { border-color: #222228 !important; margin: 1.2rem 0 !important; }
.stAlert { background: #18181c !important; border-color: #333340 !important; color: #c4b99a !important; }
</style>
"""


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(
        name="artworks",
        metadata={"hnsw:space": "cosine"}
    )


@st.cache_resource
def load_db():
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS collections (
            collection_name TEXT NOT NULL,
            artwork_id      TEXT NOT NULL,
            path            TEXT,
            style           TEXT,
            title           TEXT,
            added_at        TEXT,
            PRIMARY KEY (collection_name, artwork_id)
        )
    """)
    conn.commit()
    return conn


# ── Collections ───────────────────────────────────────────────────────────────

def add_to_collection(name: str, artworks: list):
    conn = load_db()
    now  = datetime.datetime.now().isoformat()
    conn.executemany(
        "INSERT OR IGNORE INTO collections "
        "(collection_name, artwork_id, path, style, title, added_at) "
        "VALUES (?,?,?,?,?,?)",
        [(name, a["id"], a["path"], a["style"], a["title"], now) for a in artworks]
    )
    conn.commit()


def export_collection_zip(name: str) -> bytes:
    conn = load_db()
    rows = conn.execute(
        "SELECT path FROM collections WHERE collection_name=?", (name,)
    ).fetchall()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for (path,) in rows:
            p = Path(path)
            if p.exists():
                zf.write(p, p.name)
    return buf.getvalue()


# ── UI helpers ────────────────────────────────────────────────────────────────

def style_chip(style: str) -> str:
    color = STYLE_COLORS.get(style, "#888")
    return (
        f'<span class="style-chip" '
        f'style="background:{color}22;color:{color};border:1px solid {color}55;">'
        f'{style}</span>'
    )


def score_badge(score) -> str:
    if score is None:
        return ""
    return f'<span class="score-badge">{score:.3f}</span>'


def artwork_grid(results: list, selectable=False, key_prefix="art") -> list:
    selected = []
    for row_start in range(0, len(results), COLS):
        cols = st.columns(COLS)
        for col, item in zip(cols, results[row_start: row_start + COLS]):
            with col:
                path = Path(item.get("path", ""))
                if path.exists():
                    st.image(str(path), use_container_width=True)
                else:
                    st.markdown(
                        '<div style="height:160px;background:#18181c;'
                        'display:flex;align-items:center;justify-content:center;'
                        'color:#444;font-size:11px;">not found</div>',
                        unsafe_allow_html=True
                    )
                chip  = style_chip(item.get("style", "unknown"))
                badge = score_badge(item.get("score"))
                st.markdown(f'{chip} {badge}', unsafe_allow_html=True)
                title = item.get("title", item.get("filename", ""))[:45]
                st.markdown(
                    f'<div class="artwork-title">{title}</div>',
                    unsafe_allow_html=True
                )
                if selectable:
                    if st.checkbox(
                        "Select", key=f"{key_prefix}_{item['id']}",
                        label_visibility="collapsed"
                    ):
                        selected.append(item)
    return selected


def search_breadcrumb(trail: str):
    if not trail:
        return
    parts = trail.split(" → ")
    chips = " <span style='color:#444;'>→</span> ".join(
        f'<code style="background:#18181c;color:#c9a84c;'
        f'padding:2px 8px;border-radius:2px;font-size:11px;">{p}</code>'
        for p in parts
    )
    st.markdown(chips, unsafe_allow_html=True)


def save_to_collection_ui(selected: list, key_suffix=""):
    st.markdown(f"**{len(selected)} artwork(s) selected**")
    c1, c2 = st.columns([3, 1])
    with c1:
        name = st.text_input(
            "Collection name",
            placeholder="e.g.  Dutch Masters  ·  Blue Period  ·  Lecture Examples",
            label_visibility="collapsed",
            key=f"col_name_{key_suffix}",
        )
    with c2:
        if st.button("Save 📁", use_container_width=True, key=f"col_save_{key_suffix}"):
            if name.strip():
                add_to_collection(name.strip(), selected)
                st.success(f"Saved {len(selected)} artwork(s) to **{name.strip()}**")
            else:
                st.warning("Enter a collection name.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Art Search",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(GALLERY_CSS, unsafe_allow_html=True)

    # Session state defaults
    for key, val in [
        ("results",      []),
        ("trail",        ""),
        ("tab",          "🔍 Search"),
        ("query_input",  ""),
        ("eval_results", []),
        ("eval_query",   ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = val

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<h2 style="font-family:Georgia,serif;font-weight:400;font-size:1.1rem;'
            'letter-spacing:0.08em;color:#c9a84c;text-transform:uppercase;">'
            '🎨 Art Search</h2>',
            unsafe_allow_html=True
        )
        st.session_state.tab = st.radio(
            "nav",
            ["🔍 Search", "🖼 Browse Style", "🎨 Similar Art", "🧬 Multimodal Query", "📊 Evaluate"],
            label_visibility="collapsed",
        )
        st.divider()

        st.markdown('<div class="sidebar-label">Similarity threshold</div>',
                    unsafe_allow_html=True)
        min_score = st.slider("min_score", 0.0, 1.0, 0.0, 0.05,
                              label_visibility="collapsed")

        st.markdown('<div class="sidebar-label">Max results</div>',
                    unsafe_allow_html=True)
        n_results = st.slider("n_results", 4, MAX_RESULTS, 20, 4,
                              label_visibility="collapsed")

        st.markdown('<div class="sidebar-label">Filter by style</div>',
                    unsafe_allow_html=True)
        style_filter = st.selectbox(
            "style_filter",
            ["All styles"] + ALL_STYLES,
            label_visibility="collapsed",
        )
        style_filter = None if style_filter == "All styles" else style_filter

        st.divider()
        try:
            st.caption(f"{load_chroma().count()} artworks indexed")
        except Exception:
            st.caption("No index — run indexer.py first")

    tab = st.session_state.tab

    # ════════════════════════════════════════════════════════════════
    # TAB: SEARCH
    # ════════════════════════════════════════════════════════════════
    if tab == "🔍 Search":
        st.markdown("# Art Search")
        st.markdown(
            '<div style="color:#888;font-size:13px;margin-bottom:1rem;">'
            'Describe what you are looking for — style, mood, color, subject, or composition.'
            '</div>',
            unsafe_allow_html=True
        )

        # ── Clickable example queries ─────────────────────────────────────────
        st.markdown(
            '<div style="font-size:11px;color:#555;margin-bottom:6px;'
            'letter-spacing:0.06em;text-transform:uppercase;">Try a query</div>',
            unsafe_allow_html=True
        )
        ex_cols = st.columns(3)
        for col, ex in zip(ex_cols, EXAMPLE_QUERIES[:3]):
            with col:
                if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                    st.session_state.query_input = ex
                    st.rerun()

        ex_cols2 = st.columns(3)
        for col, ex in zip(ex_cols2, EXAMPLE_QUERIES[3:]):
            with col:
                if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                    st.session_state.query_input = ex
                    st.rerun()

        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

        # ── Search form ───────────────────────────────────────────────────────
        with st.form("search_form"):
            c1, c2 = st.columns([5, 1])
            with c1:
                query = st.text_input(
                    "query",
                    value=st.session_state.query_input,
                    placeholder="e.g.  impressionist landscape with warm golden light …",
                    label_visibility="collapsed",
                )
            with c2:
                go = st.form_submit_button("Search", use_container_width=True)

        if go and query.strip():
            st.session_state.query_input = query.strip()
            with st.spinner("Searching the collection…"):
                st.session_state.results = semantic_search(
                    query.strip(),
                    min_score=min_score,
                    n=n_results,
                    style_filter=style_filter,
                )
            st.session_state.trail = f'"{query.strip()}"'

        # ── Results ───────────────────────────────────────────────────────────
        if st.session_state.results:
            ca, cb = st.columns([5, 1])
            with ca:
                st.markdown(
                    f'<div style="color:#888;font-size:12px;margin-bottom:8px;">'
                    f'{len(st.session_state.results)} results</div>',
                    unsafe_allow_html=True
                )
                search_breadcrumb(st.session_state.trail)
            with cb:
                if st.button("Clear", use_container_width=True):
                    st.session_state.results     = []
                    st.session_state.trail       = ""
                    st.session_state.query_input = ""
                    st.rerun()

            scores = [r["score"] for r in st.session_state.results
                      if r["score"] is not None]
            if scores:
                with st.expander("Similarity distribution", expanded=False):
                    import pandas as pd
                    st.bar_chart(
                        pd.DataFrame({"similarity": scores}),
                        height=100,
                        color="#c9a84c",
                    )

            selected = artwork_grid(
                st.session_state.results, selectable=True, key_prefix="search"
            )

            if selected:
                st.divider()
                save_to_collection_ui(selected, key_suffix="search")

            # ── Refine ────────────────────────────────────────────────────────
            st.divider()
            st.markdown(
                '<div style="color:#666;font-size:11px;letter-spacing:0.08em;'
                'text-transform:uppercase;margin-bottom:8px;">Refine results</div>',
                unsafe_allow_html=True
            )
            with st.form("refine_form"):
                r1, r2 = st.columns([5, 1])
                with r1:
                    rquery = st.text_input(
                        "refine",
                        placeholder="narrow further …",
                        label_visibility="collapsed"
                    )
                with r2:
                    refine = st.form_submit_button("Refine", use_container_width=True)

            if refine and rquery.strip():
                ids = [r["id"] for r in st.session_state.results]
                with st.spinner("Refining…"):
                    st.session_state.results = semantic_search(
                        rquery.strip(),
                        min_score=min_score,
                        n=n_results,
                        limit_to=ids,
                        style_filter=style_filter,
                    )
                st.session_state.trail += f' → "{rquery.strip()}"'
                st.rerun()

        elif st.session_state.trail:
            st.info("No results. Try lowering the similarity threshold or rephrasing.")

    # ════════════════════════════════════════════════════════════════
    # TAB: BROWSE BY STYLE
    # ════════════════════════════════════════════════════════════════
    elif tab == "🖼 Browse Style":
        st.markdown("# Browse by Style")

        selected_style = st.selectbox(
            "Choose a style",
            ALL_STYLES,
            format_func=lambda s: s.capitalize(),
        )

        style_descriptions = {
            "impressionism":  "Loose brushwork, everyday subjects, the transient effects of light and color.",
            "expressionism":  "Distorted forms and vivid colors to convey subjective emotion over objective reality.",
            "surrealism":     "Dream imagery, unexpected juxtapositions, and the subconscious mind made visible.",
            "baroque":        "Dramatic light and shadow, emotional intensity, and grandeur of scale.",
            "abstract":       "Non-representational form — color, shape, and line freed from depicting the real world.",
            "romanticism":    "Sublime landscapes, heroic narratives, and intense emotional atmosphere.",
        }
        st.markdown(
            f'<div style="color:#888;font-size:13px;font-style:italic;margin-bottom:1rem;">'
            f'{style_descriptions.get(selected_style, "")}</div>',
            unsafe_allow_html=True
        )

        with st.spinner(f"Loading {selected_style} paintings…"):
            results = style_search(selected_style, n=n_results)

        if results:
            st.markdown(
                f'<div style="color:#555;font-size:11px;margin-bottom:12px;">'
                f'{len(results)} paintings</div>',
                unsafe_allow_html=True
            )
            selected = artwork_grid(results, selectable=True, key_prefix="browse")
            if selected:
                st.divider()
                save_to_collection_ui(selected, key_suffix="browse")
        else:
            st.info(f"No {selected_style} paintings indexed yet.")

    # ════════════════════════════════════════════════════════════════
    # TAB: SIMILAR ART
    # ════════════════════════════════════════════════════════════════
    elif tab == "🎨 Similar Art":
        st.markdown("# Find Similar Artworks")
        st.markdown(
            '<div style="color:#888;font-size:13px;margin-bottom:1.2rem;">'
            'Upload any image — a painting, a photo, or a sketch — '
            'and find artworks with a similar visual mood, palette, or composition.'
            '</div>',
            unsafe_allow_html=True
        )

        uploaded = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
        )

        if uploaded:
            uploaded.seek(0)
            pil = Image.open(uploaded).convert("RGB")

            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(pil, caption="Query image", use_container_width=True)

                with st.spinner("Analysing style…"):
                    embedding   = embed_image_pil(pil)
                    style_preds = predict_style(embedding, top_n=3)

                st.markdown(
                    '<div style="font-size:11px;color:#888;text-transform:uppercase;'
                    'letter-spacing:0.08em;margin-top:8px;">Detected style</div>',
                    unsafe_allow_html=True
                )
                for style_name, conf in style_preds:
                    bar_width = int(conf * 100)
                    color     = STYLE_COLORS.get(style_name, "#888")
                    st.markdown(
                        f'<div style="margin:3px 0;">'
                        f'<span style="font-size:11px;color:{color};width:90px;'
                        f'display:inline-block;">{style_name}</span>'
                        f'<span style="background:{color}33;display:inline-block;'
                        f'height:8px;width:{bar_width}px;border-radius:2px;'
                        f'vertical-align:middle;"></span>'
                        f'<span style="font-size:10px;color:#555;margin-left:6px;">'
                        f'{conf:.2f}</span></div>',
                        unsafe_allow_html=True
                    )

            with c2:
                go_sim = st.button("Find similar artworks 🔍", type="primary")

            if go_sim:
                with st.spinner("Searching the collection…"):
                    sim_results = image_similarity_search(
                        pil,
                        min_score=min_score,
                        n=n_results,
                        style_filter=style_filter,
                    )

                if sim_results:
                    st.markdown(
                        f'<div style="color:#888;font-size:12px;margin-bottom:8px;">'
                        f'{len(sim_results)} similar artworks found</div>',
                        unsafe_allow_html=True
                    )
                    scores = [r["score"] for r in sim_results]
                    with st.expander("Similarity distribution", expanded=False):
                        import pandas as pd
                        st.bar_chart(
                            pd.DataFrame({"similarity": scores}),
                            height=100, color="#c9a84c",
                        )
                    selected = artwork_grid(
                        sim_results, selectable=True, key_prefix="similar"
                    )
                    if selected:
                        st.divider()
                        save_to_collection_ui(selected, key_suffix="similar")
                else:
                    st.info("No similar artworks found. Try lowering the threshold.")

    # ════════════════════════════════════════════════════════════════
    # TAB: MULTIMODAL QUERY (image + text combined)
    # ════════════════════════════════════════════════════════════════
    elif tab == "🧬 Multimodal Query":
        st.markdown("# Multimodal Query")
        st.markdown(
            '<div style="color:#888;font-size:13px;margin-bottom:1.2rem;">'
            'Upload a painting and describe a change — a different mood, '
            'palette, or style. The system searches using both the image '
            'and your description combined into a single query.'
            '</div>',
            unsafe_allow_html=True
        )

        uploaded = st.file_uploader(
            "Upload a reference image",
            type=["jpg", "jpeg", "png", "webp"],
            key="combo_uploader",
        )

        if uploaded:
            uploaded.seek(0)
            pil = Image.open(uploaded).convert("RGB")

            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(pil, caption="Reference image", use_container_width=True)

            with c2:
                modifier = st.text_input(
                    "Describe the change you want",
                    placeholder='e.g.  "but with warmer autumn colors"  ·  '
                                '"similar composition but baroque style"  ·  '
                                '"more dramatic lighting"',
                )

                st.markdown(
                    '<div class="sidebar-label" style="margin-top:12px;">'
                    'Image ↔ Text balance</div>',
                    unsafe_allow_html=True
                )
                image_weight = st.slider(
                    "balance",
                    0.0, 1.0, 0.7, 0.05,
                    label_visibility="collapsed",
                    help="1.0 = pure visual match, 0.0 = pure text match",
                )
                bal_col1, bal_col2 = st.columns(2)
                with bal_col1:
                    st.markdown(
                        f'<div style="font-size:11px;color:#888;">'
                        f'🖼 Image: {image_weight:.0%}</div>',
                        unsafe_allow_html=True
                    )
                with bal_col2:
                    st.markdown(
                        f'<div style="font-size:11px;color:#888;text-align:right;">'
                        f'📝 Text: {(1-image_weight):.0%}</div>',
                        unsafe_allow_html=True
                    )

                go_combo = st.button(
                    "Search with combined query 🔍", type="primary",
                    disabled=not modifier.strip()
                )

            if go_combo and modifier.strip():
                with st.spinner("Combining image and text embeddings…"):
                    combo_results = combined_search(
                        pil,
                        modifier.strip(),
                        image_weight=image_weight,
                        min_score=min_score,
                        n=n_results,
                        style_filter=style_filter,
                    )

                if combo_results:
                    st.divider()
                    st.markdown(
                        f'<div style="color:#888;font-size:12px;margin-bottom:4px;">'
                        f'{len(combo_results)} results for reference image '
                        f'+ <em>"{modifier.strip()}"</em></div>'
                        f'<div style="color:#555;font-size:11px;margin-bottom:12px;">'
                        f'Weighting: {image_weight:.0%} image / {(1-image_weight):.0%} text'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    scores = [r["score"] for r in combo_results]
                    with st.expander("Similarity distribution", expanded=False):
                        import pandas as pd
                        st.bar_chart(
                            pd.DataFrame({"similarity": scores}),
                            height=100, color="#c9a84c",
                        )

                    selected = artwork_grid(
                        combo_results, selectable=True, key_prefix="combo"
                    )
                    if selected:
                        st.divider()
                        save_to_collection_ui(selected, key_suffix="combo")
                else:
                    st.info("No results. Try lowering the similarity threshold "
                            "or adjusting the image/text balance.")
        else:
            st.markdown(
                '<div style="color:#555;font-size:12px;margin-top:1rem;">'
                'Upload an image above to get started. Try uploading a baroque '
                'painting and typing "but impressionist style" — watch how moving '
                'the slider toward text shifts results away from the original '
                'composition and toward the described style.'
                '</div>',
                unsafe_allow_html=True
            )

    # ════════════════════════════════════════════════════════════════
    # TAB: EVALUATE
    # ════════════════════════════════════════════════════════════════
    elif tab == "📊 Evaluate":
        st.markdown("# Evaluation — Golden Set")
        st.markdown(
            '<div style="color:#888;font-size:13px;margin-bottom:1.2rem;">'
            'Run a query, mark which results are correct, '
            'and get Precision@K and Recall@K metrics for your report.'
            '</div>',
            unsafe_allow_html=True
        )

        with st.form("eval_form"):
            ec1, ec2 = st.columns([5, 1])
            with ec1:
                equery = st.text_input(
                    "eval_query",
                    placeholder="e.g.  dark scene with dramatic lighting …",
                    label_visibility="collapsed",
                )
            with ec2:
                erun = st.form_submit_button("Run ▶", use_container_width=True)

        if erun and equery.strip():
            with st.spinner("Searching…"):
                st.session_state.eval_results = semantic_search(
                    equery.strip(),
                    min_score=0.0,
                    n=20,
                    style_filter=style_filter,
                )
            st.session_state.eval_query = equery.strip()

        if st.session_state.eval_results:
            st.markdown(
                f'<div style="color:#888;font-size:12px;margin-bottom:8px;">'
                f'Results for <em>"{st.session_state.eval_query}"</em> — '
                f'check the boxes on artworks that are relevant:</div>',
                unsafe_allow_html=True
            )

            relevant = artwork_grid(
                st.session_state.eval_results,
                selectable=True,
                key_prefix="eval"
            )

            k = st.slider(
                "K (evaluate top-K results)",
                1,
                min(20, len(st.session_state.eval_results)),
                5,
            )

            if relevant:
                retrieved = [r["id"] for r in st.session_state.eval_results]
                top_k     = set(retrieved[:k])
                rel_ids   = set(r["id"] for r in relevant)
                hits      = len(top_k & rel_ids)

                precision = hits / k
                recall    = hits / len(rel_ids) if rel_ids else 0

                m1, m2, m3 = st.columns(3)
                m1.metric(f"Precision@{k}", f"{precision:.2f}",
                          help="Of top-K results, fraction that are relevant")
                m2.metric(f"Recall@{k}",    f"{recall:.2f}",
                          help="Of all marked relevant, fraction in top-K")
                m3.metric("Relevant marked", len(relevant))

                if precision >= 0.6:
                    st.success("Good retrieval accuracy for this query.")
                elif precision >= 0.3:
                    st.warning("Moderate accuracy — try rephrasing or lowering threshold.")
                else:
                    st.error("Low accuracy — try a more descriptive query.")

                with st.expander("Export for report"):
                    rel_files = [r["filename"] for r in relevant]
                    st.code(
                        f'# Query: "{st.session_state.eval_query}"\n'
                        f'# Precision@{k}: {precision:.2f}\n'
                        f'# Recall@{k}: {recall:.2f}\n'
                        f'expected = {rel_files}',
                        language="python"
                    )
            else:
                st.info("Check the boxes above to mark relevant artworks.")


if __name__ == "__main__":
    main()