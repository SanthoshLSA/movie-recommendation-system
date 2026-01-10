import streamlit as st
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from streamlit_clickable_images import clickable_images


# ----------------------------
# MUST be first Streamlit command
# ----------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

OMDB_API_KEY = "f5127ade"


# ----------------------------
# CSS (your theme + drawer styles)
# ----------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400..900&family=Poppins&display=swap');
    * { font-family: 'Orbitron', 'sans-serif'; }

    .stApp { background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%); }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid rgba(59, 130, 246, 0.2);
    }
    [data-testid="stSidebar"] > div:first-child { background: transparent; }
    [data-testid="stSidebarCollapseButton"] { position: absolute; left: -9999px; }
    button[kind="header"] { display: none; }

    .main .block-container { padding: 2rem 3rem; max-width: 100%; }

    h1 {
        color: #ffffff; font-weight: 700; font-size: 2.5rem;
        margin-bottom: 1rem; letter-spacing: -0.5px;
        font-family: 'Orbitron'; !important
    }
    h2, h3 { color: #e0e0e0; font-weight: 600; }

    .sidebar-header {
        color: #ffffff; font-size: 1.5rem; font-weight: 700;
        margin-bottom: 2rem; padding-bottom: 1rem;
        border-bottom: 3px solid #3b82f6;
    }

    .stSelectbox label, .stSlider label {
        color: #60a5fa !important; font-weight: 600; font-size: 0.875rem;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .stSelectbox input { color: #ffffff !important; }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: #ffffff; font-weight: 600; border: none; border-radius: 12px;
        padding: 0.875rem 2.5rem; font-size: 1rem; letter-spacing: 0.3px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
        width: 100%; margin-top: 1.5rem;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
    }

    .movie-title {
        color: #ffffff; font-weight: 600; font-size: 0.95rem;
        margin: 0.6rem 0 0.2rem 0; line-height: 1.3;
    }
    .similarity-score {
        color: #3b82f6; font-weight: 500; font-size: 0.85rem;
        margin: 0 0 0.8rem 0;
    }

    div[data-testid="column"] > div > div {
        min-height: 450px;
    }

    .results-header {
        color: #ffffff; font-size: 1.5rem; font-weight: 700;
        margin-bottom: 1.25rem; padding: 1.25rem 1.5rem;
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
    }

    /* Drawer */
    .drawer-card {
        position: fixed;
        top: 2rem;
        right: 3rem;
        width: calc((100% - 6rem) / 4 - 1.5rem);
        max-height: calc(100vh - 4rem);
        overflow-y: auto;
        background: rgba(22, 33, 62, 0.55);
        border: 1px solid rgba(59, 130, 246, 0.18);
        border-radius: 14px;
        padding: 1.25rem 1.25rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        backdrop-filter: blur(6px);
        z-index: 100;
    }
    .drawer-title {
        color: #ffffff; font-weight: 800; font-size: 1.1rem;
        line-height: 1.35; margin: 0 0 0.5rem 0;
    }
    .drawer-sub {
        color: #60a5fa; font-weight: 600; font-size: 0.9rem;
        margin: 0 0 1rem 0;
    }
    .drawer-overview {
        color: #e0e0e0; line-height: 1.7; font-size: 0.95rem; margin: 0;
    }
    .drawer-kv {
        margin-top: 1rem;
        border-top: 1px solid rgba(59, 130, 246, 0.18);
        padding-top: 1rem;
        display: grid;
        grid-template-columns: 1fr;
        gap: 0.5rem;
    }
    .kv-row {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        font-size: 0.9rem;
    }
    .kv-key { color: #94a3b8; }
    .kv-val { color: #e5e7eb; text-align: right; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(ttl=3600)
def fetch_poster(title: str) -> str:
    params = {"apikey": OMDB_API_KEY, "t": title[:40], "r": "json"}
    try:
        response = requests.get("https://www.omdbapi.com/", params=params, timeout=10)
        data = response.json()
        if data.get("Response") == "True" and data.get("Poster") and data["Poster"] != "N/A":
            poster_url = data["Poster"]
            if poster_url.startswith("http://"):
                poster_url = poster_url.replace("http://", "https://")
            return poster_url
    except Exception:
        pass
    return "https://via.placeholder.com/300x450/1a1a2e/60a5fa?text=No+Poster"


def parse_genres_cell(val) -> list[str]:
    """
    Return a list of genre names from many common formats:
    - list: ["Action", "Drama"]
    - pipe string: "Action|Drama"
    - comma string: "Action, Drama"
    - stringified list: "['Action','Drama']"
    - stringified list of dicts: "[{'id': 28, 'name': 'Action'}, ...]"
    """
    if val is None:
        return []
    if isinstance(val, float) and np.isnan(val):
        return []

    if isinstance(val, list):
        out = []
        for item in val:
            if isinstance(item, dict) and "name" in item:
                out.append(str(item["name"]).strip())
            else:
                out.append(str(item).strip())
        return [g for g in out if g]

    s = str(val).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []

    if "|" in s:
        return [g.strip() for g in s.split("|") if g.strip()]

    if "," in s:
        return [g.strip() for g in s.split(",") if g.strip()]

    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                out = []
                for item in parsed:
                    if isinstance(item, dict) and "name" in item:
                        out.append(str(item["name"]).strip())
                    else:
                        out.append(str(item).strip())
                return [g for g in out if g]
        except (ValueError, SyntaxError):
            return []

    return [s]


@st.cache_data(show_spinner=False)
def load_and_process_data():
    processed_path = Path("data/processed/movies_processed.pkl")
    if not processed_path.exists():
        st.error("Run `python src/preprocess.py` first!")
        st.stop()

    df = joblib.load(processed_path)

    # Ensure required columns
    if "title" not in df.columns:
        st.error("Processed file must include a 'title' column.")
        st.stop()
    if "overview" not in df.columns:
        df["overview"] = ""
    if "combined_text" not in df.columns:
        st.error("Processed file must include a 'combined_text' column used for TF-IDF.")
        st.stop()

    # Detect which column contains genres
    candidate_genre_cols = ["genres", "genre", "Genre", "genre_names", "genres_list"]
    genre_col = next((c for c in candidate_genre_cols if c in df.columns), None)

    # Build unique genre list (if we found a column)
    unique_genres = set()
    if genre_col is not None:
        for v in df[genre_col].dropna():
            for g in parse_genres_cell(v):
                if g and len(g) > 1:
                    unique_genres.add(g)

    titles = df["title"].astype(str).tolist()
    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(df["combined_text"].fillna(""))
    similarity = cosine_similarity(tfidf)

    title_to_idx = {t: i for i, t in enumerate(titles)}

    return {
        "titles": titles,
        "title_to_idx": title_to_idx,
        "similarity": similarity,
        "df": df,
        "genre_col": genre_col,
        "unique_genres": sorted(unique_genres),
    }


def get_recommendations(model_data, title: str, top_n: int = 10):
    titles = model_data["titles"]
    title_to_idx = model_data["title_to_idx"]
    similarity = model_data["similarity"]

    if title not in title_to_idx:
        return []

    idx = title_to_idx[title]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    return [(titles[i], float(score)) for i, score in scores]


def get_movies_by_genre(model_data, genre: str, limit: int = 20):
    df = model_data["df"]
    genre_col = model_data.get("genre_col")

    # If no genre col exists, just show first `limit`
    if not genre_col:
        titles = df["title"].astype(str).tolist()[:limit]
        return [(t, 1.0) for t in titles]

    if genre == "All":
        titles = df["title"].astype(str).tolist()[:limit]
        return [(t, 1.0) for t in titles]

    mask = df[genre_col].apply(lambda v: genre in parse_genres_cell(v))
    filtered = df[mask]
    titles = filtered["title"].astype(str).tolist()[:limit]
    return [(t, 1.0) for t in titles]


def get_row(model_data, title: str):
    df = model_data["df"]
    match = df[df["title"] == title]
    if match.empty:
        return None
    return match.iloc[0]


def fmt_value(v, field_name=None):
    """Format values for display with appropriate formatting."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    
    # Format budget and revenue
    if field_name in ['budget', 'revenue']:
        try:
            num = float(v)
            if num == 0:
                return None
            if num >= 1_000_000_000:
                return f"${num/1_000_000_000:.1f}B"
            elif num >= 1_000_000:
                return f"${num/1_000_000:.0f}M"
            else:
                return f"${num:,.0f}"
        except (ValueError, TypeError):
            return None
    
    # Format runtime
    if field_name == 'runtime':
        try:
            minutes = int(float(v))
            if minutes == 0:
                return None
            hours = minutes // 60
            mins = minutes % 60
            if hours > 0:
                return f"{hours}h {mins}m"
            return f"{mins}m"
        except (ValueError, TypeError):
            return None
    
    # Format rating
    if field_name == 'vote_average':
        try:
            rating = float(v)
            if rating == 0:
                return None
            return f"{rating:.1f}/10"
        except (ValueError, TypeError):
            return None
    
    # Format vote count
    if field_name == 'vote_count':
        try:
            count = int(float(v))
            if count == 0:
                return None
            if count >= 1000:
                return f"{count:,} votes"
            return f"{count} votes"
        except (ValueError, TypeError):
            return None
    
    # Format popularity
    if field_name == 'popularity':
        try:
            pop = float(v)
            if pop == 0:
                return None
            return f"{pop:.0f}"
        except (ValueError, TypeError):
            return None
    
    # Format genres (pipe-separated to comma-separated)
    if field_name in ['genres', 'genre']:
        s = str(v).strip()
        if '|' in s:
            return s.replace('|', ', ')
        return s if s else None
    
    # Default formatting
    if isinstance(v, (list, tuple, set)):
        v = ", ".join(map(str, v))
    s = str(v).strip()
    return s if s else None


# ----------------------------
# App
# ----------------------------
def main():
    # State
    if "recs" not in st.session_state:
        st.session_state.recs = []
    if "last_query" not in st.session_state:
        st.session_state.last_query = None  # (label, n)
    if "picked_title" not in st.session_state:
        st.session_state.picked_title = None
    if "selected_genre" not in st.session_state:
        st.session_state.selected_genre = "All"

    with st.spinner("Loading model..."):
        model_data = load_and_process_data()

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Configuration</div>', unsafe_allow_html=True)

        # Genre dropdown (instant load)
        genre_options = ["All"]
        if model_data.get("unique_genres"):
            genre_options += model_data["unique_genres"]
        else:
            st.warning("No genre column found in your processed file (genres filter disabled).")

        def on_genre_change():
            g = st.session_state.selected_genre
            st.session_state.recs = get_movies_by_genre(model_data, g, limit=20)
            st.session_state.last_query = (f"Genre: {g}", 20)
            st.session_state.picked_title = None

        st.selectbox(
            "Select genre",
            options=genre_options,
            key="selected_genre",
            on_change=on_genre_change,
        )

        # Similarity mode (kept)
        movie_list = sorted(model_data["titles"])
        selected_movie = st.selectbox("Select a movie", movie_list, index=0)

        top_n = st.slider("Number of recommendations", 5, 25, 5, 1)

        if st.button("Find Similar Movies", type="primary"):
            with st.spinner("Analyzing similarities..."):
                st.session_state.recs = get_recommendations(model_data, selected_movie, top_n)
                st.session_state.last_query = (selected_movie, top_n)
                st.session_state.picked_title = None

    # Header
    st.markdown('<h1 style="font-family:Orbitron;">Movie Recommendation System</h1>', unsafe_allow_html=True)

    # If nothing loaded yet, prompt user
    if not st.session_state.recs:
        st.info('Select a genre to load movies, or pick a movie and click "Find Similar Movies".')
        return

    label, _n = st.session_state.last_query or ("Results", len(st.session_state.recs))

    # Layout: left grid + right drawer
    left, right = st.columns([3, 1], gap="large")

    with left:
        st.markdown(
            f'<div class="results-header">Showing movies for {label}</div>',
            unsafe_allow_html=True,
        )

        cols = st.columns(4, gap="medium")

        for i, (movie, score) in enumerate(st.session_state.recs):
            with cols[i % 4]:
                poster = fetch_poster(movie)

                clicked = clickable_images(
                    [poster],
                    titles=[movie],
                    div_style={
                        "display": "flex",
                        "justify-content": "center",
                        "margin-bottom": "8px",
                    },
                    img_style={
                        "width": "100%",
                        "height": "450px",
                        "border-radius": "12px",
                        "box-shadow": "0 4px 12px rgba(0,0,0,0.4)",
                        "cursor": "pointer",
                        "object-fit": "cover",
                    },
                    key=f"poster_{label}_{i}",
                )

                if clicked == 0:
                    st.session_state.picked_title = movie

                st.markdown(f'<p class="movie-title">{movie}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="similarity-score">Match: {score:.1%}</p>', unsafe_allow_html=True)

    with right:
        picked = st.session_state.picked_title

        if not picked:
            st.markdown(
                """
                <div class="drawer-card">
                    <div class="drawer-title">Details</div>
                    <div class="drawer-sub">Click a poster</div>
                    <p class="drawer-overview">Select any movie poster to view details here.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            row = get_row(model_data, picked)
            score_map = {m: s for m, s in st.session_state.recs}
            picked_score = score_map.get(picked, 0.0)

            overview = "No overview available."
            if row is not None:
                overview = fmt_value(row.get("overview")) or "No overview available."

            candidate_fields = [
                ("Release Date", "release_date"),
                ("Genres", model_data.get("genre_col") or "genres"),
                ("Language", "original_language"),
                ("Rating", "vote_average"),
                ("Votes", "vote_count"),
                ("Popularity", "popularity"),
                ("Runtime", "runtime"),
                ("Tagline", "tagline"),
                ("Status", "status"),
                ("Budget", "budget"),
                ("Revenue", "revenue"),
                ("Director", "director"),
            ]

            kv_items = []
            if row is not None:
                for label2, col in candidate_fields:
                    if col in row.index:
                        val = fmt_value(row.get(col), col)
                        if val:
                            kv_items.append((label2, val))

            kv_rows = ""
            if kv_items:
                for k, v in kv_items:
                    # Escape any HTML in the values to prevent injection
                    v_escaped = str(v).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    kv_rows += f'<div class="kv-row"><span class="kv-key">{k}</span><span class="kv-val">{v_escaped}</span></div>'
                
                kv_html = f'<div class="drawer-kv">{kv_rows}</div>'
            else:
                kv_html = ""

            # Escape overview text as well
            overview_escaped = overview.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            picked_escaped = picked.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

            full_html = f"""
<div class="drawer-card">
    <div class="drawer-title">{picked_escaped}</div>
    <div class="drawer-sub">Match: {picked_score:.1%}</div>
    <p class="drawer-overview">{overview_escaped}</p>
    {kv_html}
</div>
"""
            
            st.markdown(full_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()