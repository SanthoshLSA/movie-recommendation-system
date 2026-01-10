import streamlit as st
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import joblib
import zipfile
import io

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_clickable_images import clickable_images


st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

OMDB_API_KEY = "f5127ade"

# ========================================
# UPDATE THIS WITH YOUR GOOGLE DRIVE LINK
# ========================================
# After uploading to Google Drive, replace YOUR_FILE_ID with your actual file ID
# Example: If your share link is https://drive.google.com/file/d/1ABC123XYZ456/view?usp=sharing
# Then your FILE_ID is: 1ABC123XYZ456
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?export=download&id=1OAzul4r3lki8sb5oYa0hbtl8Dmv8Szwk"

# Alternatively, use Streamlit secrets (recommended for production)
DATA_URL = st.secrets.get("DATA_URL", GOOGLE_DRIVE_URL)
# ----------------------------
# Download and setup data
# ----------------------------
@st.cache_data(show_spinner=False)
def download_and_extract_data():
    """Download data files from Google Drive if they don't exist locally"""
    
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    
    # Check if files already exist
    movies_csv = data_dir / "tmdb_5000_movies.csv"
    credits_csv = data_dir / "tmdb_5000_credits.csv"
    processed_pkl = processed_dir / "movies_processed.pkl"
    
    if movies_csv.exists() and credits_csv.exists() and processed_pkl.exists():
        return True
    
    # Create directories
    data_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    
    # Check if URL is configured
    if "https://drive.google.com/uc?export=download&id=1OAzul4r3lki8sb5oYa0hbtl8Dmv8Szwk" in DATA_URL:
        st.error("""
        **⚠️ Data URL not configured!**
        
        Please update the Google Drive link in `app.py`:
        
        1. Upload your `movie-data.zip` to Google Drive
        2. Get shareable link (Anyone with the link)
        3. Extract FILE_ID from the link
        4. Replace `YOUR_FILE_ID_HERE` in line 29 of app.py
        
        **Or run locally:**
        - Place CSV files in `data/` folder
        - Run `python src/preprocess.py`
        - Run `streamlit run app.py`
        """)
        st.stop()
        return False
    
    # Download data
    try:
        with st.spinner("⏬ Downloading movie data (first time only, ~50MB)..."):
            # For large files, Google Drive may return a confirmation page
            # We need to handle this
            session = requests.Session()
            response = session.get(DATA_URL, stream=True, timeout=300)
            
            # Check if it's a large file warning page
            if 'text/html' in response.headers.get('Content-Type', ''):
                # Extract the confirm token
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        params = {'id': DATA_URL.split('id=')[1], 'confirm': value}
                        response = session.get("https://drive.google.com/uc", params=params, stream=True, timeout=300)
                        break
            
            response.raise_for_status()
            
            # Show download progress
            total_size = int(response.headers.get('content-length', 0))
            
            if total_size > 0:
                progress_bar = st.progress(0)
                downloaded = 0
                chunks = []
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        chunks.append(chunk)
                        downloaded += len(chunk)
                        progress = min(downloaded / total_size, 1.0)
                        progress_bar.progress(progress)
                
                content = b''.join(chunks)
                progress_bar.empty()
            else:
                content = response.content
            
            # Extract zip file
            with zipfile.ZipFile(io.BytesIO(content)) as zip_ref:
                # List files in zip
                file_list = zip_ref.namelist()
                
                # Extract all files
                for file_info in file_list:
                    filename = Path(file_info).name
                    
                    # Skip directories and hidden files
                    if not filename or filename.startswith('.'):
                        continue
                    
                    if filename.endswith('.csv'):
                        # Extract CSVs to data/
                        with zip_ref.open(file_info) as source:
                            with open(data_dir / filename, 'wb') as target:
                                target.write(source.read())
                    
                    elif filename.endswith('.pkl'):
                        # Extract pkl to data/processed/
                        with zip_ref.open(file_info) as source:
                            with open(processed_pkl, 'wb') as target:
                                target.write(source.read())
            
            st.success("✅ Data downloaded and extracted successfully!")
            return True
            
    except requests.exceptions.RequestException as e:
        st.error(f"""
        **❌ Failed to download data files!**
        
        Error: {str(e)}
        
        **Troubleshooting:**
        1. Check that your Google Drive link is correct
        2. Make sure the file is shared as "Anyone with the link"
        3. Use the direct download format: `https://drive.google.com/uc?export=download&id=FILE_ID`
        
        **For local development:**
        1. Download datasets from [Kaggle TMDB](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
        2. Place files in the `data/` folder
        3. Run `python src/preprocess.py`
        """)
        st.stop()
        return False
    except zipfile.BadZipFile:
        st.error("""
        **❌ Invalid zip file!**
        
        The downloaded file is not a valid zip file. Please check:
        1. Your Google Drive link is correct
        2. The file uploaded to Google Drive is `movie-data.zip`
        3. The link points directly to the file (not a folder)
        """)
        st.stop()
        return False
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.stop()
        return False


# ----------------------------
# CSS
# ----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400..900&family=Poppins&display=swap');
    * { font-family: 'Orbitron', 'sans-serif'; }
    .stApp { background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid rgba(59, 130, 246, 0.2);
    }
    h1 { color: #ffffff; font-weight: 700; font-size: 2.5rem; font-family: 'Orbitron'; }
    .sidebar-header {
        color: #ffffff; font-size: 1.5rem; font-weight: 700;
        margin-bottom: 2rem; padding-bottom: 1rem;
        border-bottom: 3px solid #3b82f6;
    }
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: #ffffff; font-weight: 600; border-radius: 12px;
        padding: 0.875rem 2.5rem; width: 100%;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
    }
    .movie-title { color: #ffffff; font-weight: 600; font-size: 0.95rem; margin: 0.6rem 0 0.2rem 0; }
    .similarity-score { color: #3b82f6; font-weight: 500; font-size: 0.85rem; }
    .results-header {
        color: #ffffff; font-size: 1.5rem; font-weight: 700;
        padding: 1.25rem 1.5rem; background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6; border-radius: 8px; margin-bottom: 1.5rem;
    }
    .drawer-card {
        background: rgba(22, 33, 62, 0.55); border: 1px solid rgba(59, 130, 246, 0.18);
        border-radius: 14px; padding: 1.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        backdrop-filter: blur(6px);
    }
    .drawer-title { color: #ffffff; font-weight: 800; font-size: 1.2rem; margin-bottom: 0.5rem; }
    .drawer-sub { color: #60a5fa; font-weight: 600; font-size: 0.9rem; margin-bottom: 1rem; }
    .drawer-overview { color: #e0e0e0; line-height: 1.7; font-size: 0.95rem; }
    .drawer-kv {
        margin-top: 1rem; border-top: 1px solid rgba(59, 130, 246, 0.18);
        padding-top: 1rem; display: grid; grid-template-columns: 1fr; gap: 0.5rem;
    }
    .kv-row { display: flex; justify-content: space-between; gap: 0.75rem; font-size: 0.9rem; }
    .kv-key { color: #94a3b8; }
    .kv-val { color: #e5e7eb; text-align: right; }
</style>
""", unsafe_allow_html=True)


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
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    if "|" in s:
        return [g.strip() for g in s.split("|") if g.strip()]
    return [s]


@st.cache_data(show_spinner=False)
def load_and_process_data():
    processed_path = Path("data/processed/movies_processed.pkl")
    
    if not processed_path.exists():
        st.error("Processed data not found! Downloading...")
        download_and_extract_data()
    
    if not processed_path.exists():
        st.error("Failed to load data. Please check the setup.")
        st.stop()
    
    df = joblib.load(processed_path)
    
    if "title" not in df.columns or "combined_text" not in df.columns:
        st.error("Invalid data format!")
        st.stop()
    
    if "overview" not in df.columns:
        df["overview"] = ""
    
    # Detect genre column
    candidate_genre_cols = ["genres", "genre", "Genre"]
    genre_col = next((c for c in candidate_genre_cols if c in df.columns), None)
    
    # Build unique genre list
    unique_genres = set()
    if genre_col:
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
    
    if not genre_col or genre == "All":
        titles = df["title"].astype(str).tolist()[:limit]
        return [(t, 1.0) for t in titles]
    
    mask = df[genre_col].apply(lambda v: genre in parse_genres_cell(v))
    filtered = df[mask]
    titles = filtered["title"].astype(str).tolist()[:limit]
    return [(t, 1.0) for t in titles]


def get_row(model_data, title: str):
    df = model_data["df"]
    match = df[df["title"] == title]
    return match.iloc[0] if not match.empty else None


def fmt_value(v, field_name=None):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    
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
    
    if field_name == 'runtime':
        try:
            minutes = int(float(v))
            if minutes == 0:
                return None
            hours, mins = minutes // 60, minutes % 60
            return f"{hours}h {mins}m" if hours > 0 else f"{mins}m"
        except (ValueError, TypeError):
            return None
    
    if field_name == 'vote_average':
        try:
            rating = float(v)
            return f"{rating:.1f}/10" if rating > 0 else None
        except (ValueError, TypeError):
            return None
    
    if field_name == 'vote_count':
        try:
            count = int(float(v))
            if count == 0:
                return None
            return f"{count:,} votes" if count >= 1000 else f"{count} votes"
        except (ValueError, TypeError):
            return None
    
    if field_name == 'popularity':
        try:
            return f"{float(v):.0f}" if float(v) > 0 else None
        except (ValueError, TypeError):
            return None
    
    if field_name in ['genres', 'genre']:
        s = str(v).strip()
        return s.replace('|', ', ') if '|' in s and s else None
    
    s = str(v).strip()
    return s if s else None


# ----------------------------
# Main App
# ----------------------------
def main():
    # Ensure data is downloaded
    download_and_extract_data()
    
    # State
    if "recs" not in st.session_state:
        st.session_state.recs = []
    if "last_query" not in st.session_state:
        st.session_state.last_query = None
    if "picked_title" not in st.session_state:
        st.session_state.picked_title = None
    if "selected_genre" not in st.session_state:
        st.session_state.selected_genre = "All"
    
    with st.spinner("Loading model..."):
        model_data = load_and_process_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Configuration</div>', unsafe_allow_html=True)
        
        genre_options = ["All"]
        if model_data.get("unique_genres"):
            genre_options += model_data["unique_genres"]
        
        def on_genre_change():
            g = st.session_state.selected_genre
            st.session_state.recs = get_movies_by_genre(model_data, g, limit=20)
            st.session_state.last_query = (f"Genre: {g}", 20)
            st.session_state.picked_title = None
        
        st.selectbox("Select genre", options=genre_options, key="selected_genre", on_change=on_genre_change)
        
        movie_list = sorted(model_data["titles"])
        selected_movie = st.selectbox("Select a movie", movie_list, index=0)
        top_n = st.slider("Number of recommendations", 5, 25, 5, 1)
        
        if st.button("Find Similar Movies", type="primary"):
            with st.spinner("Analyzing similarities..."):
                st.session_state.recs = get_recommendations(model_data, selected_movie, top_n)
                st.session_state.last_query = (selected_movie, top_n)
                st.session_state.picked_title = None
    
    # Header
    st.markdown('<h1>Movie Recommendation System</h1>', unsafe_allow_html=True)
    
    if not st.session_state.recs:
        st.info('Select a genre to load movies, or pick a movie and click "Find Similar Movies".')
        return
    
    label, _ = st.session_state.last_query or ("Results", len(st.session_state.recs))
    
    # Layout
    left, right = st.columns([3, 1], gap="large")
    
    with left:
        st.markdown(f'<div class="results-header">Showing {len(st.session_state.recs)} movies for {label}</div>', unsafe_allow_html=True)
        
        cols = st.columns(4, gap="medium")
        
        for i, (movie, score) in enumerate(st.session_state.recs):
            with cols[i % 4]:
                poster = fetch_poster(movie)
                clicked = clickable_images(
                    [poster], titles=[movie],
                    div_style={"display": "flex", "justify-content": "center", "margin-bottom": "8px"},
                    img_style={"width": "100%", "height": "450px", "border-radius": "12px",
                              "box-shadow": "0 4px 12px rgba(0,0,0,0.4)", "cursor": "pointer", "object-fit": "cover"},
                    key=f"poster_{label}_{i}",
                )
                
                if clicked == 0:
                    st.session_state.picked_title = movie
                
                st.markdown(f'<p class="movie-title">{movie}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="similarity-score">Match: {score:.1%}</p>', unsafe_allow_html=True)
    
    with right:
        picked = st.session_state.picked_title
        
        if not picked:
            st.markdown("""
                <div class="drawer-card">
                    <div class="drawer-title">Details</div>
                    <div class="drawer-sub">Click a poster</div>
                    <p class="drawer-overview">Select any movie poster to view details here.</p>
                </div>
            """, unsafe_allow_html=True)
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
            
            kv_rows = ""
            if row is not None:
                for label2, col in candidate_fields:
                    if col in row.index:
                        val = fmt_value(row.get(col), col)
                        if val:
                            v_escaped = str(val).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                            kv_rows += f'<div class="kv-row"><span class="kv-key">{label2}</span><span class="kv-val">{v_escaped}</span></div>'
            
            kv_html = f'<div class="drawer-kv">{kv_rows}</div>' if kv_rows else ""
            
            overview_escaped = overview.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            picked_escaped = picked.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            st.markdown(f"""
<div class="drawer-card">
    <div class="drawer-title">{picked_escaped}</div>
    <div class="drawer-sub">Match: {picked_score:.1%}</div>
    <p class="drawer-overview">{overview_escaped}</p>
    {kv_html}
</div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
