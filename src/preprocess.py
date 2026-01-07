import streamlit as st
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ⚠️ REPLACE WITH YOUR REAL TMDB KEY (50+ chars, starts with eyJ...)
TMDB_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIxMjM0NTY3ODkwIiwic3ViIjoiMTIzNDU2Nzg5MCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.abc123def456"
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"

@st.cache_data(ttl=3600)
def fetch_poster(title: str) -> str:
    """Fetch movie poster with better error handling."""
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title[:50]}  # Truncate long titles
    try:
        res = requests.get(url, params=params, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get("results") and data["results"][0].get("poster_path"):
                return f"{TMDB_IMAGE_URL}{data['results'][0]['poster_path']}"
        st.caption(f"API status: {res.status_code}")  # Debug
    except Exception as e:
        st.caption(f"API error: {str(e)[:30]}...")
    return None

@st.cache_data
def load_and_process_data():
    """Load processed data + build model live."""
    processed_path = Path("data/processed/movies_processed.pkl")
    if not processed_path.exists():
        st.error("❌ Run `python src/preprocess.py` first!")
        st.stop()
    
    df = joblib.load(processed_path)
    titles = df['title'].tolist()
    
    # Build model (fast after cache)
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))
    tfidf = vectorizer.fit_transform(df['combined_text'])
    similarity = cosine_similarity(tfidf)
    
    title_to_idx = {t: i for i, t in enumerate(titles)}
    
    return {
        'titles': titles,
        'title_to_idx': title_to_idx,
        'similarity': similarity,
        'df': df
    }

def get_recommendations(model_data, title: str, top_n: int = 10):
    titles = model_data['titles']
    title_to_idx = model_data['title_to_idx']
    similarity = model_data['similarity']
    
    if title not in title_to_idx:
        return []
    
    idx = title_to_idx[title]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    return [(titles[i], score) for i, score in scores]

def main():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
    
    st.title("🎬 Movie Recommendation System")
    st.markdown("**Content-Based TF-IDF + Live TMDB Posters**")
    
    # API Key Check
    if len(TMDB_API_KEY) < 40 or TMDB_API_KEY.startswith("eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIxMjM0"):  
        st.error("🚨 **Line 13**: Replace `TMDB_API_KEY` with your real TMDB key!")
        st.info("👉 [Get free key](https://www.themoviedb.org/settings/api)")
        st.stop()
    
    # Load model
    with st.spinner("🔄 Loading movies + building TF-IDF model..."):
        model_data = load_and_process_data()
    st.success(f"✅ {len(model_data['titles'])} movies ready!")
    
    # DEBUG SECTION - Remove after posters work
    with st.sidebar:
        st.header("🛠️ Debug Tools")
        if st.button("🖼️ Test API (Avatar)"):
            poster = fetch_poster("Avatar")
            st.write(f"**Avatar poster**: {poster[:100] if poster else '❌ FAILED'}")
        
        st.header("📊 Model Stats")
        st.metric("Movies", len(model_data['titles']))
        st.metric("TF-IDF Features", model_data['similarity'].shape[1])
    
    # Main UI
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("🎯 Pick a Movie")
        movie_list = sorted(model_data['titles'])[:2000] + sorted(model_data['titles'])[-2000:]  # Top/bottom for variety
        selected_movie = st.selectbox("Choose:", movie_list, index=100)
        
        top_n = st.slider("Recommendations:", 6, 15, 9, 3)
        recommend_btn = st.button("🎥 GET RECOMMENDATIONS", type="primary", use_container_width=True)
    
    with col2:
        if recommend_btn:
            with st.spinner("🎯 Finding similar movies..."):
                recs = get_recommendations(model_data, selected_movie, top_n)
                
                if recs:
                    st.markdown(f"### 🔥 Movies like **{selected_movie}**")
                    
                    # Netflix-style poster grid
                    cols = st.columns(3)
                    for i, (movie, score) in enumerate(recs):
                        with cols[i % 3]:
                            poster = fetch_poster(movie)
                            if poster:
                                st.image(poster, use_container_width=True)
                            else:
                                st.markdown("🖼️")
                            
                            st.markdown(f"**{movie}**")
                            color = "🟢" if score > 0.4 else "🟡" if score > 0.2 else "🔴"
                            st.caption(f"{color} **{score:.3f}** similarity")
                else:
                    st.error("❌ No recommendations found.")
    
    # Footer
    st.markdown("---")
    st.markdown("*Resume-ready ML project | SanthoshLSA | Deployed with Streamlit + TMDB API*")

if __name__ == "__main__":
    main()
