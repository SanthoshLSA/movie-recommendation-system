"""
Movie Recommender System
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

OMDB_API_KEY = "f5127ade"

# Custom CSS for dark theme with orange accents
st.markdown("""
<style>
    .stApp {
        background-color: #0a0a0a;
    }
    
    .main .block-container {
        padding-top: 2rem;
    }
    
    h1 {
        color: #ff8c42;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    
    h3 {
        color: #ff8c42;
        font-weight: 600;
    }
    
    .stSelectbox label, .stSlider label {
        color: #ff8c42 !important;
        font-weight: 600;
    }
    
    .stButton > button {
        background-color: #ff8c42;
        color: #0a0a0a;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #ff6b1a;
        box-shadow: 0 4px 12px rgba(255, 140, 66, 0.4);
    }
    
    .stImage {
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
    }
    
    .stMarkdown {
        color: #e0e0e0;
    }
    
    [data-testid="stSpinner"] {
        color: #ff8c42;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def fetch_poster(title: str) -> str:
    """Fetch movie poster from OMDb API."""
    params = {
        "apikey": OMDB_API_KEY,
        "t": title[:40],
        "r": "json"
    }
    try:
        response = requests.get("http://www.omdbapi.com/", params=params, timeout=8)
        data = response.json()
        if data.get("Response") == "True" and data.get("Poster") and data["Poster"] != "N/A":
            return data["Poster"]
    except:
        pass
    return None

@st.cache_data
def load_and_process_data():
    processed_path = Path("data/processed/movies_processed.pkl")
    if not processed_path.exists():
        st.error("Run `python src/preprocess.py` first!")
        st.stop()
    
    df = joblib.load(processed_path)
    titles = df['title'].tolist()
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
    st.set_page_config(
        page_title="Movie Recommender",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("Movie Recommendation System")
    
    with st.spinner("Loading model..."):
        model_data = load_and_process_data()
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Select Movie")
        movie_list = sorted(model_data['titles'])[:1000] + sorted(model_data['titles'])[-1000:]
        selected_movie = st.selectbox("Choose a movie:", movie_list, index=200)
        top_n = st.slider("Number of recommendations:", 6, 12, 9, 3)
        recommend_btn = st.button("Get Recommendations", type="primary")
    
    with col2:
        if recommend_btn:
            with st.spinner("Finding matches..."):
                recs = get_recommendations(model_data, selected_movie, top_n)
                
                if recs:
                    st.markdown(f"### Recommendations based on {selected_movie}")
                    cols = st.columns(3)
                    for i, (movie, score) in enumerate(recs):
                        with cols[i % 3]:
                            poster = fetch_poster(movie)
                            if poster:
                                st.image(poster, use_container_width=True)
                            else:
                                st.markdown("No poster available")
                            
                            st.markdown(f"**{movie}**")
                            st.caption(f"Similarity: **{score:.3f}**")

if __name__ == "__main__":
    main()