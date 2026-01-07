"""
Day 4: Streamlit Web App
Purpose: Load model → interactive UI → show recommendations with scores
Run with: streamlit run app.py
"""

import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


# Page config
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

@st.cache_data
def load_model_data():
    """Load model data once."""
    model_path = Path("data/processed/model.pkl")
    if not model_path.exists():
        st.error("❌ Run `python src/recommender.py` first!")
        st.stop()
    return joblib.load(model_path)

def get_recommendations(model_data, title: str, top_n: int = 10):
    """Pure function: recommend movies (no class)."""
    movies_df = model_data['movies_df']
    similarity_matrix = model_data['similarity_matrix']
    title_to_idx = model_data['title_to_idx']
    
    if title not in title_to_idx:
        return []
    
    movie_idx = title_to_idx[title]
    sim_scores = list(enumerate(similarity_matrix[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    return [(movies_df.iloc[idx]['title'], score) 
            for idx, score in sim_scores]

# In main(), replace load_model() call with:
model_data = load_model_data()

def main():
    st.title("🎬 Movie Recommendation System")
    st.markdown("**Content-Based Filtering using TF-IDF + Cosine Similarity**")
    
    # Load model data
    with st.spinner("Loading model..."):
        model_data = load_model_data()
    
    st.success(f"✅ Loaded {len(model_data['movies_df'])} movies!")
    
    # ... (sidebar same)
    
    # Main UI
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("🎯 Select Movie")
        movie_list = sorted(model_data['movies_df']['title'].tolist())
        selected_movie = st.selectbox("Choose a movie:", movie_list)
        
        top_n = st.slider("Number of recommendations:", 5, 15, 10)
        recommend_btn = st.button("🎬 Recommend Movies", type="primary")
    
    with col2:
        if recommend_btn and selected_movie:
            with st.spinner("Finding similar movies..."):
                recs = get_recommendations(model_data, selected_movie, top_n)
                
                if recs:
                    st.subheader(f"🔥 Top {len(recs)} movies like **{selected_movie}**")
                    
                    rec_df = pd.DataFrame(recs, columns=['Movie', 'Similarity Score'])
                    st.dataframe(
                        rec_df,
                        column_config={
                            "Movie": st.column_config.TextColumn("Movie Title"),
                            "Similarity Score": st.column_config.NumberColumn(
                                "Score", format="%.3f", min_value=0.0, max_value=1.0
                            )
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.error("No recommendations found.")
        else:
            st.info("👆 Select a movie and click Recommend!")

    
    # Footer
    st.markdown("---")
    st.markdown("*Built for technical interviews/portfolio*")

if __name__ == "__main__":
    main()
