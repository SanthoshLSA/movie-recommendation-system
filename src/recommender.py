"""
Day 3: TF-IDF Vectorization + Cosine Similarity Matrix + recommend() function
Purpose: Convert text → numeric vectors → similarity scores → top recommendations
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from pathlib import Path
from typing import List, Tuple

class MovieRecommender:
    def __init__(self, processed_data_path: str):
        """
        Load processed data and build TF-IDF + similarity matrix.
        Purpose: Precompute everything so recommendations are instant.
        """
        self.movies_df = joblib.load(processed_data_path)
        self._build_model()
        
    def _build_model(self):
        """Convert text → TF-IDF vectors → similarity matrix."""
        print("🔄 Building TF-IDF model...")
        
        # 1. TF-IDF Vectorizer
        # Purpose: Convert text to numeric vectors, TF-IDF = Term Frequency * Inverse Document Frequency
        self.vectorizer = TfidfVectorizer(
            max_features=5000,      # Top 5000 most important words
            stop_words='english',   # Ignore common words (the, a, an...)
            ngram_range=(1, 2)      # Single words + 2-word phrases ("super hero")
        )
        
        # Fit + transform all movie texts → sparse matrix (4800 movies x 5000 words)
        tfidf_matrix = self.vectorizer.fit_transform(self.movies_df['combined_text'])
        print(f"✅ TF-IDF matrix: {tfidf_matrix.shape} (movies x words)")
        
        # 2. Cosine Similarity Matrix
        # Purpose: For every pair of movies, compute similarity score (0-1)
        # Matrix is 4800 x 4800, each entry = how similar movie i is to movie j
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        print(f"✅ Similarity matrix: {self.similarity_matrix.shape}")
        
        # 3. Title → index mapping (for fast lookup)
        self.title_to_idx = {title: idx for idx, title in enumerate(self.movies_df['title'])}
        print(f"✅ Ready! {len(self.title_to_idx)} movies indexed")
    
    def recommend(self, title: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Core recommendation function.
        Purpose: Given movie title → return top N similar movies + scores.
        """
        if title not in self.title_to_idx:
            raise ValueError(f"❌ Movie '{title}' not found. Try exact title match.")
        
        # 1. Get this movie's index
        movie_idx = self.title_to_idx[title]
        
        # 2. Get similarity scores for ALL movies to this movie (row movie_idx)
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        
        # 3. Sort by score DESC, exclude itself (score=1.0)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        # 4. Return (title, score) pairs
        recommendations = [(self.movies_df.iloc[idx]['title'], score) 
                          for idx, score in sim_scores]
        
        print(f"✅ Recommended {len(recommendations)} movies for '{title}':")
        for movie, score in recommendations:
            print(f"  {movie:<50} {score:.3f}")
        
        return recommendations

def save_model(recommender: MovieRecommender, model_path: str):
    """Save model components for Streamlit app."""
    model_data = {
        'movies_df': recommender.movies_df,
        'similarity_matrix': recommender.similarity_matrix,
        'title_to_idx': recommender.title_to_idx,
        'vectorizer': recommender.vectorizer  # Optional, for new movies
    }
    joblib.dump(model_data, model_path)
    print(f"💾 Model saved: {model_path}")

def main():
    """Test the recommender."""
    PROCESSED_PATH = Path("data/processed/movies_processed.pkl")
    MODEL_PATH = Path("data/processed/model.pkl")
    
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError("❌ Run preprocess.py first!")
    
    # Build recommender
    rec = MovieRecommender(PROCESSED_PATH)
    
    # Test it!
    rec.recommend("Avatar")
    rec.recommend("The Dark Knight")
    
    # Save for Day 4 Streamlit
    MODEL_PATH.parent.mkdir(exist_ok=True)
    save_model(rec, MODEL_PATH)
    print(f"\n🎉 Day 3 complete! Model ready for Streamlit.")

if __name__ == "__main__":
    main()
