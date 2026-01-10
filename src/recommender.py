import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from pathlib import Path

class MovieRecommender:
    def __init__(self, processed_data_path: str):
        self.movies_df = joblib.load(processed_data_path)
        self._build_model()
    
    def _build_model(self):
        print("🔄 Building TF-IDF model...")
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1,2))
        tfidf_matrix = self.vectorizer.fit_transform(self.movies_df['combined_text'])
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        self.title_to_idx = {title: idx for idx, title in enumerate(self.movies_df['title'])}
        print(f"✅ Ready! {len(self.title_to_idx)} movies")
    
    def recommend(self, title: str, top_n: int = 5):
        if title not in self.title_to_idx:
            return []
        idx = self.title_to_idx[title]
        scores = sorted(list(enumerate(self.similarity_matrix[idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]
        return [(self.movies_df.iloc[i]['title'], score) for i, score in scores]

def main():
    PROCESSED_PATH = Path("data/processed/movies_processed.pkl")
    if not PROCESSED_PATH.exists():
        print("❌ Run preprocess.py first!")
        return
    
    rec = MovieRecommender(PROCESSED_PATH)
    print("\n🎬 Test recs:")
    for test_movie in ["Avatar", "The Dark Knight", "Inception"]:
        recs = rec.recommend(test_movie)
        print(f"\n{test_movie}:")
        for movie, score in recs:
            print(f"  {movie:<40} {score:.3f}")

if __name__ == "__main__":
    main()
