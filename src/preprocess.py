"""
Day 1-2: Full TMDB Dataset Processing (Movies + Credits)
Purpose: Load both CSVs → merge → parse JSON → create rich combined text → save
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
import joblib

def load_full_dataset() -> pd.DataFrame:
    """Load movies.csv + credits.csv and merge correctly."""
    movies_path = Path("data/raw/tmdb_5000_movies.csv")
    credits_path = Path("data/raw/tmdb_5000_credits.csv")
    
    movies_df = pd.read_csv(movies_path)
    credits_df = pd.read_csv(credits_path)
    
    # Credits CSV has 'movie_id', movies has 'id' → rename for merge
    credits_df = credits_df.rename(columns={'movie_id': 'id'})
    
    # Now merge works
    df = movies_df.merge(credits_df, on='id', how='left', suffixes=('', '_credits'))
    print(f"✅ Merged: {len(df)} movies with cast/crew data")
    print("📊 Sample columns after merge:", df[['title', 'genres', 'cast', 'crew']].columns.tolist())
    print("\n📈 First row sample:")
    print(df[['title', 'genres', 'cast']].head(1))
    return df

def parse_json_column(df: pd.DataFrame, col: str, max_items: int = 3) -> pd.DataFrame:
    """
    JSON → clean names string. Purpose: Extract readable info from messy JSON.
    genres: ["Action", "Adventure"] → "Action Adventure"
    cast: [{"name": "Chris Hemsworth"}] → "Chris Hemsworth Robert Downey Jr."
    """
    def extract_names(json_str):
        if pd.isna(json_str):
            return ""
        try:
            items = json.loads(json_str)
            if col == 'crew':
                # For crew, only get Directors (job == 'Director')
                directors = [item.get('name', '') for item in items if item.get('job') == 'Director']
                return " ".join(directors[:max_items])
            else:
                # genres/keywords/cast: top N names
                names = [item.get('name', '') for item in items[:max_items]]
                return " ".join(names)
        except:
            return ""
    
    parsed_col = col + '_parsed'
    df[parsed_col] = df[col].apply(extract_names)
    print(f"✅ Parsed {col}: '{df[parsed_col].iloc[0]}'")
    return df

def create_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rich feature combination: genres + keywords + top cast + director + overview.
    Purpose: Single text field representing ALL movie characteristics for TF-IDF.
    """
    text_features = [
        'genres_parsed',    # "Action Adventure Fantasy"
        'keywords_parsed',  # "superhero marvel avengers"
        'cast_parsed',      # "Chris Hemsworth Natalie Portman"
        'crew_parsed',      # "Taika Waititi" (director)
        'overview'          # "Thor returns to Asgard..."
    ]
    
    # Fill NaN → ""
    for feature in text_features:
        df[feature] = df[feature].fillna("")
    
    # Combine into ONE text field per movie
    df['combined_text'] = df[text_features].agg(' '.join, axis=1).str.lower()
    
    # Clean: collapse multiple spaces
    df['combined_text'] = df['combined_text'].apply(
        lambda x: re.sub(r'\s+', ' ', x)
    )
    
    print("✅ Combined text created. Sample (first 200 chars):")
    print(repr(df['combined_text'].iloc[0][:200]))
    return df

def main():
    """Complete pipeline."""
    RAW_MOVIES = Path("data/raw/tmdb_5000_movies.csv")
    RAW_CREDITS = Path("data/raw/tmdb_5000_credits.csv")
    PROCESSED_PATH = Path("data/processed/movies_processed.pkl")
    
    # Check files exist
    for path in [RAW_MOVIES, RAW_CREDITS]:
        if not path.exists():
            raise FileNotFoundError(f"❌ Missing {path}. Download from Kaggle TMDB dataset!")
    
    # 1. Load + merge
    df = load_full_dataset()
    
    # 2. Parse all JSON columns
    json_cols = ['genres', 'keywords', 'cast', 'crew']
    for col in json_cols:
        df = parse_json_column(df, col)
    
    # 3. Create combined text
    df = create_combined_text(df)
    
    # 4. Final cleaned dataset
    keep_cols = ['id', 'title', 'combined_text', 'overview']
    df_final = df[keep_cols].drop_duplicates(subset=['title']).reset_index(drop=True)
    
    print(f"\n🎉 Final dataset ready: {len(df_final)} unique movies")
    print("📝 Sample titles + text length:")
    print(df_final[['title', 'combined_text']].head())
    
    # 5. Save
    PROCESSED_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(df_final, PROCESSED_PATH)
    print(f"💾 Saved processed data: {PROCESSED_PATH}")
    
    return df_final

if __name__ == "__main__":
    df_processed = main()
