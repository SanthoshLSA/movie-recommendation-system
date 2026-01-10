import pandas as pd
import joblib
from pathlib import Path
import ast
import numpy as np


def safe_literal_eval(val):
    """Safely evaluate string representations of lists/dicts."""
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []


def extract_names(obj_list, key='name', limit=None):
    """Extract names from a list of dictionaries."""
    if not isinstance(obj_list, list):
        return []
    names = [item[key] for item in obj_list if isinstance(item, dict) and key in item]
    if limit:
        names = names[:limit]
    return names


def extract_director(crew_list):
    """Extract director name from crew list."""
    if not isinstance(crew_list, list):
        return None
    for person in crew_list:
        if isinstance(person, dict) and person.get('job') == 'Director':
            return person.get('name')
    return None


def preprocess_tmdb_data():
    """
    Combine TMDB movies and credits datasets, extract features,
    and prepare for the recommendation system.
    """
    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    movies_path = Path("data/raw/tmdb_5000_movies.csv")
    credits_path = Path("data/raw/tmdb_5000_credits.csv")
    
    if not movies_path.exists():
        raise FileNotFoundError(f"Could not find {movies_path}")
    if not credits_path.exists():
        raise FileNotFoundError(f"Could not find {credits_path}")
    
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    
    print(f"Loaded {len(movies)} movies and {len(credits)} credits")
    
    # Merge datasets on movie_id (or title if ids don't match)
    if 'id' in movies.columns and 'movie_id' in credits.columns:
        df = movies.merge(credits, left_on='id', right_on='movie_id', how='left', suffixes=('', '_credits'))
    else:
        df = movies.merge(credits, on='title', how='left', suffixes=('', '_credits'))
    
    print(f"Combined dataset: {len(df)} rows")
    print(f"Columns after merge: {list(df.columns)}")
    
    # Parse JSON-like columns
    print("Parsing JSON columns...")
    
    # Genres
    if 'genres' in df.columns:
        df['genres_parsed'] = df['genres'].apply(safe_literal_eval)
        df['genre_names'] = df['genres_parsed'].apply(lambda x: extract_names(x))
        df['genres'] = df['genre_names'].apply(lambda x: '|'.join(x) if x else '')
    
    # Keywords
    if 'keywords' in df.columns:
        df['keywords_parsed'] = df['keywords'].apply(safe_literal_eval)
        df['keyword_names'] = df['keywords_parsed'].apply(lambda x: extract_names(x))
    
    # Cast (top 5 actors)
    if 'cast' in df.columns:
        df['cast_parsed'] = df['cast'].apply(safe_literal_eval)
        df['cast_names'] = df['cast_parsed'].apply(lambda x: extract_names(x, limit=5))
    
    # Crew (extract director)
    if 'crew' in df.columns:
        df['crew_parsed'] = df['crew'].apply(safe_literal_eval)
        df['director'] = df['crew_parsed'].apply(extract_director)
    
    # Production companies
    if 'production_companies' in df.columns:
        df['companies_parsed'] = df['production_companies'].apply(safe_literal_eval)
        df['company_names'] = df['companies_parsed'].apply(lambda x: extract_names(x, limit=3))
    
    # Create combined_text for TF-IDF
    print("Creating combined text features...")
    text_parts = []
    
    # Overview
    if 'overview' in df.columns:
        text_parts.append(df['overview'].fillna(''))
    
    # Genres
    if 'genre_names' in df.columns:
        text_parts.append(df['genre_names'].apply(lambda x: ' '.join(x) if isinstance(x, list) else ''))
    
    # Keywords
    if 'keyword_names' in df.columns:
        text_parts.append(df['keyword_names'].apply(lambda x: ' '.join(x) if isinstance(x, list) else ''))
    
    # Cast
    if 'cast_names' in df.columns:
        text_parts.append(df['cast_names'].apply(lambda x: ' '.join(x) if isinstance(x, list) else ''))
    
    # Director
    if 'director' in df.columns:
        text_parts.append(df['director'].fillna(''))
    
    # Tagline
    if 'tagline' in df.columns:
        text_parts.append(df['tagline'].fillna(''))
    
    # Combine all text
    df['combined_text'] = pd.Series([' '.join(str(p) for p in parts) for parts in zip(*text_parts)])
    
    # Select final columns
    columns_to_keep = [
        'id',
        'title',
        'overview',
        'genres',  # pipe-separated string
        'release_date',
        'runtime',
        'vote_average',
        'vote_count',
        'popularity',
        'budget',
        'revenue',
        'original_language',
        'status',
        'tagline',
        'combined_text',
    ]
    
    # Add optional columns if they exist
    optional_cols = ['director', 'homepage']
    for col in optional_cols:
        if col in df.columns:
            columns_to_keep.append(col)
    
    # Keep only available columns
    available_cols = [col for col in columns_to_keep if col in df.columns]
    df_processed = df[available_cols].copy()
    
    # Clean data
    print("Cleaning data...")
    
    # Remove duplicates
    df_processed = df_processed.drop_duplicates(subset=['title'], keep='first')
    
    # Remove rows with missing titles
    df_processed = df_processed[df_processed['title'].notna()]
    
    # Fill NaN values
    if 'overview' in df_processed.columns:
        df_processed['overview'] = df_processed['overview'].fillna('No overview available.')
    
    if 'genres' in df_processed.columns:
        df_processed['genres'] = df_processed['genres'].fillna('')
    
    # Convert numeric columns
    numeric_cols = ['vote_average', 'vote_count', 'popularity', 'budget', 'revenue', 'runtime']
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Reset index
    df_processed = df_processed.reset_index(drop=True)
    
    # Save
    output_path = output_dir / "movies_processed.pkl"
    joblib.dump(df_processed, output_path)
    
    print(f"\n✓ Successfully processed {len(df_processed)} movies")
    print(f"✓ Saved to {output_path}")
    print(f"\nColumns included: {list(df_processed.columns)}")
    print(f"\nSample data:")
    print(df_processed[['title', 'genres', 'vote_average']].head())
    
    # Show genre distribution
    if 'genres' in df_processed.columns:
        all_genres = []
        for genres_str in df_processed['genres'].dropna():
            all_genres.extend(genres_str.split('|'))
        genre_counts = pd.Series(all_genres).value_counts()
        print(f"\nTop 10 genres:")
        print(genre_counts.head(10))
    
    return df_processed


if __name__ == "__main__":
    try:
        preprocess_tmdb_data()
        print("\n✓ Preprocessing complete! You can now run your Streamlit app.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()