# Movie Recommendation System

A content-based movie recommender using TMDB dataset with 4,800+ movies.

## Live Demo
[View App](https://movie-recommendation-system-pnyhiy9hd2ofuljtf4leqr.streamlit.app/)

## Features
- Find similar movies based on content
- Browse by genre
- View detailed movie information
- Beautiful dark-themed UI

## Tech Stack
- Streamlit
- Scikit-learn (TF-IDF + Cosine Similarity)
- TMDB 5000 Movie Dataset

## Local Setup
```bash
git clone https://github.com/SanthoshLSA/movie-recommendation-system.git
cd movie-recommendation-system
pip install -r requirements.txt
streamlit run app.py
```

Data is automatically downloaded on first run.
