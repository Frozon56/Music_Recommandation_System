# model.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def recommend_songs(song_name, df, top_n=5):
    if song_name not in df['name'].values:
        return None

    # Use only numeric features
    numeric_df = df.select_dtypes(include='number')
    
    # Song features for the input song
    song_vector = numeric_df[df['name'] == song_name].values

    # Cosine similarity
    similarities = cosine_similarity(song_vector, numeric_df)[0]

    # Attach similarity to dataframe
    df['similarity'] = similarities

    # Exclude input song, sort by similarity
    recommended = df[df['name'] != song_name].sort_values(by='similarity', ascending=False)

    return recommended[['name', 'artists', 'similarity']].head(top_n)
