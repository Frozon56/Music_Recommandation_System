# streamlit_app.py

import streamlit as st
import pandas as pd
from model import recommend_songs  # Make sure model.py is in the same folder

# Load your dataset
df = pd.read_csv("data.csv")

# Page configuration
st.set_page_config(page_title="Music Recommendation System", layout="centered")
st.title("🎵 Music Recommendation System")

st.markdown("Enter a song name to get recommendations based on audio features.")

# Input fields
song_input = st.text_input("🎶 Song Name:")
top_n = st.slider("How many recommendations?", 1, 10, 5)

# Recommendation logic
if st.button("Recommend"):
    if song_input.strip() == "":
        st.warning("Please enter a song name.")
    else:
        results = recommend_songs(song_input, df, top_n)
        if results is None:
            st.error("Song not found in the dataset.")
        else:
            st.success(f"Top {top_n} recommendations for: {song_input}")
            st.dataframe(results.reset_index(drop=True))
