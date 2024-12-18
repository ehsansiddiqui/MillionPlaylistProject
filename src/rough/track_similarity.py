import pandas as pd
import numpy as np
import os
from collections import defaultdict
from itertools import groupby
from gensim.models import Word2Vec
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity


df_tracks = pd.read_parquet('../data/processed/df_tracks.parquet', engine="pyarrow")
df_playlist_info = pd.read_parquet('../data/processed/df_playlists_info.parquet', engine="pyarrow")
playlist_df = pd.read_parquet('../data/processed/df_playlists.parquet', engine="pyarrow")


playlists = playlist_df.groupby('pid')['tid'].apply(list).tolist()
print("Playlist:", playlists[0])


# Train Word2Vec model on playlists
model = Word2Vec(sentences=playlists, vector_size=100, window=5, min_count=1, workers=4)

# Example: Get embedding for a track
track_id = 0  # Example track ID
embedding = model.wv[track_id]
print("Embedding for Track 0:", embedding)

def compute_embedding_similarity(track1_id, track2_id, model):
    # Retrieve embeddings
    embed1 = model.wv[track1_id]
    embed2 = model.wv[track2_id]
    # Compute cosine similarity
    return cosine_similarity([embed1], [embed2])[0][0]



# List of track IDs (subset for demonstration)
track_ids = playlist_df['tid'].unique().tolist()

# Compute similarity matrix
similarity_matrix = np.zeros((len(track_ids), len(track_ids)))
for i, track1 in enumerate(track_ids):
    for j, track2 in enumerate(track_ids):
        similarity_matrix[i, j] = compute_embedding_similarity(track1, track2, model)

# Convert to DataFrame for readability
similarity_df = pd.DataFrame(similarity_matrix, index=track_ids, columns=track_ids)
print(similarity_df)


track1_id = 0  # Example track IDs
track2_id = 1

similarity_score = compute_embedding_similarity(track1_id, track2_id, model)
print(f"Similarity between Track {track1_id} and Track {track2_id}: {similarity_score:.2f}")