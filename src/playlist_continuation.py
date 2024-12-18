from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from track_similarity import calculate_similar_tracks
from playlist_similarity import compute_playlist_similarity_name


# In this file, task 5 of the project is tackled. The task is to solve the playlist continuation problem challenge.
# The goal of the challenge is to develop a system for the task of automatic playlist continuation. Given playlist data,
# the system should generate a list of recommended tracks that can be added to that playlist, thereby "continuing"
# the playlist. The task is defined formally as follows:
#
# Input: A user-created playlist, represented by:
#       - A playlist ID (pid) for the playlist to be continued from the challenge dataset.
# Output: A list of 500 recommended candidate tracks, ordered by relevance in decreasing order.
#
# Note that the system should also be able to cope with playlists for which no initial seed tracks are given!
#
# The general idea of the solution proposed here is the following:
#   - If the playlist has seed tracks, we will use the information from these tracks to compute the similarity between
#     the seed tracks and all other tracks in the dataset. We will then recommend the most similar tracks.
#   - If the playlist has no seed tracks, we will use the playlist metadata to find a similar playlist in the dataset.
#     We will then use the tracks from this similar playlist as seed tracks to recommend tracks for the new playlist.
#
# For this, we use the functionality implemented in the previous tasks, where we implemented a method to compute the
# similarity between playlists (playlist_similarity.py) and tracks (track_similarity.py).
#
# To test our implementation, we will use the provided challenge dataset.

def continue_playlist(pid, path):
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("Playlist Similarity") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "4") \
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
        .getOrCreate()

    # Load the challenge dataset
    df_playlists_test = spark.read.parquet(f"{path}/df_playlists_test_spark")
    # Load the playlists-tracks dataset
    df_playlists_tracks = spark.read.parquet(f"{path}/df_playlists_spark")
    # Load the tracks dataset
    df_tracks = spark.read.parquet(f"{path}/df_tracks_spark")

    # Get the playlist information for the playlist to continue
    df_to_continue = df_playlists_test.filter(col("pid") == pid)

    # If the resulting dataframe is empty, the playlist does not have any seed tracks. If it is not empty, the dataframe
    # contains the seed tracks for the playlist.
    if df_to_continue.count() == 0:
        print("Playlist does not have seed tracks. Finding similar playlist.")
        # Find a similar playlist based on the playlist metadata. This returns a Row with the most similar playlist ID.
        most_similar_playlist = compute_playlist_similarity_name([x for x in range(100_000)], path, pid)
        print(f"Most similar playlist: {most_similar_playlist.pid}. Getting seed tracks:")
        # Get the seed tracks from the similar playlist
        seed_tracks = df_playlists_tracks.filter(col("pid") == most_similar_playlist.pid)
        seed_tracks.show()
    else:
        print("Playlist has seed tracks.")
        # Get the seed tracks from the playlist that needs to be continued
        seed_tracks = df_playlists_tracks.filter(col("pid") == pid)
        seed_tracks.show()

    # Pick a seed track from the playlist
    seed_track = seed_tracks.first()
    seed_track_data = df_tracks.filter(col("tid") == seed_track.tid)
    seed_track_data.show()
    sample_tracks = df_tracks.sample(0.1)
    sample_tracks = sample_tracks.union(seed_track_data)
    # Compute the similarity between the seed tracks and all other tracks in the dataset
    print("Calculating similar tracks.")
    cont_tracks = calculate_similar_tracks(sample_tracks, seed_track.tid)
    cont_tracks.show()
    return cont_tracks


if __name__ == "__main__":
    pid_to_continue_no_tracks = 1000002
    pid_to_continue_with_tracks = 1013752
    continue_playlist(pid_to_continue_no_tracks, "../CFDS/data/processed")
