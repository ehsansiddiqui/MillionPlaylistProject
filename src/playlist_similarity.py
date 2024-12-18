from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, array_intersect, size, abs as spark_abs, udf
from pyspark.sql.types import DoubleType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cosine_sim


# In this file, task 4 of the project is tackled. The task is to propose a method to compute the similarity
# (score ranging from 0 to 1) between playlists. The similarity is based on the following attributes:
#       - (1) Overlap (Jaccard similarity) between tracks, artists, and albums.
#       - (2) Similarity based on the number of followers and duration of the playlist.
#       - (3) Similarity between the names of the playlists.
# The final similarity score is computed as a weighted sum of these three individual similarity scores.

# Define a UDF to compute cosine similarity between playlist names
def cosine_similarity(name1, name2):
    try:
        vectorizer = TfidfVectorizer().fit_transform([name1, name2])
        vectors = vectorizer.toarray()
        return float(cosine_sim(vectors)[0, 1])
    # Handle the case where the playlist name is empty or contains only stop words
    except ValueError:
        return 0.0


# Register the UDF with PySpark
cosine_similarity_udf = udf(cosine_similarity, DoubleType())


def compute_playlist_similarity(pids, weights, output_path):
    """
    Compute similarity between playlists using PySpark. The similarity is based on the intersection of tracks, artists,
    and albums in two playlists, as well as the number of followers and duration of the playlist.

    Args:
        pids (list): List of playlist IDs to compare.
        weights (dict): Dictionary of weights for the similarity components.
        output_path (str): Path to the directory containing processed parquet files.
        continuation_pid (int): ID of the playlist to continue.

    Returns:
        None
    """
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("Playlist Similarity") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.cores", "6") \
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
        .getOrCreate()

    # Load preprocessed data
    df_playlists_tracks = spark.read.parquet(f"{output_path}/df_playlists_spark")
    df_playlists_info = spark.read.parquet(f"{output_path}/df_playlists_info_spark")

    # Select only the necessary columns
    df_playlists_tracks = df_playlists_tracks.select("pid", "track_uri", "artist_uri", "album_uri")
    df_playlists_info = df_playlists_info.select("pid", "num_followers", "duration_ms", "name")

    # Filter to only keep playlists to compare
    df_playlists_tracks = df_playlists_tracks.filter(col("pid").isin(pids))
    df_playlists_info = df_playlists_info.filter(col("pid").isin(pids)).repartition("pid")

    # Aggregate tracks, artists, and albums per playlist
    df_playlist_data = df_playlists_tracks.groupBy("pid") \
        .agg(collect_list("track_uri").alias("track_uris"),
             collect_list("artist_uri").alias("artist_uris"),
             collect_list("album_uri").alias("album_uris")) \
        .repartition("pid")

    # Add number of followers, duration, and name to the playlist data
    df_playlist_data = df_playlist_data.join(df_playlists_info, "pid")

    # Normalize the number of followers and duration to be between 0 and 1
    df_playlist_data = df_playlist_data.withColumn("num_followers_norm", col("num_followers") / df_playlist_data.agg(
        {"num_followers": "max"}).collect()[0][0]) \
        .withColumn("duration_norm", col("duration_ms") / df_playlist_data.agg(
        {"duration_ms": "max"}).collect()[0][0])

    # Select only the necessary columns
    df_playlist_data = df_playlist_data.select("pid", "track_uris", "artist_uris", "album_uris", "num_followers_norm", "duration_norm", "name")

    # Cross join playlists to compute similarity
    df_cross = df_playlist_data.alias("p1").crossJoin(df_playlist_data.alias("p2")) \
        .filter(col("p1.pid") < col("p2.pid"))  # Avoid duplicate and self-comparisons

    # Compute track, artist, album, followers, and duration similarity
    df_similarity = df_cross \
        .withColumn("track_similarity", size(array_intersect(col("p1.track_uris"), col("p2.track_uris"))) /
                    (size(col("p1.track_uris")) + size(col("p2.track_uris")) - size(
                        array_intersect(col("p1.track_uris"), col("p2.track_uris"))))) \
        .withColumn("artist_similarity", size(array_intersect(col("p1.artist_uris"), col("p2.artist_uris"))) /
                    (size(col("p1.artist_uris")) + size(col("p2.artist_uris")) - size(
                        array_intersect(col("p1.artist_uris"), col("p2.artist_uris"))))) \
        .withColumn("album_similarity", size(array_intersect(col("p1.album_uris"), col("p2.album_uris"))) /
                    (size(col("p1.album_uris")) + size(col("p2.album_uris")) - size(
                        array_intersect(col("p1.album_uris"), col("p2.album_uris"))))) \
        .withColumn("followers_similarity", 1 - spark_abs(col("p1.num_followers_norm") - col("p2.num_followers_norm"))) \
        .withColumn("duration_similarity", 1 - spark_abs(col("p1.duration_norm") - col("p2.duration_norm"))) \
        .withColumn("name_similarity", cosine_similarity_udf(col("p1.name"), col("p2.name")))

    # Combine all similarities using a weighted sum
    df_final_similarity = df_similarity \
        .withColumn("final_similarity",
                    weights["track_similarity"] * col("track_similarity") +
                    weights["artist_similarity"] * col("artist_similarity") +
                    weights["album_similarity"] * col("album_similarity") +
                    weights["followers_similarity"] * col("followers_similarity") +
                    weights["duration_similarity"] * col("duration_similarity") +
                    weights["name_similarity"] * col("name_similarity")) \
        .select("p1.pid", "p2.pid", "final_similarity")

    # Display results in descending order of similarity
    df_final_similarity.select("p1.pid", "p2.pid", "final_similarity") \
        .orderBy("final_similarity", ascending=False).show(10, False)

    # Stop Spark session
    spark.stop()


def compute_playlist_similarity_name(pids, output_path, pid_to_continue):
    """
    Compute similarity between playlists based on playlist names using PySpark.

    Args:
        pids (list): List of playlist IDs to compare.
        output_path (str): Path to the directory containing processed parquet files.
        pid_to_continue (int): ID of the playlist to continue.

    Returns:
        None
    """
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("Playlist Similarity") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "4") \
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
        .getOrCreate()

    # Load preprocessed data
    df_playlists_info = spark.read.parquet(f"{output_path}/df_playlists_info_spark")
    df_playlists_test_info = spark.read.parquet(f"{output_path}/df_playlists_test_info_spark")

    # Select only the necessary columns
    df_playlists_info = df_playlists_info.select("name", "pid")
    df_playlists_test_info = df_playlists_test_info.select("name", "pid")

    # Filter to only keep playlists to compare
    df_playlists_info = df_playlists_info.filter(col("pid").isin(pids))
    df_playlists_test_info = df_playlists_test_info.filter(col("pid") == pid_to_continue)

    # Concatenate df_playlists_info and df_playlists_test_info
    df_playlists_info = df_playlists_info.union(df_playlists_test_info)

    # Cross join playlists to compute similarity
    df_cross = df_playlists_info.alias("p1").crossJoin(df_playlists_info.alias("p2")) \
        .filter(col("p1.pid") < col("p2.pid"))  # Avoid duplicate and self-comparisons

    # Filter the results to only calculate the similarity with the playlist to continue
    df_cross = df_cross.filter(col("p2.pid") == pid_to_continue)

    # Compute name similarity
    df_similarity = df_cross.withColumn("name_similarity", cosine_similarity_udf(col("p1.name"), col("p2.name")))

    # Filter the results to only keep the playlist to continue
    df_similarity = df_similarity.orderBy("name_similarity", ascending=False)

    # Return the entry corresponding to the playlist that is most similar to the playlist to continue
    return df_similarity.select("p1.pid").first()


if __name__ == "__main__":
    output_path = "../CFDS/data/processed"
    pids = [x for x in range(5_000)]
    weights = {
        "track_similarity": 6 / 24,
        "artist_similarity": 6 / 24,
        "album_similarity": 6 / 24,
        "name_similarity": 4 / 24,
        "followers_similarity": 1 / 24,
        "duration_similarity": 1 / 24
    }
    compute_playlist_similarity(pids, weights, output_path)
