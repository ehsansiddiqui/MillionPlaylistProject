from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, lit, monotonically_increasing_id
from pyspark.sql.functions import broadcast

def create_dataframes(data_path, challenge_file, output_path, partitions):
    """
    Process playlist and track data using Spark, and save the output as CSV files.

    Args:
        data_path (str): Path to the directory containing JSON files.
        challenge_file (str): Path to the challenge_set.json file.
        output_path (str): Path to save the resulting CSV files.

    Returns:
        None
    """

     # Initialize SparkSession with parallelism configurations
    spark = SparkSession.builder \
        .appName("Create Spark DataFrames for Playlist Dataset") \
        .config("spark.sql.shuffle.partitions", str(partitions)) \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "4") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    
    # Define column sets
    playlist_col = ['collaborative', 'duration_ms', 'modified_at', 
                    'name', 'num_albums', 'num_artists', 'num_edits',
                    'num_followers', 'num_tracks', 'pid']
    tracks_col = ['album_name', 'album_uri', 'artist_name', 'artist_uri', 
                  'duration_ms', 'track_name', 'track_uri'] 
    playlist_test_col = ['name', 'num_holdouts', 'num_samples', 'num_tracks', 'pid']

    # Read all JSON files in the directory
    df_raw = spark.read.option("multiline", "true").json(f"{data_path}/*.json").repartition(partitions)

    # Extract playlist data
    df_playlists = df_raw.select(explode(col("playlists")).alias("playlist"))

    # Extract playlist-level information
    df_playlists_info = df_playlists.select(*[col(f"playlist.{cols}").alias(cols) for cols in playlist_col])

    # Extract track-level information
    df_tracks = df_playlists.select(
        col("playlist.pid").alias("pid"),
        explode(col("playlist.tracks")).alias("track")
    ).select(
        col("track.track_uri").alias("track_uri1"),
        *[col(f"track.{cols}").alias(cols) for cols in tracks_col]
    ).drop_duplicates()

    
    
    # Add unique track ID (tid) in parallel
    df_tracks = df_tracks.withColumn("tid", monotonically_increasing_id())

    # Join playlist and track information to create a relationship DataFrame
    df_playlists_tracks = df_playlists.select(
        col("playlist.pid").alias("pid"),
        explode(col("playlist.tracks")).alias("track")
    ).select(
        col("pid"),
        col("track.track_uri").alias("track_uri"),
        col("track.pos").alias("pos")
    )
    
    df_playlists_tracks = df_playlists_tracks.join(broadcast(df_tracks), on="track_uri", how="left")
    
    # Join with track ID (tid)
#     df_playlists_tracks = df_playlists_tracks.join(df_tracks, on="track_uri", how="left")

    # Process challenge set
    df_challenge_raw = spark.read.option("multiline", "true").json(challenge_file).repartition(partitions)
    df_playlists_test_info = df_challenge_raw.select(
        explode(col("playlists")).alias("playlist")
    ).select(*[col(f"playlist.{cols}").alias(cols) for cols in playlist_test_col])

    df_playlists_test = df_challenge_raw.select(
        explode(col("playlists")).alias("playlist")
    ).select(
        col("playlist.pid").alias("pid"),
        explode(col("playlist.tracks")).alias("track")
    ).select(
        col("pid"),
        col("track.track_uri").alias("track_uri"),
        col("track.pos").alias("pos")
    ).join(broadcast(df_tracks), on="track_uri", how="left")

    # Save DataFrames as CSV files
    df_playlists_info.write.parquet(f"{output_path}/df_playlists_info_spark", header=True, mode="overwrite")
    df_tracks.write.parquet(f"{output_path}/df_tracks_spark", header=True, mode="overwrite")
    df_playlists_tracks.write.parquet(f"{output_path}/df_playlists_spark", header=True, mode="overwrite")
    df_playlists_test_info.write.parquet(f"{output_path}/df_playlists_test_info_spark", header=True, mode="overwrite")
    df_playlists_test.write.parquet(f"{output_path}/df_playlists_test_spark", header=True, mode="overwrite")

    print("DataFrames successfully created and saved as parquet files.")

if __name__ == "__main__":
    # Define paths
    data_path = "../data/raw/"  # Path to directory with JSON files
    challenge_file = "../data/raw/challenge_set.json"  # Challenge set file
    output_path = "../data/processed/"  # Output directory for CSV files
     # Number of partitions for parallelism
    num_partitions = 50  # Adjust based on your system's resources

    # Run the function
    create_dataframes(data_path, challenge_file, output_path, partitions=num_partitions)