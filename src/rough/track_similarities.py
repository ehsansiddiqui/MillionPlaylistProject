from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.ml.feature import OneHotEncoder, StringIndexer, MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pyspark.ml.linalg import Vectors
from pyspark.ml.linalg import DenseVector


# Initialize Spark session
spark = SparkSession.builder \
        .appName("SpotifyHybridEmbedding")\
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.cores", "4") \
        .config("spark.dynamicAllocation.enabled", "true") \
        .config("spark.dynamicAllocation.initialExecutors", "2") \
        .config("spark.dynamicAllocation.maxExecutors", "10") \
        .config("spark.memory.fraction", "0.6") \
        .config("spark.memory.storageFraction", "0.2") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
        .getOrCreate()


def generate_contextual_embeddings(playlist_df, vector_size=25, window_size=3, min_count=5):
    """
    Generate contextual embeddings for playlists using Word2Vec.
    
    Parameters:
    - file_path (str): Path to the input parquet file containing playlist data.
    - vector_size (int): Dimensionality of the word vectors.
    - window_size (int): The size of the context window.
    - min_count (int): Minimum frequency for a track ID to be included.
    
    Returns:
    - DataFrame: A Spark DataFrame with playlist IDs and their contextual embeddings.
    """

     # Group track IDs by playlist
    playlist_sequences = playlist_df.groupBy("pid").agg(
        F.collect_list("tid").alias("track_sequence")
    )
    
    # Convert track IDs to strings in track_sequence
    playlist_sequences = playlist_sequences.withColumn(
        "track_sequence", col("track_sequence").cast("array<string>")
    )
    
    # Train Word2Vec model
    word2vec = Word2Vec(
        vectorSize=vector_size, 
        windowSize=window_size, 
        minCount=min_count, 
        inputCol="track_sequence", 
        outputCol="contextual_embedding"
    )
    model = word2vec.fit(playlist_sequences)
    
    # Transform to get contextual embeddings
    playlist_with_embeddings = model.transform(playlist_sequences)
    contextual_embeddings = playlist_with_embeddings.select("pid", "contextual_embedding")
    
    return contextual_embeddings


def generate_feature_based_embeddings(df_tracks):
    """
    Generate feature-based embeddings for tracks using metadata features.
    
    Parameters:
    - file_path (str): Path to the input parquet file containing track metadata.
    
    Returns:
    - DataFrame: A Spark DataFrame with track IDs and their feature-based embeddings.
    """

     # Index and encode categorical features
    artist_indexer = StringIndexer(inputCol="artist_name", outputCol="artist_index")
    album_indexer = StringIndexer(inputCol="album_name", outputCol="album_index")
    artist_encoder = OneHotEncoder(inputCol="artist_index", outputCol="artist_vec")
    album_encoder = OneHotEncoder(inputCol="album_index", outputCol="album_vec")
    
    # VectorAssembler to convert duration_ms into a vector
    duration_vector_assembler = VectorAssembler(inputCols=["duration_ms"], outputCol="duration_vector")
    
    # MinMaxScaler for duration
    scaler = MinMaxScaler(inputCol="duration_vector", outputCol="scaled_duration")
    
    # Combine features
    assembler = VectorAssembler(
        inputCols=["artist_vec", "album_vec", "scaled_duration"], 
        outputCol="metadata_embedding"
    )
    
    # Build pipeline
    pipeline = Pipeline(stages=[
        artist_indexer,
        album_indexer,
        artist_encoder,
        album_encoder,
        duration_vector_assembler,
        scaler,
        assembler
    ])
    
    # Train pipeline model
    metadata_model = pipeline.fit(df_tracks)
    
    # Transform to get feature-based embeddings
    track_with_metadata = metadata_model.transform(df_tracks)
    metadata_embeddings = track_with_metadata.select("tid", "metadata_embedding")
    
    return metadata_embeddings

def combine_embeddings(contextual_embeddings, metadata_embeddings):
    """
    Combine contextual and metadata embeddings into hybrid embeddings.
    
    Parameters:
    - contextual_embeddings (DataFrame): Spark DataFrame with contextual embeddings.
    - metadata_embeddings (DataFrame): Spark DataFrame with metadata embeddings.
    
    Returns:
    - DataFrame: A Spark DataFrame with hybrid embeddings combining contextual and metadata features.
    """
    # Define UDF to concatenate embeddings
    def combine_embeddings(contextual, metadata):
        return list(contextual) + list(metadata)
    combine_udf = udf(combine_embeddings, ArrayType(FloatType()))
    
    # Join datasets on playlist ID and track ID
    hybrid_df = contextual_embeddings.join(
        metadata_embeddings, 
        contextual_embeddings.pid == metadata_embeddings.tid, 
        "inner"
    )
    
    # Add hybrid embeddings
    hybrid_df = hybrid_df.withColumn(
        "hybrid_embedding", combine_udf(col("contextual_embedding"), col("metadata_embedding"))
    )
    
    return hybrid_df.select("pid", "hybrid_embedding")