import math

from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler, Normalizer, Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType


# Function to set up PySpark session
def setup_spark():
    return SparkSession.builder \
        .appName("Track_Similarity") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "4") \
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
        .getOrCreate()


def apply_tfidf(df, input_col, output_col):
    # Tokenize the text field
    tokenizer = Tokenizer(inputCol=input_col, outputCol=f"{output_col}_tokens")
    df = tokenizer.transform(df)

    # Remove stop words
    remover = StopWordsRemover(inputCol=f"{output_col}_tokens", outputCol=f"{output_col}_filtered")
    df = remover.transform(df)

    # Apply TF (term frequency)
    hashing_tf = HashingTF(inputCol=f"{output_col}_filtered", outputCol=f"{output_col}_raw_features", numFeatures=10000)
    df = hashing_tf.transform(df)

    # Apply IDF (inverse document frequency)
    idf = IDF(inputCol=f"{output_col}_raw_features", outputCol=f"{output_col}_tfidf")
    idf_model = idf.fit(df)
    df = idf_model.transform(df)

    return df


# Function to perform feature engineering
def feature_engineering(df, tfidf=False):

    if tfidf:
        df = apply_tfidf(df, "track_name", "track_name")
        df = apply_tfidf(df, "album_name", "album_name")
        df = apply_tfidf(df, "artist_name", "artist_name")
    else:
        # Encode categorical variables
        album_indexer = StringIndexer(inputCol="album_name", outputCol="album_index")
        artist_indexer = StringIndexer(inputCol="artist_name", outputCol="artist_index")
        track_indexer = StringIndexer(inputCol="track_name", outputCol="track_index")
        df = album_indexer.fit(df).transform(df)
        df = artist_indexer.fit(df).transform(df)
        df = track_indexer.fit(df).transform(df)

    # Combine features into a vector
    feature_cols = ["album_index", "artist_index", "track_index", "duration_ms"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    return df


# Function to scale features
def scale_features(df):
    scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)

    # Normalize features
    normalizer = Normalizer(inputCol="scaled_features", outputCol="normalized_features", p=2.0)
    df = normalizer.transform(df)
    # Convert normalized vector to array
    df = df.withColumn("normalized_array", vector_to_array(col("normalized_features")))
    return df


# Define a UDF for the dot product
def dot_product(vector1, vector2):
    return float(sum(x * y for x, y in zip(vector1, vector2)))


# Define a UDF for the magnitude (norm) of a vector
def magnitude(vector):
    return float(math.sqrt(sum(x ** 2 for x in vector)))


# Define the final cosine similarity function
def cosine_similarity(vector1, vector2):
    dot_prod = dot_product(vector1, vector2)
    mag1 = magnitude(vector1)
    mag2 = magnitude(vector2)

    # To prevent division by zero, return 0 if either magnitude is zero
    if mag1 == 0 or mag2 == 0:
        return 0.0
    else:
        return dot_prod / (mag1 * mag2)


# Register the UDF
cosine_similarity_udf = udf(cosine_similarity, DoubleType())


# Update pairwise similarity computation
def compute_pairwise_similarity_with_udf(df, tid=None):
    # Perform Cartesian join for pairwise comparisons
    pairwise_df = df.alias("a").crossJoin(df.alias("b")).filter(col("a.tid") < col("b.tid"))

    if tid is not None:
        pairwise_df = pairwise_df.filter(col("a.tid") == tid)

    # Calculate cosine similarity using the UDF
    pairwise_df = pairwise_df.withColumn(
        "cosine_similarity",
        cosine_similarity_udf(col("a.normalized_array"), col("b.normalized_array"))
    )
    return pairwise_df


def display_results(pairwise_df):
    pairwise_df.select(
        col("a.track_name").alias("track_1"),
        col("b.track_name").alias("track_2"),
        col("a.artist_name").alias("artists_1"),
        col("b.artist_name").alias("artist_2"),
        col("a.album_name").alias("album_1"),
        col("b.album_name").alias("album_2"),
        round("cosine_similarity", 3).alias("track similarity")
    ).orderBy(col("track similarity").desc()).show()


def calculate_similar_tracks(df, tid):
    setup_spark()
    # Select only the necessary columns and perform feature engineering
    df = df.select("tid", "album_name", "artist_name", "track_name", "duration_ms")
    df = feature_engineering(df)
    # Scale and normalize features
    df = df.select("tid", "features", "artist_name", "track_name")
    df = scale_features(df)
    # Compute pairwise similarity
    df = df.select("tid", "normalized_array", "artist_name", "track_name")
    print("Computing pairwise similarity...")
    pairwise_df = compute_pairwise_similarity_with_udf(df, tid)
    # Get the 500 tracks with the highest similarity to the given track with tid
    similar_tracks = pairwise_df.orderBy(col("cosine_similarity").desc()).limit(500)
    return similar_tracks


# Main function to orchestrate the workflow
def main():
    # Step 1: Set up Spark session
    spark = setup_spark()

    # Step 2: Load data
    df = spark.read.parquet('../CFDS/data/processed/df_tracks_spark', header=True, inferSchema=True)
    df = df.limit(1000)

    # Step 3: Feature engineering
    df = feature_engineering(df)

    # Step 4: Scale and normalize features
    df = scale_features(df)

    # Step 5: Compute pairwise similarity
    pairwise_df = compute_pairwise_similarity_with_udf(df)

    # Step 6: Display results
    display_results(pairwise_df)

    # Stop Spark session
    spark.stop()


# Entry point
if __name__ == "__main__":
    main()
