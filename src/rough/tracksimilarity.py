from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, abs


# Function to set up PySpark session
def setup_spark(app_name="TrackSimilarity"):
    return SparkSession.builder \
        .appName("Track_Similarity") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "4") \
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=35") \
        .getOrCreate()


# Function to compute similarity metrics
def compute_similarity(df):
    # Cartesian join to create all track pairs
    pairwise_df = df.alias("a").crossJoin(df.alias("b")).filter(col("a.tid") < col("b.tid"))

    # Add similarity metrics
    pairwise_df = pairwise_df.withColumn(
        "artist_similarity", when(col("a.artist_name") == col("b.artist_name"), lit(1.0)).otherwise(lit(0.0))
    ).withColumn(
        "album_similarity", when(col("a.album_name") == col("b.album_name"), lit(1.0)).otherwise(lit(0.0))
    ).withColumn(
        "duration_diff", abs(col("a.duration_ms") - col("b.duration_ms"))
    ).withColumn(
        "duration_similarity", 1 - (col("duration_diff") / lit(500000))  # Normalize duration difference to [0, 1]
    ).withColumn(
        "similarity_score",
        (col("artist_similarity") * 0.5) + (col("album_similarity") * 0.3) + (col("duration_similarity") * 0.2)
    )
    return pairwise_df

# Function to display results
def display_results(pairwise_df):
    pairwise_df.select(
        col("a.track_name").alias("track_1"),
        col("b.track_name").alias("track_2"),
        "similarity_score"
    ).orderBy(col("similarity_score").desc()).show(truncate=False)

# Main function to orchestrate the workflow
def main():
    # Step 1: Set up Spark session
    spark = setup_spark()

    # Step 2: Load data
    df = spark.read.parquet('../data/processed/df_tracks.parquet', header=True, inferSchema=True)
    df = df.limit(10000)

    # Step 3: Compute similarity metrics
    pairwise_df = compute_similarity(df)

    # Step 4: Display results
    display_results(pairwise_df)

    # Stop Spark session
    spark.stop()

# Entry point
if __name__ == "__main__":
    main()
