from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

spark = SparkSession.builder.appName("ParquetExample").getOrCreate()

df_playlists_info = spark.read.parquet("/Users/daanweetjens/Downloads/Group Project/df_playlists_info_spark/*.parquet")
df_tracks = spark.read.parquet("/Users/daanweetjens/Downloads/Group Project/df_tracks_spark/*.parquet")
df_playlists_tracks = spark.read.parquet("/Users/daanweetjens/Downloads/Group Project/df_playlists_spark/*.parquet")
df_playlists_test_info = spark.read.parquet("/Users/daanweetjens/Downloads/Group Project/df_playlists_test_info_spark/*.parquet")
df_playlists_test = spark.read.parquet("/Users/daanweetjens/Downloads/Group Project/df_playlists_test_spark/*.parquet")

# Average number of Followers, Artists, Albums, and Tracks per playlist
avg_followers = df_playlists_info.agg({"num_followers": "avg"}).collect()[0][0]
avg_artists = df_playlists_info.agg({"num_artists": "avg"}).collect()[0][0]
avg_albums = df_playlists_info.agg({"num_albums": "avg"}).collect()[0][0]
avg_tracks = df_playlists_info.agg({"num_tracks": "avg"}).collect()[0][0]
print(f"Average Number of Followers per playlist: {avg_followers}")
print(f"Average Number of artists per playlist: {avg_artists}")
print(f"Average Number of albums per playlist: {avg_albums}")
print(f"Average Number of tracks per playlist: {avg_tracks}")

# Prepare data for boxplots and histograms
df_boxplot = df_playlists_info.select("num_followers", "num_artists", "num_albums", "num_tracks", "modified_at").toPandas()

# Determine the threshold for capping the number of followers
threshold = df_boxplot['num_followers'].quantile(0.99)  # Cap at the 99.9th percentile
print(f"Threshold for capping the number of followers: {threshold}")

# Cap the number of followers at the threshold
df_boxplot['num_followers_capped'] = df_boxplot['num_followers'].apply(lambda x: min(x, threshold))

# Filter out the outliers
df_filtered = df_boxplot[df_boxplot['num_followers'] <= threshold]

# Filter out the outliers for the number of tracks feature specifically for the scatterplot matrix
df_filtered_tracks = df_filtered[df_filtered['num_tracks'] <= 350]

# Scatterplot matrix including all features with improved visibility on the diagonal and without extreme outliers for the number of tracks
sns.pairplot(df_filtered_tracks[['num_followers', 'num_artists', 'num_albums', 'num_tracks', 'modified_at']],
             diag_kind='kde', plot_kws={'alpha': 0.5, 's': 10})
plt.suptitle('Scatterplot Matrix of All Features (Without Extreme Outliers for Number of Tracks)')
plt.show()

# Create boxplot with capped followers
plt.figure(figsize=(12, 8))
sns.boxplot(y='num_followers_capped', data=df_boxplot)
plt.title('Boxplot of Number of Followers per Playlist (Capped)')
plt.ylabel('Number of Followers')
plt.show()

# Create histogram with capped followers
plt.figure(figsize=(12, 8))
sns.histplot(df_boxplot['num_followers_capped'], bins=50, kde=True)
plt.title('Histogram of Number of Followers per Playlist (Capped)')
plt.xlabel('Number of Followers')
plt.ylabel('Frequency')
plt.show()

# Create boxplots of number of Artists, Albums, and Tracks per playlist
plt.figure(figsize=(12, 8))
sns.boxplot(y='num_artists', data=df_boxplot)
plt.title('Boxplot of Number of Artists per Playlist')
plt.ylabel('Number of Artists')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(y='num_albums', data=df_boxplot)
plt.title('Boxplot of Number of Albums per Playlist')
plt.ylabel('Number of Albums')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(y='num_tracks', data=df_boxplot)
plt.title('Boxplot of Number of Tracks per Playlist')
plt.ylabel('Number of Tracks')
plt.show()

# Create histograms of number of Artists, Albums, and Tracks per playlist
plt.figure(figsize=(12, 8))
sns.histplot(df_boxplot['num_artists'], bins=30, kde=True)
plt.xlim(1, 250)
plt.title('Histogram of Number of Artists per Playlist')
plt.xlabel('Number of Artists')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(df_boxplot['num_albums'], bins=30, kde=True)
plt.title('Histogram of Number of Albums per Playlist')
plt.xlabel('Number of Albums')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(df_boxplot['num_tracks'], bins=30, kde=True)
plt.title('Histogram of Number of Tracks per Playlist')
plt.xlim(1, 300)
plt.xlabel('Number of Tracks')
plt.ylabel('Frequency')
plt.show()

# Use the filtered DataFrame for correlation analysis
df_filtered_spark = spark.createDataFrame(df_filtered)

# Correlation between Number of Artists and Number of Albums
correlation_artists_albums = df_filtered_spark.stat.corr("num_artists", "num_albums")
print(f"Correlation Between Number of Artists and Number of Albums (Without Extreme Outliers): {correlation_artists_albums}")

# Correlation between Number of Tracks and Number of Albums
correlation_tracks_albums = df_filtered_spark.stat.corr("num_tracks", "num_albums")
print(f"Correlation Between Number of Tracks and Number of Albums (Without Extreme Outliers): {correlation_tracks_albums}")

# Correlation between Number of Artists and Number of Tracks
correlation_artists_tracks = df_filtered_spark.stat.corr("num_artists", "num_tracks")
print(f"Correlation Between Number of Artists and Number of Tracks (Without Extreme Outliers): {correlation_artists_tracks}")

## Correlation Between Followers and Number of Tracks
correlation = df_filtered_spark.stat.corr("num_followers", "num_tracks")
print(f"Correlation Between Followers and Number of Tracks (Without Extreme Outliers): {correlation}")

# Scatter plot for correlation between Followers and Number of Tracks without log transformation
df_filtered_sample = df_filtered_spark.sample(False, 0.1).toPandas()
plt.figure(figsize=(10, 6))
sns.scatterplot(x='num_tracks', y='num_followers', data=df_filtered_sample)
plt.title('Correlation between number of tracks and number of followers (Without Extreme Outliers)')
plt.xlabel('Number of Tracks')
plt.ylabel('Number of Followers')
plt.show()

# Scatter histogram for correlation between Followers and Number of Tracks
sns.jointplot(x='num_tracks', y='num_followers', data=df_filtered_sample, kind='scatter', marginal_kws=dict(bins=50, fill=True))
plt.suptitle('Scatter Histogram of Number of Tracks and Number of Followers (Without Extreme Outliers)')
plt.show()

# Scatterplot matrix for correlation between Followers and Number of Tracks
sns.pairplot(df_filtered_sample[['num_tracks', 'num_followers']])
plt.suptitle('Scatterplot Matrix of Number of Tracks and Number of Followers (Without Extreme Outliers)')
plt.show()

# Scatterplot heatmap for correlation between Followers and Number of Tracks
sns.jointplot(x='num_tracks', y='num_followers', data=df_filtered_sample, kind='hex', gridsize=50, cmap='Blues')
plt.suptitle('Scatterplot Heatmap of Number of Tracks and Number of Followers (Without Extreme Outliers)')
plt.show()

# Correlation between Time of Creation and Number of followers
correlation_time_followers = df_filtered_spark.stat.corr("modified_at", "num_followers")
print(f"Correlation Between Time of Creation and Number of Followers (Without Extreme Outliers): {correlation_time_followers}")

# Scatter plot for correlation between Time of Creation and Number of followers
plt.figure(figsize=(10, 6))
sns.scatterplot(x="modified_at", y='num_followers', data=df_filtered_sample)
plt.title('Correlation Between time of creation and number of followers (Without Extreme Outliers)')
plt.xlabel('Time of creation')
plt.ylabel('Number of Followers')
plt.show()

# Scatter histogram for correlation between Time of Creation and Number of followers
sns.jointplot(x='modified_at', y='num_followers', data=df_filtered_sample, kind='scatter', marginal_kws=dict(bins=50, fill=True))
plt.suptitle('Scatter Histogram of Time of Creation and Number of Followers (Without Extreme Outliers)')
plt.show()

# Scatterplot matrix for correlation between Time of Creation and Number of followers
sns.pairplot(df_filtered_sample[['modified_at', 'num_followers']])
plt.suptitle('Scatterplot Matrix of Time of Creation and Number of Followers (Without Extreme Outliers)')
plt.show()

# Scatterplot heatmap for correlation between Time of Creation and Number of followers
sns.jointplot(x='modified_at', y='num_followers', data=df_filtered_sample, kind='hex', gridsize=50, cmap='Blues')
plt.suptitle('Scatterplot Heatmap of Time of Creation and Number of Followers (Without Extreme Outliers)')
plt.show()

# Most Common Opening and closing Tracks
most_common_opening_tracks = df_playlists_tracks.filter(df_playlists_tracks["pos"] == 0).groupBy("track_uri").count().orderBy("count", ascending=False).limit(10)
most_common_opening_tracks_with_names = most_common_opening_tracks.join(df_tracks, "track_uri").select("track_name", "count")
print("Most common opening tracks:")
most_common_opening_tracks_with_names.show()

common_closing_tracks = df_playlists_tracks.groupBy("pid").agg({"pos": "max"}).alias("max_pos")
closing_tracks = df_playlists_tracks.join(common_closing_tracks, (df_playlists_tracks["pid"] == common_closing_tracks["pid"]) & (df_playlists_tracks["pos"] == common_closing_tracks["max(pos)"])).groupBy("track_uri").count().orderBy("count", ascending=False).limit(10)
closing_tracks_with_names = closing_tracks.join(df_tracks, "track_uri").select("track_name", "count")
print("Most common closing trakcs:")
closing_tracks_with_names.show()

# Most Common Tracks, albums, artists
most_common_tracks = df_playlists_tracks.groupBy("track_uri").count().orderBy("count", ascending=False).limit(10)
most_common_tracks_with_names = most_common_tracks.join(df_tracks, "track_uri").select("track_name", "count")
print("Most common tracks:")
most_common_tracks_with_names.show()

most_common_artists = df_tracks.groupBy("artist_uri").count().orderBy("count", ascending=False).limit(10)
most_common_artists_with_names = most_common_artists.join(df_tracks, "artist_uri").select("artist_name", "count").distinct()
print("Most common artists:")
most_common_artists_with_names.show()

most_common_albums = df_tracks.groupBy("album_uri").count().orderBy("count", ascending=False).limit(10)
most_common_albums_with_names = most_common_albums.join(df_tracks, "album_uri").select("album_name", "count").distinct()
print("Most common albums:")
most_common_albums_with_names.show()

# Most Common Genres (based on playlist name)
genres = ["rock", "pop", "hip-hop", "jazz", "classical"]
for genre in genres:
    genre_count = df_playlists_info.filter(col("name").contains(genre)).count()
    print(f"Number of playlists with genre {genre}: {genre_count}")
