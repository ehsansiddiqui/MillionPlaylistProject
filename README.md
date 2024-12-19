# Spotify Playlist Recommender

## Table of Contents
1. [The challenge](#the-task)
2. [The dataset](#the-dataset)
3. [Metrics](#metrics)
4. [Proposed Solutions](#proposed-solutions)
5. [Technology used](#technology-used)

## The challenge
TThe RecSys Challenge 2018 is organized by Spotify, The University of Massachusetts, Amherst, and Johannes Kepler University, Linz. The goal of the challenge is to develop a system for the task of automatic playlist continuation. Given a set of playlist features, participants’ systems shall generate a list of recommended tracks that can be added to that playlist, thereby ‘continuing’ the playlist.

The challenge is split into two parallel challenge tracks. In the main track, teams can only use data that is provided through the Million Playlist Dataset, while in the creative track participants can use external, public and freely available data sources to boost their system.

## The dataset
The Million Playlist Dataset (MPD) contains 1,000,000 playlists created by users on the Spotify platform. It can be used by researchers interested in exploring how to improve the music listening experience.

The MPD contains a million user-generated playlists. These playlists were created during the period of January 2010 through October 2017. Each playlist in the MPD contains a playlist title, the track list (including track metadata) editing information (last edit time, number of playlist edits) and other miscellaneous information about the playlist.

__Detailed description__

The Million Playlist Dataset consists of 1,000 slice files. These files have the naming convention of:

mpd.slice._STARTING\_PLAYLIST\_ID\_-\_ENDING\_PLAYLIST\_ID_.json

For example, the first 1,000 playlists in the MPD are in a file called
`mpd.slice.0-999.json` and the last 1,000 playlists are in a file called
`mpd.slice.999000-999999.json`.

Each slice file is a JSON dictionary with two fields:
*info* and *playlists*.


### `info` Field
The info field is a dictionary that contains general information about the particular slice:

   * **slice** - the range of slices that in in this particular file - such as 0-999
   * ***version*** -  - the current version of the MPD (which should be v1)
   * ***generated_on*** - a timestamp indicating when the slice was generated.

### `playlists` field
This is an array that typically contains 1,000 playlists. Each playlist is a dictionary that contains the following fields:


* ***pid*** - integer - playlist id - the MPD ID of this playlist. This is an integer between 0 and 999,999.
* ***name*** - string - the name of the playlist
* ***description*** - optional string - if present, the description given to the playlist.  Note that user-provided playlist descrptions are a relatively new feature of Spotify, so most playlists do not have descriptions.
* ***modified_at*** - seconds - timestamp (in seconds since the epoch) when this playlist was last updated. Times are rounded to midnight GMT of the date when the playlist was last updated.
* ***num_artists*** - the total number of unique artists for the tracks in the playlist.
* ***num_albums*** - the number of unique albums for the tracks in the playlist
* ***num_tracks*** - the number of tracks in the playlist
* ***num_followers*** - the number of followers this playlist had at the time the MPD was created. (Note that the follower count does not including the playlist creator)
* ***num_edits*** - the number of separate editing sessions. Tracks added in a two hour window are considered to be added in a single editing session.
* ***duration_ms*** - the total duration of all the tracks in the playlist (in milliseconds)
* ***collaborative*** -  boolean - if true, the playlist is a collaborative playlist. Multiple users may contribute tracks to a collaborative playlist.
* ***tracks*** - an array of information about each track in the playlist. Each element in the array is a dictionary with the following fields:
   * ***track_name*** - the name of the track
   * ***track_uri*** - the Spotify URI of the track
   * ***album_name*** - the name of the track's album
   * ***album_uri*** - the Spotify URI of the album
   * ***artist_name*** - the name of the track's primary artist
   * ***artist_uri*** - the Spotify URI of track's primary artist
   * ***duration_ms*** - the duration of the track in milliseconds
   * ***pos*** - the position of the track in the playlist (zero-based)

Here's an example of a typical playlist entry:

        {
            "name": "musical",
            "collaborative": "false",
            "pid": 5,
            "modified_at": 1493424000,
            "num_albums": 7,
            "num_tracks": 12,
            "num_followers": 1,
            "num_edits": 2,
            "duration_ms": 2657366,
            "num_artists": 6,
            "tracks": [
                {
                    "pos": 0,
                    "artist_name": "Degiheugi",
                    "track_uri": "spotify:track:7vqa3sDmtEaVJ2gcvxtRID",
                    "artist_uri": "spotify:artist:3V2paBXEoZIAhfZRJmo2jL",
                    "track_name": "Finalement",
                    "album_uri": "spotify:album:2KrRMJ9z7Xjoz1Az4O6UML",
                    "duration_ms": 166264,
                    "album_name": "Dancing Chords and Fireflies"
                },
                {
                    "pos": 1,
                    "artist_name": "Degiheugi",
                    "track_uri": "spotify:track:23EOmJivOZ88WJPUbIPjh6",
                    "artist_uri": "spotify:artist:3V2paBXEoZIAhfZRJmo2jL",
                    "track_name": "Betty",
                    "album_uri": "spotify:album:3lUSlvjUoHNA8IkNTqURqd",
                    "duration_ms": 235534,
                    "album_name": "Endless Smile"
                },
                {
                    "pos": 2,
                    "artist_name": "Degiheugi",
                    "track_uri": "spotify:track:1vaffTCJxkyqeJY7zF9a55",
                    "artist_uri": "spotify:artist:3V2paBXEoZIAhfZRJmo2jL",
                    "track_name": "Some Beat in My Head",
                    "album_uri": "spotify:album:2KrRMJ9z7Xjoz1Az4O6UML",
                    "duration_ms": 268050,
                    "album_name": "Dancing Chords and Fireflies"
                },
                // 8 tracks omitted
                {
                    "pos": 11,
                    "artist_name": "Mo' Horizons",
                    "track_uri": "spotify:track:7iwx00eBzeSSSy6xfESyWN",
                    "artist_uri": "spotify:artist:3tuX54dqgS8LsGUvNzgrpP",
                    "track_name": "Fever 99\u00b0",
                    "album_uri": "spotify:album:2Fg1t2tyOSGWkVYHlFfXVf",
                    "duration_ms": 364320,
                    "album_name": "Come Touch The Sun"
                }
            ],

        }


## Challenge Set
I build my own challenge Set based on criteria of official one but with some modification

__Test Set Format__

The challenge set consists of a single JSON dictionary with three fields:

- __date__ - the date the challenge set was generated. This should be "2018-01-16 08:47:28.198015"
- __version__ - the version of the challenge set. This should be "v1"
- __playlists__ - an array of 10,000 incomplete playlists. Each element in this array contains the following fields:
  - __pid__ - the playlist ID
  - __name__ - (optional) - the name of the playlist. For some challenge playlists, the name will be missing.
  - __num_holdouts__ - the number of tracks that have been omitted from the playlist
  - __tracks__ - a (possibly empty) array of tracks that are in the playlist. Each element of this array contains the following fields:
    - __pos__ - the position of the track in the playlist (zero offset)
    - __track_name__ - the name of the track
    - __track_uri__ - the Spotify URI of the track
    - __artist_name__ - the name of the primary artist of the track
    - __artist_uri__ - the Spotify URI of the primary artist of the track
    - __album_name__ - the name of the album that the track is on
    - __album_uri__ -- the Spotify URI of the album that the track is on
    - __duration_ms__ - the duration of the track in milliseconds
    - __num_samples__ the number of tracks included in the playlist
    - __num_tracks__ - the total number of tracks in the playlist.
    

## Metrics

__Precision@k__:
- Proportion of relevant tracks in the top-k recommendations.
- Indicates the accuracy of recommendations.

__Recall@k__:
- Proportion of relevant tracks retrieved among all relevant tracks in the dataset.
- Reflects the coverage of recommendations.

__Mean Average Precision (MAP)__:
- Averages precision across multiple playlists and their recommendations.
- Accounts for the order of recommendations.

__Normalized Discounted Cumulative Gain (nDCG)__:
- Evaluates the ranking quality of recommendations.
- Gives higher weight to relevant tracks ranked near the top.
- Ranges between 0 (poor ranking) and 1 (optimal ranking).

## Proposed Solutions
The playlist continuation challenge aims to generate a ranked list of recommended tracks to extend a given playlist. Our solution effectively handles two scenarios:

__Playlists with Seed Tracks__:
For playlists containing existing tracks, the similarity between seed tracks and other tracks in the dataset is computed using the cosine similarity method developed in the project. The most relevant tracks are then ranked and recommended.

__Playlists without Seed Tracks__:
For playlists lacking initial tracks, the system utilizes playlist metadata (such as name) to identify a similar playlist within the dataset. The tracks from this similar playlist serve as seed tracks for generating further recommendations.

This hybrid approach ensures adaptability to various input scenarios, leveraging track and playlist similarity measures developed earlier in the project. 


## Technology Used
- __Apache Spark__: Distributed data processing framework.
- __PySpark__: Python API for Spark.
- __Scikit-learn__: For TF-IDF vectorization and cosine similarity computations.
- __Parquet__: For optimized data storage.
