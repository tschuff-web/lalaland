"""
Thomas Schuff
Professor MacIsaac
CPSC 222 - Data Science
"""

# --- IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib as mpl
import requests
import json
import base64
import requests
from urllib.parse import quote_plus
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import csv
import time
import os

# --- GLOBAL VARIABLES ---
CLIENT_ID = "9eda6ddd3d0742aebd8d82b0172b807f"
CLIENT_SECRET = "d4090a65cb5044af8b09e1f1e82bf8b4"


# --- FUNCTIONS ---
def load_metadata_cache():
    """
    Open and copy the song metadata cache into a dictionary (to reduce redundant Spotify API requests)
    """
    with open("lalaland/output files/metadata_cache.json", "r") as infile:
        metadata_cache = json.load(infile)
    return metadata_cache


def load_library_tracks():
    """
    Open and copy the song information from my saved library to a DataFrame to reduce the number of API calls necessary to get song info
    """
    with open("lalaland/dataset files/Apple Music Library Tracks.json", "r") as infile:
        library_track_data = json.load(infile)

    library_df = pd.DataFrame(library_track_data)

    # rename columns to be the same as my other DataFrames
    library_df = library_df.rename(
        columns={"Title": "Song Name", "Artist": "Artist Name", "Album": "Album Name"}
    )
    return library_df


def get_spotify_token():
    """
    Get the access token for the Spotify Web API using my Client ID and Client Secret.
    """
    message = CLIENT_ID + ":" + CLIENT_SECRET
    message_bytes = message.encode("ascii")
    base_64_bytes = base64.b64encode(message_bytes)
    encoded_message = base_64_bytes.decode("ascii")

    headers = {"Authorization": "Basic " + encoded_message}
    body = {"grant_type": "client_credentials"}
    endpoint = "https://accounts.spotify.com/api/token"

    response = requests.post(url=endpoint, headers=headers, data=body)
    response_js = response.json()
    return response_js["access_token"]


def load_data(filename):
    """
    Loads the data from a .csv file into a Pandas DataFrame
    """
    df = pd.read_csv(filename)
    return df


def clean_data(activity_df):
    """
    Takes the raw data from the .csv file and cleans it
    """

    # open and extract the list of irrelevant column names (to remove) from a .csv file
    with open(
        "lalaland/dataset files/columns_to_remove.csv",
        mode="r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.reader(file)
        cols_to_remove = next(reader)

    # strip quotes and whitespace
    cols_to_remove = [col.strip().strip("'") for col in cols_to_remove]

    # remove all the irrelevant columns
    activity_df = activity_df.drop(columns=cols_to_remove)

    # Rename column for less typing
    activity_df.rename(
        columns={"Event Received Timestamp": "Event Timestamp"}, inplace=True
    )

    # Convert timestamps to datetime objects
    activity_df["Event Timestamp"] = pd.to_datetime(
        activity_df["Event Timestamp"],
        format="mixed",  # Tells pandas to infer the correct format for each row (Apple Music is very inconsistent)
        utc=True,
    )

    # Remove duplicate entries (Apple makes one entry for song start, one for song stop, etc.)
    # Add date column to filter songs by (entries for the same song across different days are different 'listens')
    activity_df["date"] = activity_df["Event Timestamp"].dt.date
    activity_df = activity_df.groupby(["Song Name", "date"], group_keys=False).apply(
        calc_song_session
    )

    # Filter out only the songs that are unique events
    activity_df = activity_df[activity_df["new_session"]]

    return activity_df


def merge_library_activity(library_df, activity_df):
    """
    Merge the information from my saved library and my listening activity (can get song, album, and artist name from ym library rather than having to do another API call)
    """

    merged_df = activity_df.merge(
        library_df[["Song Name", "Artist Name", "Album Name"]],
        on="Song Name",
        how="left",
        suffixes=("_play", "_lib"),  # to distinguish where each column came from
    )

    merged_df["Artist Name"] = merged_df["Artist Name_lib"]
    merged_df["Album Name"] = merged_df["Album Name_lib"]

    merged_df = merged_df.drop(
        columns=[
            "Artist Name_play",
            "Artist Name_lib",
            "Album Name_play",
            "Album Name_lib",
        ]
    )

    return merged_df


def write_to_cache_file(cache, df):
    """
    Write the song metadata cache to a file to reduce repeat API calls
    """
    for _, row in df.iterrows():
        song = row["Song Name"]
        artist = row["Artist Name"]
        album = row["Album Name"]

        # Only add if we have valid metadata
        if pd.notna(artist) and pd.notna(album):
            cache[song] = (artist, album)

    with open("lalaland/output files/metadata_cache.json", "w") as f:
        json.dump(cache, f, indent=2)
    return


def search_track(song_name, token):
    """
    Search the Spotiy API for the artist and album names from the song title
    """

    # setup request message
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": quote_plus(song_name), "type": "track", "limit": 1}

    while True:
        # create the request
        response = requests.get(
            "https://api.spotify.com/v1/search",
            headers=headers,
            params=params,
            timeout=10,
        )

        # slow down the api calls so I don't get banned lol
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 2))
            print(f"[RATE LIMITED] Waiting {retry_after} seconds for {song_name}")
            time.sleep(retry_after)
            continue  # retry the request

        # check if no response
        if not response.text.strip():
            print(f"[EMPTY RESPONSE] {song_name}")
            return None, None

        # parse the json
        try:
            data = response.json()
        except Exception:
            print(f"[JSON ERROR] {song_name}")
            print("Response:", response.text[:200])
            return None, None

        # no results
        items = data.get("tracks", {}).get("items", [])
        if not items:
            return None, None

        track = items[0]
        artist = track["artists"][0]["name"]
        album = track["album"]["name"]

        return artist, album


def get_metadata(cache, song_name, token):
    """
    Get the metadata info for a song, first checking if the info exists in the metadata cache, then saving the new info to the cache to reduce future API calls
    """

    if song_name in cache:
        return cache[song_name]

    artist, album = search_track(song_name, token)
    if pd.notna(artist) and pd.notna(album):
        cache[song_name] = (artist, album)

    # Store the new song info in the cache file to avoid future API requests
    with open("lalaland/output files/metadata_cache.json", "w") as outfile:
        json.dump(cache, outfile)

    return artist, album


def calc_song_session(group):
    """
    Group the songs into 'sessions' (an event is recorded every time you pause and play a song, so this removes duplicate entries if they're within 5 min of each other)
    """
    # Sort by timestamp column
    group = group.sort_values("Event Timestamp")

    # For each entry, calculate the time difference from the previous
    group["time_diff"] = group["Event Timestamp"].diff()

    # Create a new session if the difference is greater than 5 min (ie; I listened to the song again)
    group["new_session"] = (group["time_diff"] > pd.Timedelta(minutes=5)) | (
        group["time_diff"].isna()
    )

    return group


# TODO: fetch audio features for a track id
def get_audio_features(track_ids: list):
    """
    Get the following audio features from a list of song IDs:
        - Acousticness
        - Danceability
        - Energy
        - Instrumentalness
        - Song Key
        - Liveliness
        - Loudness
        - Mode (major or minor)
        - Speechiness
        - Tempo
        - Valence
    """

    # Convert tack_ids list into comma-separated-string
    ids = ",".join(track_ids)
    endpoint = f"https://api.reccobeats.com/v1/audio-features?ids={ids}"
    payload = {}
    headers = {"Accept": "application/json"}
    response = requests.request("GET", endpoint, headers=headers, data=payload)
    res_js = response.json()
    print(json.dumps(res_js, indent=2))
    df = pd.DataFrame()
    return df


def spotify_id_search(title, artist, token, cache):
    """
    Returns the Spotify Song ID of a given song and adds it to the metadata cache
    """

    key = f"{title}:{artist}"
    if key in cache:
        return cache[key]

    query = f"track:{title} artist:{artist}"
    url = "https://api.spotify.com/v1/search"
    params = {"q": query, "type": "track", "limit": 1}
    headers = {"Authorization": f"Bearer {token}"}

    while True:
        r = requests.get(url, params=params, headers=headers)

        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 2))
            print(f"[429] Waiting {wait}s for {title}")
            time.sleep(wait)
            continue

        if r.status_code >= 500:
            print(f"[SERVER ERROR] retrying {title}")
            time.sleep(2)
            continue

        r.raise_for_status()
        items = r.json()["tracks"]["items"]
        track_id = items[0]["id"] if items else None
        cache[key] = track_id
        return track_id


def save_spotify_cache(cache, filename="lalaland/output files/spotify_cache.json"):
    """
    Save
    """

    tmp = filename + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, filename)


# TODO: Merge listening history and audio features from spotipy

# TODO: STATISTICS!
