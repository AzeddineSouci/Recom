import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.cluster import KMeans

client_id = st.secrets["SPOTIPY_CLIENT_ID"]
client_secret = st.secrets["SPOTIPY_CLIENT_SECRET"]

spotify = Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=client_id, client_secret=client_secret
    )
)

features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_sec",
    "popularity",
]


@st.cache_data
def load_data():
    df = pd.read_csv("SpotifyFeatures.csv")
    df["duration_sec"] = df["duration_ms"] / 1000
    return df.dropna().reset_index(drop=True)

@st.cache_resource
def create_model(df, n_clusters=12):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(df_scaled)
    return scaler, kmeans, df


def get_recommendations(song_name, df, scaler, kmeans, n_songs=5):
    song_row = df[df["track_name"].str.lower() == song_name.lower()]
    if song_row.empty:
        return pd.DataFrame()

    cluster_label = song_row["cluster"].values[0]
    same_cluster_songs = df[df["cluster"] == cluster_label]

    # Supprimer la chanson originale
    same_cluster_songs = same_cluster_songs[
        same_cluster_songs["track_name"].str.lower() != song_name.lower()
    ]

    return same_cluster_songs.sample(n=min(n_songs, len(same_cluster_songs)))


def get_spotify_info(track_name, artist_name):
    try:
        results = spotify.search(
            q=f"track:{track_name} artist:{artist_name}", type="track", limit=1
        )
        if results["tracks"]["items"]:
            track = results["tracks"]["items"][0]
            return {
                "image": (
                    track["album"]["images"][0]["url"]
                    if track["album"]["images"]
                    else None
                ),
                "preview": track.get("preview_url"),
                "link": track.get("external_urls", {}).get("spotify"),
            }
    except:
        pass
    return None


st.set_page_config(page_title="Recommandation Musicale", page_icon="🎵", layout="wide")
st.title("🎵 Recommandation Musicale Spotify")

df = load_data()
scaler, kmeans, df = create_model(df)

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Recherche")
    song_input = st.text_input("Nom de la chanson:")
    n_recs = st.slider("Nombre de recommandations:", 3, 12, 6)
    search_btn = st.button("Rechercher")

with col2:
    if search_btn and song_input:
        if song_input not in df["track_name"].values:
            st.error("Chanson non trouvée!")
        else:
            selected = df[df["track_name"] == song_input].iloc[0]
            st.subheader(f"Chanson: {selected['track_name']}")
            st.write(f"**Artiste:** {selected['artist_name']}")
            st.write(f"**Genre:** {selected['genre']}")
            st.write(f"**Popularité:** {selected['popularity']}/100")

            spotify_info = get_spotify_info(
                selected["track_name"], selected["artist_name"]
            )
            if spotify_info:
                if spotify_info["image"]:
                    st.image(spotify_info["image"], width=200)
                if spotify_info["link"]:
                    st.markdown(f"[🎵 Écouter sur Spotify]({spotify_info['link']})")
                if spotify_info["preview"]:
                    st.audio(spotify_info["preview"])

            recommendations = get_recommendations(
                song_input, df, scaler, kmeans, n_recs
            )

            if not recommendations.empty:
                st.subheader("Recommandations:")

                for idx, song in recommendations.iterrows():
                    with st.expander(f"{song['track_name']} - {song['artist_name']}"):
                        col_a, col_b = st.columns([1, 2])

                        with col_a:
                            st.write(f"**Genre:** {song['genre']}")
                            st.write(f"**Popularité:** {song['popularity']}/100")
                            st.write(f"**Tempo:** {song['tempo']:.0f} BPM")

                            rec_spotify = get_spotify_info(
                                song["track_name"], song["artist_name"]
                            )
                            if rec_spotify and rec_spotify["link"]:
                                st.markdown(f"[🎵 Spotify]({rec_spotify['link']})")

                        with col_b:
                            if rec_spotify and rec_spotify["image"]:
                                st.image(rec_spotify["image"], width=180)

                            if rec_spotify and rec_spotify["preview"]:
                                st.audio(rec_spotify["preview"])
            else:
                st.warning("Aucune recommandation trouvée")

with st.sidebar:
    st.header("💡 Aide")
    st.write("**Chansons populaires à tester:**")
    st.write("• Not Afraid")
    st.write("• Rap God")
    st.write("• Bohemian Rhapsody")
    st.write("• Under the Bridge")
    st.write("• Don't Stop Believin'")
    st.write("• All Out Life")
