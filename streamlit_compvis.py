# streamlit_compvis.py
# 5-point wildfire prediction demo (center + N/S/E/W) using Mapbox satellite tiles + your trained Keras model

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

import folium
from folium import CircleMarker
from streamlit_folium import st_folium


# ----------------------------
# CONFIG (match your dataset + training)
# ----------------------------
STYLE_USER = "mapbox"
STYLE_ID = "satellite-v9"

ZOOM = 15
BEARING = 0
TILE_SIZE = 350
RESCALE = 1.0 / 255.0

# If your model is 2-class softmax and class_indices was {'nowildfire': 0, 'wildfire': 1}
WILDFIRE_INDEX = 1

# Path to your .keras model file inside the repo
MODEL_PATH = Path("saved_model") / "vgg16_model.keras"


# ----------------------------
# Helpers
# ----------------------------
def get_mapbox_token() -> str:
    # Prefer Streamlit secrets, fall back to environment
    token = st.secrets.get("MAPBOX_ACCESS_TOKEN", "")
    if not token:
        token = st.session_state.get("_env_token", "")  # internal, just in case
    if not token:
        token = st.experimental_get_query_params().get("MAPBOX_ACCESS_TOKEN", [""])[0]
    if not token:
        token = ""  # keep explicit

    # fallback to env var if present
    if not token:
        import os

        token = os.getenv("MAPBOX_ACCESS_TOKEN", "")

    token = (token or "").strip()
    if not token:
        st.error(
            "MAPBOX_ACCESS_TOKEN is missing.\n\n"
            "On Streamlit Cloud: App → Settings → Secrets → add:\n"
            'MAPBOX_ACCESS_TOKEN="pk.XXXX..."\n'
        )
        st.stop()
    return token


def build_mapbox_url(lon: float, lat: float, token: str) -> str:
    # match dataset script: round to 6 decimals, lon,lat order, zoom=15, bearing=0, size=350x350, logo/attrib off
    lon = round(float(lon), 6)
    lat = round(float(lat), 6)
    base = f"https://api.mapbox.com/styles/v1/{STYLE_USER}/{STYLE_ID}/static/"
    coords = f"{lon},{lat}"
    rest = f",{ZOOM},{BEARING}/{TILE_SIZE}x{TILE_SIZE}?access_token={token}&logo=false&attribution=false"
    return base + coords + rest


def cross5_from_center(lat: float, lon: float, spacing_km: float = 1.0):
    """
    Returns exactly 5 (name, lat, lon) points:
      center, north, south, east, west
    with ~spacing_km distance from center.
    """
    dlat = spacing_km / 110.574
    dlon = spacing_km / (111.320 * math.cos(math.radians(lat)))

    pts = [
        ("center", lat, lon),
        ("north", lat + dlat, lon),
        ("south", lat - dlat, lon),
        ("east", lat, lon + dlon),
        ("west", lat, lon - dlon),
    ]
    # round like dataset
    return [(name, round(la, 6), round(lo, 6)) for name, la, lo in pts]


def preprocess_pil(img: Image.Image) -> np.ndarray:
    # match ImageDataGenerator(dtype='float32', rescale=1./255.) — no resizing
    x = np.asarray(img, dtype=np.float32) * RESCALE
    return np.expand_dims(x, axis=0)  # (1, 350, 350, 3)


def predict_wildfire_prob(model, img: Image.Image) -> float:
    x = preprocess_pil(img)
    y = np.array(model.predict(x, verbose=0))

    if y.ndim == 2 and y.shape[1] == 2:  # softmax 2-class
        return float(y[0, WILDFIRE_INDEX])
    if y.ndim == 2 and y.shape[1] == 1:  # sigmoid 1-unit
        return float(y[0, 0])

    raise ValueError(f"Unexpected model output shape: {y.shape}")


# ----------------------------
# Cached resources
# ----------------------------
SESSION = requests.Session()


@st.cache_resource
def load_model_cached():
    import tensorflow as tf

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")

    # loads a .keras model file
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data(ttl=3600, show_spinner=False)
def download_bytes(url: str) -> bytes:
    r = SESSION.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.content


def fetch_tile(lon: float, lat: float, token: str) -> Image.Image:
    url = build_mapbox_url(lon, lat, token)
    content = download_bytes(url)
    img = Image.open(io.BytesIO(content)).convert("RGB")
    if img.size != (TILE_SIZE, TILE_SIZE):
        img = img.resize((TILE_SIZE, TILE_SIZE))
    return img


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Wildfire predictor (5-point)", layout="wide")
st.title("Wildfire prediction (5-point cross grid)")

token = get_mapbox_token()

# Load model once (cached)
try:
    model = load_model_cached()
except Exception as e:
    st.error(f"Failed to load model from {MODEL_PATH}.\n\n{e}")
    st.stop()

# Sidebar form (prevents rerun spam while editing)
with st.sidebar:
    st.header("Inputs")
    with st.form("params"):
        # explicit lon/lat inputs
        lon = st.number_input("Center longitude", value=-64.84903, format="%.6f")
        lat = st.number_input("Center latitude", value=50.33874, format="%.6f")
        spacing_km = st.slider(
            "Spacing (km)", min_value=0.5, max_value=10.0, value=1.0, step=0.5
        )
        submitted = st.form_submit_button("Run prediction")

# Run + store results in session_state so they don't disappear after reruns
if submitted:
    pts = cross5_from_center(lat, lon, spacing_km=spacing_km)

    rows = []
    imgs = []

    with st.spinner("Downloading tiles + predicting..."):
        for name, la, lo in pts:
            try:
                img = fetch_tile(lo, la, token)
                p = predict_wildfire_prob(model, img)
                rows.append({"point": name, "lat": la, "lon": lo, "p_wildfire": p})
                imgs.append((name, la, lo, p, img))
            except Exception as e:
                rows.append(
                    {
                        "point": name,
                        "lat": la,
                        "lon": lo,
                        "p_wildfire": None,
                        "error": str(e),
                    }
                )

    st.session_state["df"] = pd.DataFrame(rows)
    st.session_state["imgs"] = imgs
    st.session_state["center"] = (lat, lon)
    st.session_state["spacing_km"] = spacing_km

# Display last results if available
if "df" not in st.session_state:
    st.info("Set lon/lat (and spacing), then click **Run prediction**.")
    st.stop()

df = st.session_state["df"].copy()
imgs = st.session_state.get("imgs", [])
center_lat, center_lon = st.session_state.get("center", (lat, lon))

st.subheader("Results")
st.dataframe(df, use_container_width=True)

st.download_button(
    "Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="wildfire_predictions_5points.csv",
    mime="text/csv",
)

# Tiles
st.subheader("Tiles (sanity check)")
if imgs:
    cols = st.columns(5)
    for i, (name, la, lo, p, img) in enumerate(imgs):
        with cols[i]:
            cap = f"{name}\n{lo:.6f},{la:.6f}\n"
            cap += f"p={p:.3f}" if p is not None else "p=None"
            st.image(img, caption=cap, use_container_width=True)
else:
    st.warning(
        "No images were produced (all fetch/predict steps failed). Check errors in the table above."
    )

# Map
st.subheader("Map")
m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

# Draw points
for _, row in df.iterrows():
    la = float(row["lat"])
    lo = float(row["lon"])
    p = row.get("p_wildfire", None)

    if p is None or (isinstance(p, float) and np.isnan(p)):
        color = "gray"
        popup = f"{row['point']}: error"
    else:
        p = float(p)
        color = "red" if p >= 0.5 else "blue"
        popup = f"{row['point']}: p={p:.3f}"

    CircleMarker(
        location=(la, lo),
        radius=10,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=popup,
    ).add_to(m)

st_folium(m, width=900, height=520)
