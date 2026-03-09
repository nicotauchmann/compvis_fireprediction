import io
import math
import time
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
# Settings (match your training/data collection)
# ----------------------------
ZOOM = 15
BEARING = 0
TILE_SIZE = 350
RESCALE = 1.0 / 255.0
WILDFIRE_INDEX = 1  # assumes nowildfire=0, wildfire=1

MODEL_PATH = Path("saved_model") / "vgg16_model.keras"


# ----------------------------
# Secrets
# ----------------------------
def get_mapbox_token() -> str:
    # Streamlit Cloud: set via Secrets UI; Local: .streamlit/secrets.toml
    token = st.secrets.get("MAPBOX_ACCESS_TOKEN", "")
    token = (token or "").strip()
    if not token:
        st.error("MAPBOX_ACCESS_TOKEN is missing. Set it in Streamlit Secrets.")
        st.stop()
    return token


MAPBOX_TOKEN = get_mapbox_token()

# Reuse one HTTP session
SESSION = requests.Session()


# ----------------------------
# Cache model + downloads
# ----------------------------
@st.cache_resource
def load_model():
    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data(ttl=3600, show_spinner=False)
def download_url_bytes(url: str) -> bytes:
    r = SESSION.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.content


# ----------------------------
# Mapbox tile fetch (exact URL pattern from your dataset script)
# ----------------------------
def build_mapbox_url(
    lon: float, lat: float, zoom=ZOOM, bearing=BEARING, w=TILE_SIZE, h=TILE_SIZE
) -> str:
    lon = round(float(lon), 6)
    lat = round(float(lat), 6)
    base = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
    coords = f"{lon},{lat}"
    rest = f",{zoom},{bearing}/{w}x{h}?access_token={MAPBOX_TOKEN}&logo=false&attribution=false"
    return base + coords + rest


def fetch_tile(lon: float, lat: float) -> Image.Image:
    url = build_mapbox_url(lon, lat)
    content = download_url_bytes(url)
    img = Image.open(io.BytesIO(content)).convert("RGB")
    if img.size != (TILE_SIZE, TILE_SIZE):
        img = img.resize((TILE_SIZE, TILE_SIZE))
    return img


# ----------------------------
# 5-point grid (center + N/S/E/W) with ~1km spacing
# ----------------------------
def cross5_from_center(lat: float, lon: float, spacing_km: float = 1.0):
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


# ----------------------------
# Preprocess + predict
# ----------------------------
def preprocess_pil(img: Image.Image) -> np.ndarray:
    x = (
        np.asarray(img, dtype=np.float32) * RESCALE
    )  # matches ImageDataGenerator(rescale=1./255.)
    return np.expand_dims(x, axis=0)  # (1, 350, 350, 3)


def predict_wildfire_prob(model, img: Image.Image) -> float:
    x = preprocess_pil(img)
    y = np.array(model.predict(x, verbose=0))

    if y.ndim == 2 and y.shape[1] == 2:
        return float(y[0, WILDFIRE_INDEX])  # softmax 2-class
    if y.ndim == 2 and y.shape[1] == 1:
        return float(y[0, 0])  # sigmoid 1-unit
    raise ValueError(f"Unexpected model output shape: {y.shape}")


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Wildfire map predictor (5-point)", layout="wide")
st.title("Wildfire prediction (5-point cross grid)")

# --- Sidebar inputs in a form (prevents reruns while editing) ---
with st.sidebar:
    st.header("Inputs")
    with st.form("params"):
        lon = st.number_input("Center longitude", value=-64.84903, format="%.6f")
        lat = st.number_input("Center latitude", value=50.33874, format="%.6f")
        spacing_km = st.slider("Spacing (km)", 0.5, 10.0, 1.0, 0.5)
        submitted = st.form_submit_button("Run prediction")

# --- Run and STORE results once when submitted ---
if submitted:
    pts = cross5_from_center(lat, lon, spacing_km=spacing_km)

    rows, imgs = [], []
    with st.spinner("Downloading tiles + predicting..."):
        for name, la, lo in pts:
            img = fetch_tile(lo, la)
            p = predict_wildfire_prob(model, img)
            rows.append({"point": name, "lat": la, "lon": lo, "p_wildfire": p})
            imgs.append((name, la, lo, p, img))

    st.session_state["df"] = pd.DataFrame(rows).sort_values("point")
    st.session_state["imgs"] = imgs
    st.session_state["center"] = (lat, lon)

# --- Display: show stored results if they exist ---
if "df" in st.session_state:
    df = st.session_state["df"]
    imgs = st.session_state["imgs"]
    lat, lon = st.session_state["center"]

    st.subheader("Results")
    st.dataframe(df, use_container_width=True)

    st.subheader("Tiles (sanity check)")
    cols = st.columns(5)
    for i, (name, la, lo, p, img) in enumerate(imgs):
        with cols[i]:
            st.image(
                img,
                caption=f"{name}\n{lo:.6f},{la:.6f}\np={p:.3f}",
                use_container_width=True,
            )

    st.subheader("Map")
    # ... your folium code here ...
else:
    st.info("Set lon/lat (and spacing), then click **Run prediction**.")
