# streamlit_compvis.py
# Wildfire prediction demo (5-point cross grid) with:
# 1) fixed spacing = 2 km
# 2) user selects the center by clicking on the map (no manual lat/lon input)
# 3) custom "Loading model…" message instead of "Running load_model_cached()"
# 4) one-click selection: the marker updates immediately after a single click
# UI changes:
# - map starts centered on Québec (province) and zoomed out to show most/all of Québec
# - sidebar no longer shows "Fixed spacing: 2.0 km"

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

import folium
from folium import CircleMarker, Marker
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

# Fixed spacing
SPACING_KM = 2.0

WILDFIRE_INDEX = 1

MODEL_PATH = Path("saved_model") / "vgg16_model.keras"

# Default view: Québec (province) — centered and zoomed out further
DEFAULT_CENTER = (52.0, -71.0)  # lat, lon (approx center of Québec)
DEFAULT_ZOOM_PICK = 5  # zoomed out to show most/all of Québec
DEFAULT_ZOOM_RESULT = 6  # a bit closer for the 5-point result view


# ----------------------------
# Helpers
# ----------------------------
def get_mapbox_token() -> str:
    token = st.secrets.get("MAPBOX_ACCESS_TOKEN", "")
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
    lon = round(float(lon), 6)
    lat = round(float(lat), 6)
    base = f"https://api.mapbox.com/styles/v1/{STYLE_USER}/{STYLE_ID}/static/"
    coords = f"{lon},{lat}"
    rest = f",{ZOOM},{BEARING}/{TILE_SIZE}x{TILE_SIZE}?access_token={token}&logo=false&attribution=false"
    return base + coords + rest


def cross5_from_center(lat: float, lon: float):
    """Exactly 5 points: center + N/S/E/W at fixed 2 km spacing."""
    dlat = SPACING_KM / 110.574
    dlon = SPACING_KM / (111.320 * math.cos(math.radians(lat)))

    pts = [
        ("center", lat, lon),
        ("north", lat + dlat, lon),
        ("south", lat - dlat, lon),
        ("east", lat, lon + dlon),
        ("west", lat, lon - dlon),
    ]
    return [(name, round(la, 6), round(lo, 6)) for name, la, lo in pts]


def preprocess_pil(img: Image.Image) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32) * RESCALE
    return np.expand_dims(x, axis=0)  # (1, 350, 350, 3)


def predict_wildfire_prob(model, img: Image.Image) -> float:
    x = preprocess_pil(img)
    y = np.array(model.predict(x, verbose=0))

    if y.ndim == 2 and y.shape[1] == 2:
        return float(y[0, WILDFIRE_INDEX])  # softmax 2-class
    if y.ndim == 2 and y.shape[1] == 1:
        return float(y[0, 0])  # sigmoid 1-unit

    raise ValueError(f"Unexpected model output shape: {y.shape}")


def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# ----------------------------
# Cached resources
# ----------------------------
SESSION = requests.Session()


@st.cache_resource(show_spinner=False)
def load_model_cached():
    import tensorflow as tf

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")
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

try:
    with st.spinner("Loading model…"):
        model = load_model_cached()
except Exception as e:
    st.error(f"Failed to load model from {MODEL_PATH}.\n\n{e}")
    st.stop()

# Keep a selected center in session_state
if "selected_center" not in st.session_state:
    st.session_state["selected_center"] = DEFAULT_CENTER

sel_lat, sel_lon = st.session_state["selected_center"]

with st.sidebar:
    st.header("Pick a location")
    st.write("Click once on the map to set the center point.")
    st.write(f"**Selected:** lat `{sel_lat:.6f}`, lon `{sel_lon:.6f}`")
    run = st.button("Run prediction", type="primary")
    clear = st.button("Clear results")

if clear:
    for k in ["df", "imgs"]:
        if k in st.session_state:
            del st.session_state[k]

# --- Map for picking a point ---
st.subheader(
    "Click on the map to choose the center point (drag to move, click to select)"
)

# If the user never selected anything yet, start zoomed out on Québec.
# Otherwise keep the selected marker but still start zoomed out.
pick_map = folium.Map(
    location=[sel_lat, sel_lon], zoom_start=DEFAULT_ZOOM_PICK, tiles="OpenStreetMap"
)
Marker(location=[sel_lat, sel_lon], popup="Selected center").add_to(pick_map)

picked = st_folium(pick_map, width=900, height=520, key="pick_map")

if picked and picked.get("last_clicked"):
    new_lat = float(picked["last_clicked"]["lat"])
    new_lon = float(picked["last_clicked"]["lng"])
    new_center = (round(new_lat, 6), round(new_lon, 6))

    if new_center != st.session_state["selected_center"]:
        st.session_state["selected_center"] = new_center
        rerun_app()

sel_lat, sel_lon = st.session_state["selected_center"]

# --- Run prediction and STORE results ---
if run:
    pts = cross5_from_center(sel_lat, sel_lon)

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

# --- Display stored results ---
if "df" not in st.session_state:
    st.info("Click a point on the map, then press **Run prediction**.")
    st.stop()

df = st.session_state["df"].copy()
imgs = st.session_state.get("imgs", [])

st.subheader("Results")
st.dataframe(df, use_container_width=True)

st.download_button(
    "Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="wildfire_predictions_5points.csv",
    mime="text/csv",
)

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

st.subheader("Prediction map (5 points)")
m = folium.Map(
    location=[sel_lat, sel_lon], zoom_start=DEFAULT_ZOOM_RESULT, tiles="OpenStreetMap"
)

for _, row in df.iterrows():
    la = float(row["lat"])
    lo = float(row["lon"])
    p = row.get("p_wildfire", None)

    if p is None or (isinstance(p, float) and np.isnan(p)):
        color = "gray"
        popup = f"{row['point']}: error"
    else:
        p = float(p)
        color = "red" if p >= 0.8 else "blue"
        popup = f"{row['point']}: p={p:.3f}"

    CircleMarker(
        location=(la, lo),
        radius=10,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=popup,
    ).add_to(m)

st_folium(m, width=900, height=520, key="result_map")
