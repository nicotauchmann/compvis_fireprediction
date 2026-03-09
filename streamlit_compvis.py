# streamlit_compvis.py
# Wildfire prediction demo (5-point cross grid) with:
# - fixed spacing = 3 km
# - user selects center by clicking on the map (no manual lat/lon input)
# - custom "Loading model…" message
# - one-click selection with immediate rerun
# - map starts centered on Québec (province) and zoomed out
# - sidebar does NOT show fixed spacing line
# - NEW: 3-star (🔥) rating at top of results based on how many of 5 points exceed threshold

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

SPACING_KM = 3.0

WILDFIRE_INDEX = 1
MODEL_PATH = Path("saved_model") / "vgg16_model.keras"

# UI defaults: Québec (province)
DEFAULT_CENTER = (52.0, -71.0)  # lat, lon (approx center of Québec)
DEFAULT_ZOOM_PICK = 5
DEFAULT_ZOOM_RESULT = 6

# Threshold for "likely to host a forest fire" (match your map coloring logic)
LIKELY_THRESHOLD = 0.8


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
    """Exactly 5 points: center + N/S/E/W at fixed spacing."""
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


def compute_fire_rating(df: pd.DataFrame, threshold: float = LIKELY_THRESHOLD):
    """
    df has p_wildfire column with floats or None.
    Count how many of the 5 points are "likely" (>= threshold), then map to 0-3 stars.
    """
    probs = pd.to_numeric(df.get("p_wildfire"), errors="coerce")
    likely_count = int((probs >= threshold).sum())

    if likely_count == 0:
        stars = 0
        msg = "A fire is unlikely in this environment."
    elif likely_count == 1:
        stars = 1
        msg = "The fire potential of this environment is low."
    elif likely_count in (2, 3):
        stars = 2
        msg = "The fire potential of this environment is moderate. Check local safety precautions."
    else:  # 4 or 5
        stars = 3
        msg = "The fire potential of this environment is high. Check local safety precautions."

    emoji = "🔥" * stars if stars > 0 else "—"
    return likely_count, stars, emoji, msg


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
st.set_page_config(page_title="Infrastructural Wildfire Likelihood", layout="wide")
st.title("Infrastructural Wildfire Likelihood")

token = get_mapbox_token()

try:
    with st.spinner("Loading model…"):
        model = load_model_cached()
except Exception as e:
    st.error(f"Failed to load model from {MODEL_PATH}.\n\n{e}")
    st.stop()

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

st.subheader(
    "Click on the map to choose the center point (drag to move, click to select)"
)

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

    with st.spinner("Downloading satellite images + predicting..."):
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

# --- NEW: Rating block at top of results ---
likely_count, stars, emoji, msg = compute_fire_rating(df, threshold=LIKELY_THRESHOLD)

st.subheader("Fire potential rating")
st.markdown(f"### {emoji}")
st.write(
    f"**Likely fire tiles:** {likely_count} / 5 (threshold ≥ {LIKELY_THRESHOLD:.1f})"
)
st.info(msg)

st.subheader("Results")
st.dataframe(df, use_container_width=True)

st.download_button(
    "Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="wildfire_predictions_5points.csv",
    mime="text/csv",
)

st.subheader("Satellite Images")
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
        color = "red" if p >= LIKELY_THRESHOLD else "blue"
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
