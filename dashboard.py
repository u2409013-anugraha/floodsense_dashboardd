import streamlit as st
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium
import os
import pandas as pd
import json

st.set_page_config(page_title="FloodSense Dashboard", layout="wide", page_icon=":ocean:")

DISTRICTS = {
    "Ernakulam": {
        "flood_map": "flood_map.tif",
        "risk_map": "risk_map.tif",
        "buildings": "high_risk_buildings_light.geojson",
        "center": (9.9816, 76.2999),
        "zoom": 10,
        "ready": True,
    },
    "Thiruvananthapuram": {"ready": False},
    "Kollam": {"ready": False},
    "Pathanamthitta": {"ready": False},
    "Alappuzha": {"ready": False},
    "Kottayam": {"ready": False},
    "Idukki": {"ready": False},
    "Thrissur": {"ready": False},
    "Palakkad": {"ready": False},
    "Malappuram": {"ready": False},
    "Kozhikode": {"ready": False},
    "Wayanad": {"ready": False},
    "Kannur": {"ready": False},
    "Kasaragod": {"ready": False},
}

COMPARISON_IMAGE_PATH = "flood_detection_result.png"

st.markdown("""
<style>
.metric-card {
    background-color: #1a2332;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    border: 1px solid #2d3b52;
}
.metric-label { font-size: 14px; color: #9ca3af; margin-bottom: 8px; }
.metric-value { font-size: 32px; font-weight: 700; color: white; }
.metric-flood { color: #4d94ff; }
.metric-risk { color: #ff5555; }
.metric-building { color: #ffaa33; }
.legend-box {
    background-color: #1a2332;
    border-radius: 8px;
    padding: 12px 20px;
    display: flex;
    gap: 30px;
    align-items: center;
    border: 1px solid #2d3b52;
    margin-bottom: 10px;
}
.legend-item { display: flex; align-items: center; gap: 8px; font-size: 14px; color: #d1d5db; }
.legend-dot { width: 14px; height: 14px; border-radius: 3px; display: inline-block; }
.pending-box {
    background-color: #2d2416;
    border: 1px solid #5c4a1f;
    border-radius: 8px;
    padding: 16px 20px;
    color: #e0c988;
}
</style>
""", unsafe_allow_html=True)


def load_flood_overlay(path, color_rgba):
    with rasterio.open(path) as src:
        data = src.read(1)
        bounds = src.bounds
        src_crs = src.crs
        lon_min, lat_min, lon_max, lat_max = transform_bounds(src_crs, "EPSG:4326", *bounds)
    rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
    mask = data > 0
    rgba[mask] = color_rgba
    return rgba, [[lat_min, lon_min], [lat_max, lon_max]]


def load_risk_overlay(path):
    with rasterio.open(path) as src:
        data = src.read(1)
        bounds = src.bounds
        src_crs = src.crs
        lon_min, lat_min, lon_max, lat_max = transform_bounds(src_crs, "EPSG:4326", *bounds)
    rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.uint8)
    rgba[data == 1] = (60, 200, 100, 130)
    rgba[data == 2] = (255, 165, 0, 150)
    rgba[data == 3] = (255, 40, 40, 170)
    rgba[data == 4] = (255, 40, 40, 170)
    return rgba, [[lat_min, lon_min], [lat_max, lon_max]]


def compute_risk_zone_summary(path):
    with rasterio.open(path) as src:
        data = src.read(1)
        transform = src.transform
        px_w = abs(transform[0])
        px_h = abs(transform[4])
        bounds = src.bounds
        mean_lat = (bounds.top + bounds.bottom) / 2
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * np.cos(np.radians(mean_lat))
        px_area_km2 = (px_w * km_per_deg_lon) * (px_h * km_per_deg_lat)

    rows = []
    for val, label in [(1, "Low"), (2, "Medium"), (3, "High")]:
        if val == 3:
            count = int(np.sum((data == 3) | (data == 4)))
        else:
            count = int(np.sum(data == val))
        area_km2 = count * px_area_km2
        rows.append({"Risk Level": label, "Area (km2)": round(area_km2, 2), "Pixel Count": count})
    return pd.DataFrame(rows)


def safe_file_check(path):
    return path is not None and os.path.exists(path)


@st.cache_data
def count_geojson_features(path):
    with open(path) as f:
        data = json.load(f)
    return len(data.get("features", []))


st.sidebar.title(":ocean: FloodSense")
st.sidebar.caption("Flood detection & risk dashboard using Sentinel-1 SAR + DEM")

selected_district = st.sidebar.selectbox("Select District", list(DISTRICTS.keys()), index=0)
district_info = DISTRICTS[selected_district]

st.sidebar.markdown("---")
show_flood = st.sidebar.checkbox("Show Flood Extent", value=True)
show_risk = st.sidebar.checkbox("Show Risk Zones", value=True)
show_buildings = st.sidebar.checkbox("Show High-Risk Buildings", value=False)
if show_buildings:
    st.sidebar.caption(":warning: Rendering 50k+ building shapes can be slow.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Team:** Anugraha, Bettina, Ganga, Lanet")
st.sidebar.markdown("**Guide:** Ms. Veena Rani")

st.title("FloodSense - Inundation Mapping & Prediction Dashboard")
st.caption(f"{selected_district} District, Kerala | Sentinel-1 SAR flood detection (Aug 2018 flood event)")

if not district_info.get("ready", False):
    st.markdown(f"""
    <div class="pending-box">
    <b>Data not yet processed for {selected_district}.</b><br>
    This district is part of FloodSense's planned coverage, but Sentinel-1 SAR processing,
    flood detection, and risk analysis have only been completed for <b>Ernakulam</b> so far.
    Select "Ernakulam" from the sidebar to view a fully processed result.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

FLOOD_MAP_PATH = district_info["flood_map"]
RISK_MAP_PATH = district_info["risk_map"]
BUILDINGS_PATH = district_info.get("buildings")

flooded_pct = None
risk_pct = None
building_count = None
if safe_file_check(FLOOD_MAP_PATH):
    with rasterio.open(FLOOD_MAP_PATH) as src:
        d = src.read(1)
    flooded_pct = 100 * (d > 0).sum() / d.size
if safe_file_check(RISK_MAP_PATH):
    with rasterio.open(RISK_MAP_PATH) as src:
        r = src.read(1)
    risk_pct = 100 * (r >= 3).sum() / r.size
if safe_file_check(BUILDINGS_PATH):
    building_count = count_geojson_features(BUILDINGS_PATH)

col1, col2, col3 = st.columns(3)
with col1:
    val = f"{flooded_pct:.2f}%" if flooded_pct is not None else "N/A"
    st.markdown(f"""<div class="metric-card"><div class="metric-label">FLOODED AREA</div>
    <div class="metric-value metric-flood">{val}</div></div>""", unsafe_allow_html=True)
with col2:
    val = f"{risk_pct:.2f}%" if risk_pct is not None else "N/A"
    st.markdown(f"""<div class="metric-card"><div class="metric-label">HIGH RISK AREA</div>
    <div class="metric-value metric-risk">{val}</div></div>""", unsafe_allow_html=True)
with col3:
    val = f"{building_count:,}" if building_count is not None else "N/A"
    st.markdown(f"""<div class="metric-card"><div class="metric-label">BUILDINGS AT HIGH RISK</div>
    <div class="metric-value metric-building">{val}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="legend-box">
<div class="legend-item"><span class="legend-dot" style="background-color:#4d94ff;"></span>Flooded Area</div>
<div class="legend-item"><span class="legend-dot" style="background-color:#3cc864;"></span>Low Risk</div>
<div class="legend-item"><span class="legend-dot" style="background-color:#ffa500;"></span>Medium Risk</div>
<div class="legend-item"><span class="legend-dot" style="background-color:#ff2828;"></span>High Risk</div>
<div class="legend-item"><span class="legend-dot" style="background-color:#ffaa33;"></span>At-Risk Building</div>
</div>
""", unsafe_allow_html=True)

map_col, table_col = st.columns([2, 1])

with map_col:
    lat, lon = district_info["center"]
    m = folium.Map(location=[lat, lon], zoom_start=district_info["zoom"], tiles="OpenStreetMap")

    if show_risk and safe_file_check(RISK_MAP_PATH):
        rgba, bounds = load_risk_overlay(RISK_MAP_PATH)
        ImageOverlay(rgba, bounds=bounds, name="Risk Zones", opacity=0.65).add_to(m)

    if show_flood and safe_file_check(FLOOD_MAP_PATH):
        rgba, bounds = load_flood_overlay(FLOOD_MAP_PATH, color_rgba=(77, 148, 255, 220))
        ImageOverlay(rgba, bounds=bounds, name="Flood Extent", opacity=0.9).add_to(m)

    if show_buildings and safe_file_check(BUILDINGS_PATH):
        with open(BUILDINGS_PATH) as f:
            buildings_geojson = json.load(f)
        folium.GeoJson(
            buildings_geojson,
            name="High-Risk Buildings",
            style_function=lambda x: {"fillColor": "#ffaa33", "color": "#ffaa33", "weight": 0.5, "fillOpacity": 0.6},
        ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=800, height=550)

with table_col:
    st.markdown("**Risk Zone Summary**")
    if safe_file_check(RISK_MAP_PATH):
        summary_df = compute_risk_zone_summary(RISK_MAP_PATH)

        def highlight_risk(row):
            colors = {"High": "background-color: #4d1a1a; color: #ff8080",
                      "Medium": "background-color: #4d3a1a; color: #ffcc80",
                      "Low": "background-color: #1a4d2a; color: #80ff9f"}
            return [colors.get(row["Risk Level"], "")] * len(row)

        st.dataframe(
            summary_df.style.apply(highlight_risk, axis=1),
            hide_index=True,
            use_container_width=True
        )
        st.caption("Area estimated from raster pixel counts (approx. km2 based on WGS84 pixel size).")
    else:
        st.info("risk_map.tif not found")

    if building_count is not None:
        st.markdown("**Infrastructure at Risk**")
        st.markdown(f"""<div class="metric-card" style="padding:14px;">
        <div class="metric-label">Buildings (OSM footprints)</div>
        <div class="metric-value metric-building" style="font-size:24px;">{building_count:,}</div>
        <div style="font-size:12px; color:#9ca3af;">overlapping High-Risk flood zones</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

if safe_file_check(COMPARISON_IMAGE_PATH):
    st.subheader("Before vs. During Flood - SAR Comparison")
    st.image(COMPARISON_IMAGE_PATH, use_container_width=True)
    st.caption("Left: before-flood SAR backscatter | Middle: during-flood SAR backscatter | Right: detected new flood extent")

st.markdown("---")
st.caption("Data: Sentinel-1 SAR (Copernicus), SRTM DEM, OpenStreetMap buildings. Prototype dashboard - FloodSense mini project.")
