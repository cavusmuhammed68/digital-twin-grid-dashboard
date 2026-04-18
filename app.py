import json
import math
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import urllib3
import folium
from folium.plugins import HeatMap
import plotly.graph_objects as go
import networkx as nx
from streamlit_plotly_events import plotly_events

# =========================================================
# PAGE + SESSION
# =========================================================
st.set_page_config(
    page_title="North East & Yorkshire Grid Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "credo_selected_node" not in st.session_state:
    st.session_state["credo_selected_node"] = None

if "credo_last_step" not in st.session_state:
    st.session_state["credo_last_step"] = 0

st.markdown(
    """
<style>
body {
    background-color: #0e1117;
}

.block-container {
    padding-top: 1rem;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 12px;
    border-radius: 12px;
    backdrop-filter: blur(8px);
}

.stTabs [role="tab"] {
    font-size: 16px;
    padding: 10px;
}

.stTabs [aria-selected="true"] {
    background-color: rgba(255,255,255,0.1);
    border-radius: 10px;
}

.small-note {
    font-size: 0.85rem;
    opacity: 0.85;
}
</style>
""",
    unsafe_allow_html=True,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

session = requests.Session()
session.verify = False
session.headers.update({
    "User-Agent": "north-east-yorkshire-digital-twin/4.0"
})

components.html(
    """
    <script>
        setTimeout(function() {
            window.location.reload();
        }, 30000);
    </script>
    """,
    height=0,
)

# =========================================================
# CONSTANTS
# =========================================================
NPG_DATASET_URL = (
    "https://northernpowergrid.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/live-power-cuts-data/records"
)

WEATHER_CURRENT_VARS = ",".join([
    "temperature_2m",
    "apparent_temperature",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "relative_humidity_2m",
    "precipitation",
    "is_day",
])

WEATHER_HOURLY_VARS = ",".join([
    "temperature_2m",
    "apparent_temperature",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "cloud_cover",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "relative_humidity_2m",
    "precipitation",
    "is_day",
])

AIR_CURRENT_VARS = ",".join([
    "european_aqi",
    "pm10",
    "pm2_5",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
    "carbon_monoxide",
    "aerosol_optical_depth",
    "dust",
    "uv_index",
])

AIR_HOURLY_VARS = AIR_CURRENT_VARS

SATELLITE_LAYERS = {
    "MODIS Terra True Colour": {
        "tile_url": "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/{date}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg"
    },
    "VIIRS NOAA-20 True Colour": {
        "tile_url": "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/VIIRS_NOAA20_CorrectedReflectance_TrueColor/default/{date}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg"
    },
    "Black Marble Night Lights": {
        "tile_url": "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlackMarble_2016/default/{date}/GoogleMapsCompatible_Level8/{z}/{y}/{x}.png"
    },
}

REGIONS = {
    "North East": {
        "center": {"lat": 54.85, "lon": -1.65, "zoom": 7.5},
        "bbox": [-3.35, 54.10, -0.60, 55.95],
        "polygon": [
            [-3.20, 55.80],
            [-2.50, 55.92],
            [-1.10, 55.85],
            [-0.80, 55.35],
            [-0.75, 54.70],
            [-1.20, 54.30],
            [-2.20, 54.15],
            [-3.05, 54.35],
            [-3.20, 55.00],
            [-3.20, 55.80],
        ],
        "places": {
            "Newcastle": (54.9783, -1.6178),
            "Sunderland": (54.9069, -1.3838),
            "Durham": (54.7761, -1.5733),
            "Middlesbrough": (54.5742, -1.2350),
            "Darlington": (54.5236, -1.5595),
            "Hexham": (54.9730, -2.1010),
        },
        "search_tokens": [
            "newcastle", "newcastle upon tyne", "northumberland", "sunderland",
            "durham", "county durham", "teesside", "middlesbrough", "gateshead",
            "south tyneside", "north tyneside", "darlington", "hartlepool",
            "stockton", "redcar", "tyne and wear",
        ],
    },
    "Yorkshire": {
        "center": {"lat": 53.95, "lon": -1.30, "zoom": 7.2},
        "bbox": [-2.90, 53.20, -0.10, 54.75],
        "polygon": [
            [-2.80, 54.55],
            [-2.00, 54.70],
            [-0.50, 54.62],
            [-0.20, 54.10],
            [-0.18, 53.45],
            [-0.95, 53.25],
            [-1.90, 53.28],
            [-2.70, 53.55],
            [-2.80, 54.10],
            [-2.80, 54.55],
        ],
        "places": {
            "Leeds": (53.8008, -1.5491),
            "Sheffield": (53.3811, -1.4701),
            "York": (53.9600, -1.0873),
            "Hull": (53.7676, -0.3274),
            "Bradford": (53.7950, -1.7594),
            "Doncaster": (53.5228, -1.1285),
        },
        "search_tokens": [
            "yorkshire", "leeds", "sheffield", "york", "hull", "bradford",
            "wakefield", "rotherham", "doncaster", "barnsley", "huddersfield",
            "harrogate", "scarborough", "halifax", "east riding",
        ],
    },
}

# =========================================================
# BASIC HELPERS
# =========================================================
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value) -> Optional[float]:
    try:
        if isinstance(value, pd.Series):
            if len(value) == 0:
                return None
            value = value.iloc[0]
        elif isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return None
            value = value[0]

        if value is None or value == "":
            return None

        if pd.isna(value):
            return None

        return float(value)
    except Exception:
        return None


def force_scalar(x):
    if isinstance(x, pd.Series):
        if len(x) == 0:
            return None
        return safe_float(x.iloc[0])
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return None
        return safe_float(x[0])
    return safe_float(x) if isinstance(x, (int, float, np.integer, np.floating, str, type(None))) else x


def ensure_scalar_dict(row):
    if isinstance(row, pd.Series):
        row = row.to_dict()
    return {
        k: force_scalar(v)
        if isinstance(v, (pd.Series, list, tuple, np.ndarray, int, float, np.integer, np.floating, str, type(None)))
        else v
        for k, v in row.items()
    }


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def point_in_bbox(lat: float, lon: float, bbox: List[float]) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def normalise_to_unit(x: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp((x - low) / (high - low), 0.0, 1.0)


def get_risk_label(x):
    x = safe_float(x)
    if x is None:
        return "Unknown"
    if x >= 75:
        return "Severe"
    elif x >= 55:
        return "High"
    elif x >= 35:
        return "Moderate"
    return "Low"


def get_satellite_tile_url(satellite_name: str) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return SATELLITE_LAYERS[satellite_name]["tile_url"].replace("{date}", today)


def solar_interpretation(current: Dict) -> Tuple[str, str]:
    solar = current.get("shortwave_radiation")
    is_day = current.get("is_day")
    cloud = current.get("cloud_cover")

    if solar is None:
        return "—", "Solar radiation not available."

    if is_day == 0:
        return f"{solar} W/m²", "It is currently night-time at this location, so near-zero solar radiation is physically expected."

    if cloud is not None and solar == 0:
        return f"{solar} W/m²", "Daytime zero solar radiation is unusual and may reflect transient API or observation conditions."

    return f"{solar} W/m²", f"Daytime solar conditions. Cloud cover is {cloud}%."


def get_place_from_node(node_name: str) -> str:
    for suffix in ["_power", "_water", "_telecom"]:
        if node_name.endswith(suffix):
            return node_name.replace(suffix, "")
    return node_name


def node_icon(node_type: str) -> str:
    return {
        "power": "⚡",
        "water": "💧",
        "telecom": "📡",
        "city": "🏙️",
    }.get(node_type, "•")


def node_marker_symbol(node_type: str) -> str:
    return {
        "power": "square",
        "water": "circle",
        "telecom": "diamond",
        "city": "triangle-up",
    }.get(node_type, "circle")


def node_display_colour(node_type: str, state: int) -> str:
    """
    Colour encodes infrastructure TYPE only.
    State (failure/stress) should be shown via border, glow, or opacity.
    """
    return {
        "power": "#ffd60a",     # yellow
        "water": "#4ea5ff",     # blue
        "telecom": "#b57cff",   # purple
        "city": "#34c759",      # green
    }.get(node_type, "#cccccc")

# =========================================================
# NETWORK / CREDO HELPERS
# =========================================================
def build_credo_layout(place_names: List[str]) -> Dict[str, Tuple[float, float]]:
    pos = {}
    n = max(len(place_names), 1)
    x_positions = np.linspace(-1.0, 1.0, n)

    y_levels = {
        "city": 0.95,
        "power": 0.30,
        "telecom": -0.05,
        "water": -0.40,
    }

    for i, place in enumerate(place_names):
        x = float(x_positions[i])
        pos[place] = (x, y_levels["city"])
        pos[f"{place}_power"] = (x - 0.08, y_levels["power"])
        pos[f"{place}_telecom"] = (x + 0.08, y_levels["telecom"])
        pos[f"{place}_water"] = (x, y_levels["water"])

    return pos


def infer_flood_depth_for_place(row: Dict, scenario_name: str = "baseline") -> float:
    rain = safe_float(row.get("precipitation")) or 0.0
    outages = safe_float(row.get("nearby_outages_25km")) or 0.0
    risk = safe_float(row.get("risk_score")) or 0.0
    cloud = safe_float(row.get("cloud_cover")) or 0.0

    scenario_multiplier = {
        "baseline": 1.00,
        "storm_cascade": 1.25,
        "flood_infrastructure": 1.80,
        "heatwave_peak": 0.30,
        "pollution_event": 0.20,
        "compound_extreme": 2.00,
    }.get(scenario_name, 1.00)

    depth = (
        0.035 * rain +
        0.018 * outages +
        0.002 * risk +
        0.001 * cloud
    ) * scenario_multiplier

    return round(clamp(depth, 0.0, 2.5), 3)


def direct_failure_probability(asset_type: str, row: Dict, flood_depth: float, t: int) -> float:
    wind = safe_float(row.get("wind_speed_10m")) or 0.0
    rain = safe_float(row.get("precipitation")) or 0.0
    outages = safe_float(row.get("nearby_outages_25km")) or 0.0
    aqi = safe_float(row.get("european_aqi")) or 0.0

    base = {
        "power": 0.10,
        "water": 0.12,
        "telecom": 0.08,
        "city": 0.02,
    }.get(asset_type, 0.05)

    prob = base
    prob += clamp(flood_depth / 1.2, 0, 1) * 0.45
    prob += clamp(wind / 35, 0, 1) * 0.15
    prob += clamp(rain / 8, 0, 1) * 0.15
    prob += clamp(outages / 8, 0, 1) * 0.12
    prob += clamp(aqi / 100, 0, 1) * 0.05
    prob += min(t * 0.015, 0.18)

    return clamp(prob, 0.0, 0.95)


def recovery_probability(asset_type: str, flood_depth: float, t: int) -> float:
    base = {
        "power": 0.05,
        "water": 0.04,
        "telecom": 0.05,
        "city": 0.03,
    }.get(asset_type, 0.04)

    prob = base + min(t * 0.01, 0.15) - clamp(flood_depth / 1.5, 0, 1) * 0.12
    return clamp(prob, 0.0, 0.30)


def build_synced_credo_map(
    region_name: str,
    selected_place_name: Optional[str],
    places_df: pd.DataFrame,
    outages_df: pd.DataFrame,
    scenario_choice: str,
):
    center = REGIONS[region_name]["center"]

    if selected_place_name and "place" in places_df.columns:
        hit = places_df[places_df["place"] == selected_place_name]
        if not hit.empty:
            center = {
                "lat": safe_float(hit.iloc[0].get("lat")) or center["lat"],
                "lon": safe_float(hit.iloc[0].get("lon")) or center["lon"],
                "zoom": 10
            }

    m = folium.Map(
        location=[center["lat"], center["lon"]],
        zoom_start=max(int(center["zoom"]), 6),
        tiles="CartoDB positron",
        control_scale=True,
    )

    region_polygon_latlon = [[lat, lon] for lon, lat in REGIONS[region_name]["polygon"]]
    folium.Polygon(
        locations=region_polygon_latlon,
        color="#ff8800",
        weight=2,
        fill=True,
        fill_opacity=0.04,
    ).add_to(m)

    flood_layer = folium.FeatureGroup(name="Flood overlay", show=True)
    node_layer = folium.FeatureGroup(name="Assets", show=True)
    outage_layer = folium.FeatureGroup(name="Outages", show=True)

    for _, r in places_df.iterrows():
        place = r.get("place")
        lat = safe_float(r.get("lat"))
        lon = safe_float(r.get("lon"))
        if lat is None or lon is None:
            continue

        flood_depth = infer_flood_depth_for_place(r.to_dict(), scenario_choice)
        radius = 8000 + flood_depth * 7000

        if flood_depth > 0.05:
            folium.Circle(
                location=[lat, lon],
                radius=radius,
                color="#4ea5ff",
                weight=1,
                fill=True,
                fill_color="#4ea5ff",
                fill_opacity=min(0.08 + flood_depth * 0.08, 0.35),
                popup=f"{place} flood depth proxy: {flood_depth} m",
            ).add_to(flood_layer)

        highlight = place == selected_place_name
        marker_colour = "darkred" if highlight else "green"
        marker_radius = 11 if highlight else 8

        folium.CircleMarker(
            location=[lat, lon],
            radius=marker_radius,
            color="white",
            weight=2,
            fill=True,
            fill_color=marker_colour,
            fill_opacity=0.95,
            tooltip=f"{place}",
            popup=folium.Popup(
                f"""
                <b>{place}</b><br>
                Wind: {safe_float(r.get('wind_speed_10m')) or 0:.1f} km/h<br>
                Rain: {safe_float(r.get('precipitation')) or 0:.1f} mm<br>
                AQI: {safe_float(r.get('european_aqi')) or 0:.1f}<br>
                Nearby outages: {safe_float(r.get('nearby_outages_25km')) or 0:.0f}<br>
                Flood depth proxy: {flood_depth:.2f} m
                """,
                max_width=300,
            ),
        ).add_to(node_layer)

    if not outages_df.empty:
        for _, r in outages_df.iterrows():
            lat = safe_float(r.get("latitude"))
            lon = safe_float(r.get("longitude"))
            if lat is None or lon is None:
                continue

            folium.Marker(
                location=[lat, lon],
                tooltip=f"Outage | {r.get('outage_status', 'Unknown')}",
                popup=folium.Popup(
                    f"""
                    <b>Power outage</b><br>
                    Reference: {r.get('outage_reference', 'N/A')}<br>
                    Status: {r.get('outage_status', 'Unknown')}<br>
                    Category: {r.get('outage_category', 'Unknown')}<br>
                    Customers affected: {r.get('affected_customers', '')}<br>
                    Estimated restore: {r.get('estimated_restore', '')}
                    """,
                    max_width=320,
                ),
                icon=folium.Icon(color="red", icon="flash"),
            ).add_to(outage_layer)

    flood_layer.add_to(m)
    node_layer.add_to(m)
    outage_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# =========================================================
# API FETCH
# =========================================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather(lat: float, lon: float) -> Dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": WEATHER_CURRENT_VARS,
        "hourly": WEATHER_HOURLY_VARS,
        "forecast_days": 2,
        "timezone": "Europe/London",
    }
    response = session.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_air_quality(lat: float, lon: float) -> Dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": AIR_CURRENT_VARS,
        "hourly": AIR_HOURLY_VARS,
        "forecast_days": 2,
        "timezone": "Europe/London",
    }
    response = session.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=600, show_spinner=False)
def fetch_npg_live_power_cuts(limit: int = 100) -> Dict:
    safe_limit = min(max(int(limit), 1), 100)
    params = {"limit": safe_limit}

    try:
        response = session.get(NPG_DATASET_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "results" not in data:
            st.warning("NPG returned unexpected format.")
            return {"results": []}
        return data
    except Exception as e:
        st.warning(f"NPG fetch failed: {e}")
        return {"results": []}


# =========================================================
# OUTAGE NORMALISATION
# =========================================================
def payload_to_df(payload: Dict) -> pd.DataFrame:
    results = payload.get("results", [])
    if not results:
        return pd.DataFrame()
    return pd.json_normalize(results)


def region_filter_text(df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    tokens = REGIONS[region_name]["search_tokens"]
    text_cols = [c for c in df.columns if df[c].dtype == "object"]

    if not text_cols:
        return df.copy()

    haystack = (
        df[text_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )

    mask = pd.Series(False, index=df.index)
    for token in tokens:
        mask = mask | haystack.str.contains(token, regex=False)

    filtered = df[mask].copy()
    if filtered.empty:
        return df.copy()

    return filtered


def standardise_outage_df(df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    try:
        df = region_filter_text(df, region_name)
    except Exception:
        pass

    if "latitude" not in df.columns:
        df["latitude"] = np.nan
    if "longitude" not in df.columns:
        df["longitude"] = np.nan

    lat_candidates = [c for c in df.columns if "lat" in c.lower()]
    lon_candidates = [c for c in df.columns if ("lon" in c.lower() or "lng" in c.lower())]

    try:
        if lat_candidates and lon_candidates:
            df["latitude"] = pd.to_numeric(df[lat_candidates[0]], errors="coerce")
            df["longitude"] = pd.to_numeric(df[lon_candidates[0]], errors="coerce")
    except Exception:
        pass

    region_places = REGIONS.get(region_name, {}).get("places", {})

    try:
        combined_text = df.fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    except Exception:
        combined_text = pd.Series([""] * len(df))

    for place_name, (plat, plon) in region_places.items():
        try:
            mask = combined_text.str.contains(place_name.lower(), regex=False)
            df.loc[mask & df["latitude"].isna(), "latitude"] = plat
            df.loc[mask & df["longitude"].isna(), "longitude"] = plon
        except Exception:
            continue

    bbox = REGIONS.get(region_name, {}).get("bbox", None)
    if bbox is not None and "latitude" in df.columns and "longitude" in df.columns:
        try:
            geo_mask = df.apply(
                lambda r: (
                    pd.notna(r["latitude"])
                    and pd.notna(r["longitude"])
                    and point_in_bbox(float(r["latitude"]), float(r["longitude"]), bbox)
                ),
                axis=1,
            )
            geo_filtered = df[geo_mask].copy()
            if not geo_filtered.empty:
                df = geo_filtered
        except Exception:
            pass

    def safe_col(col_list):
        return col_list[0] if col_list else None

    possible_reference = [c for c in df.columns if "reference" in c.lower()]
    possible_status = [c for c in df.columns if "status" in c.lower()]
    possible_category = [c for c in df.columns if "category" in c.lower()]
    possible_postcode = [c for c in df.columns if "postcode" in c.lower()]
    possible_customers = [c for c in df.columns if "customer" in c.lower()]
    possible_estimated = [c for c in df.columns if "estimated" in c.lower() or "restore" in c.lower()]
    possible_time = [c for c in df.columns if "time" in c.lower()]

    ref_col = safe_col(possible_reference)
    status_col = safe_col(possible_status)
    category_col = safe_col(possible_category)
    postcode_col = safe_col(possible_postcode)
    customers_col = safe_col(possible_customers)
    estimated_col = safe_col(possible_estimated)
    time_col = safe_col(possible_time)

    df["outage_reference"] = df[ref_col] if ref_col else "N/A"
    df["outage_status"] = df[status_col] if status_col else "Unknown"
    df["outage_category"] = df[category_col] if category_col else "Unknown"
    df["postcode_label"] = df[postcode_col] if postcode_col else ""

    if customers_col:
        df["affected_customers"] = pd.to_numeric(df[customers_col], errors="coerce")
    else:
        df["affected_customers"] = np.nan

    df["estimated_restore"] = df[estimated_col] if estimated_col else ""
    df["time_label"] = df[time_col] if time_col else ""

    df = df.reset_index(drop=True)

    if "latitude" in df.columns and "longitude" in df.columns:
        df = df[df["latitude"].notna() & df["longitude"].notna()].copy()

    return df


# =========================================================
# MODELS
# =========================================================
def renewable_generation_model(row):
    if isinstance(row, pd.Series):
        row = row.to_dict()

    solar = safe_float(row.get("shortwave_radiation"))
    wind = safe_float(row.get("wind_speed_10m"))

    if solar is None:
        solar = 0
    if wind is None:
        wind = 0

    solar_power = solar * 0.2
    wind_power = min((wind / 12) ** 3, 1) * 100

    return solar_power + wind_power


def ev_load_model(hour: int, base_load=1.0):
    if 17 <= hour <= 22:
        return base_load * 1.8
    elif 0 <= hour <= 6:
        return base_load * 0.6
    return base_load


def compound_hazard_index(row):
    wind = safe_float(row.get("wind_speed_10m")) or 0
    rain = safe_float(row.get("precipitation")) or 0
    temp = safe_float(row.get("temperature_2m")) or 0
    return (wind * rain) / 50 + abs(temp - 18) / 10


def compute_multilayer_risk(row, outage_intensity=0.0, hour=12):
    if isinstance(row, pd.Series):
        row = row.to_dict()

    row = ensure_scalar_dict(row)

    wind = safe_float(row.get("wind_speed_10m")) or 0
    rain = safe_float(row.get("precipitation")) or 0
    cloud = safe_float(row.get("cloud_cover")) or 0
    pm25 = safe_float(row.get("pm2_5")) or 0
    aqi = safe_float(row.get("european_aqi")) or 0

    env = (wind / 40) * 0.45 + (rain / 4) * 0.25 + (cloud / 100) * 0.1 + (aqi / 80) * 0.35
    infra = float(outage_intensity or 0)

    ev_load = ev_load_model(hour)
    renewable = renewable_generation_model(row)
    net_load = ev_load * 100 - renewable

    operational = clamp(net_load / 200, 0, 1)

    total = (0.5 * env + 0.3 * infra + 0.2 * operational) * 100
    total = clamp(float(total), 0, 100)
    compound = compound_hazard_index(row)
    total += clamp(compound * 6, 0, 10)

    failure_prob = 1 / (1 + np.exp(-0.045 * (total - 70)))

    return {
        "risk_score": round(float(total), 2),
        "failure_probability": round(float(failure_prob), 3),
        "net_load": round(float(net_load), 2),
        "renewable_generation": round(float(renewable), 2),
    }


class ScenarioEngine:
    def __init__(self, scenario_name="baseline"):
        self.scenario_name = scenario_name
        self.params = self._load_scenario(scenario_name)

    def _load_scenario(self, name):
        scenarios = {
            "baseline": {"wind": 1.0, "rain": 1.0, "temp": 0, "infra": 1.0},
            "storm_cascade": {"wind": 2.2, "rain": 1.6, "temp": -1, "infra": 1.8},
            "flood_infrastructure": {"wind": 1.3, "rain": 3.5, "temp": 0, "infra": 2.2},
            "heatwave_peak": {"wind": 0.6, "rain": 0.2, "temp": +10, "infra": 1.5},
            "pollution_event": {"wind": 0.4, "rain": 0.1, "temp": +5, "infra": 1.2},
            "compound_extreme": {"wind": 1.8, "rain": 2.8, "temp": +6, "infra": 3.0},
        }
        return scenarios.get(name, scenarios["baseline"])

    def apply(self, row):
        if isinstance(row, pd.Series):
            row = row.to_dict()
        row = dict(row)

        row["wind_speed_10m"] = (safe_float(row.get("wind_speed_10m")) or 0) * self.params["wind"]
        row["precipitation"] = (safe_float(row.get("precipitation")) or 0) * self.params["rain"]
        row["temperature_2m"] = (safe_float(row.get("temperature_2m")) or 0) + self.params["temp"]

        row["shortwave_radiation"] = (safe_float(row.get("shortwave_radiation")) or 0) * max(0, (1 - 0.3 * self.params["rain"]))
        row["european_aqi"] = (safe_float(row.get("european_aqi")) or 0) * (1 + 0.4 * self.params["infra"])
        return row


class InfrastructureGraph:
    def __init__(self):
        self.graph = {
            "power": {"depends_on": [], "impact_factor": 1.0},
            "water": {"depends_on": ["power"], "impact_factor": 0.7},
            "telecom": {"depends_on": ["power"], "impact_factor": 0.8},
            "transport": {"depends_on": ["power", "telecom"], "impact_factor": 0.6},
        }

    def propagate_failure(self, base_failure):
        base_failure = clamp(float(base_failure or 0), 0, 1)
        system_state = {"power": base_failure}

        for infra, config in self.graph.items():
            if infra == "power":
                continue

            dependency_failures = [system_state.get(dep, 0) for dep in config["depends_on"]]

            if dependency_failures:
                combined = np.mean(dependency_failures)
                system_state[infra] = clamp((combined ** 1.5) * config["impact_factor"], 0, 1)
            else:
                system_state[infra] = 0

        total_system_stress = np.mean(list(system_state.values()))
        return clamp(float(total_system_stress), 0, 1), system_state


def enhanced_risk_with_cascade(row, outage_intensity, scenario_engine):
    if isinstance(row, pd.Series):
        row = row.to_dict()

    scenario_row = scenario_engine.apply(row)

    if scenario_engine.scenario_name == "baseline":
        outage_intensity *= 0.55

    base = compute_multilayer_risk(scenario_row, outage_intensity)

    graph = InfrastructureGraph()
    system_stress, system_breakdown = graph.propagate_failure(base["failure_probability"])

    infra_multiplier = scenario_engine.params.get("infra", 1.0)
    scenario_name = scenario_engine.scenario_name
    cascade_factor = 0.9 if scenario_name == "baseline" else 1.3

    final_risk = base["risk_score"] * (1 + system_stress * infra_multiplier * cascade_factor)

    if scenario_name != "baseline":
        severity = infra_multiplier
        scenario_boost = (final_risk / 100) ** 1.7 * severity * 30
        final_risk += scenario_boost

    final_risk = clamp(final_risk, 0, 100)
    final_failure_probability = 1 / (1 + np.exp(-0.06 * (final_risk - 60)))

    return {
        **base,
        "failure_probability": round(float(final_failure_probability), 3),
        "system_stress": round(float(system_stress), 3),
        "final_risk_score": round(float(final_risk), 2),
        "cascade_power": float(system_breakdown["power"]),
        "cascade_water": float(system_breakdown["water"]),
        "cascade_telecom": float(system_breakdown["telecom"]),
        "cascade_transport": float(system_breakdown["transport"]),
    }


def monte_carlo_risk(row, outage_intensity, simulations=30):
    scores = []
    base_row = dict(row) if not isinstance(row, pd.Series) else row.to_dict()

    for _ in range(int(simulations)):
        perturbed = base_row.copy()
        perturbed["wind_speed_10m"] = safe_float(perturbed.get("wind_speed_10m")) or 0
        perturbed["precipitation"] = safe_float(perturbed.get("precipitation")) or 0

        perturbed["wind_speed_10m"] *= random.uniform(0.9, 1.1)
        perturbed["precipitation"] *= random.uniform(0.8, 1.2)

        risk = compute_multilayer_risk(perturbed, outage_intensity)
        scores.append(risk["risk_score"])

    return {
        "risk_std": float(np.std(scores)) if scores else 0.0,
        "risk_p95": float(np.percentile(scores, 95)) if scores else 0.0,
    }


def run_time_simulation(row, outage_intensity, scenario_engine, hours=24):
    timeline = []

    for h in range(hours):
        modified = dict(row)
        modified["wind_speed_10m"] = (safe_float(modified.get("wind_speed_10m")) or 0) * random.uniform(0.9, 1.2)
        modified["precipitation"] = (safe_float(modified.get("precipitation")) or 0) * random.uniform(0.8, 1.3)

        risk = enhanced_risk_with_cascade(modified, outage_intensity * (1 + h / 24), scenario_engine)

        timeline.append({
            "hour": h,
            "risk": risk["final_risk_score"],
            "failure_prob": risk["failure_probability"],
        })

    return pd.DataFrame(timeline)

def safe_tile_url(layer_mode, date_str):
    for offset in range(0, 5):
        test_date = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=offset)
        url = get_story_satellite_url(layer_mode, test_date.strftime("%Y-%m-%d"))

        # hızlı test request
        try:
            r = requests.head(url.replace("{z}", "2").replace("{x}", "2").replace("{y}", "2"), timeout=3)
            if r.status_code == 200:
                return url
        except:
            continue

    return get_story_satellite_url(layer_mode, (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d"))

def compute_location_risk(row: Dict, outage_intensity: float = 0.0) -> Dict:
    row = ensure_scalar_dict(row)

    wind = safe_float(row.get("wind_speed_10m")) or 0
    cloud = safe_float(row.get("cloud_cover")) or 0
    rain = safe_float(row.get("precipitation")) or 0
    pm25 = safe_float(row.get("pm2_5")) or 0
    no2 = safe_float(row.get("nitrogen_dioxide")) or 0
    aqi = safe_float(row.get("european_aqi")) or 0
    humidity = safe_float(row.get("relative_humidity_2m")) or 0
    temp = safe_float(row.get("temperature_2m")) or 0

    wind_score = clamp((wind / 60) * 35, 0, 35)
    rain_score = clamp((rain / 5) * 15, 0, 15)
    cloud_score = clamp((cloud / 100) * 7, 0, 7)
    air_score = clamp((aqi / 100) * 15, 0, 15) + clamp((pm25 / 60) * 8, 0, 8) + clamp((no2 / 120) * 5, 0, 5)
    humidity_score = clamp((humidity / 100) * 5, 0, 5)
    thermal_score = clamp(abs(temp - 18) / 18 * 5, 0, 5)
    outage_score = clamp(outage_intensity * 10, 0, 10)
    impact = outage_intensity * 100

    total = wind_score + rain_score + cloud_score + air_score + humidity_score + thermal_score + outage_score
    total = clamp(total, 0, 100)

    if total >= 75:
        label = "Severe"
        colour = [215, 48, 39, 180]
    elif total >= 55:
        label = "High"
        colour = [252, 141, 89, 170]
    elif total >= 35:
        label = "Moderate"
        colour = [254, 224, 144, 160]
    else:
        label = "Low"
        colour = [145, 207, 96, 150]

    failure_probability = clamp(total / 100, 0, 1)

    return {
        "risk_score": round(float(total), 1),
        "outage_impact_score": round(float(impact), 1),
        "risk_label": label,
        "risk_colour": colour,
        "failure_probability": round(float(failure_probability), 3),
    }


from concurrent.futures import ThreadPoolExecutor

def fetch_all_places(places_dict):
    def worker(item):
        place, (lat, lon) = item
        return place, combine_weather_air(place, lat, lon)

    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(worker, places_dict.items()))

    return dict(results)
# =========================================================
# DATA COMBINATION
# =========================================================
def combine_weather_air(place_name: str, lat: float, lon: float) -> Dict:
    weather = fetch_weather(lat, lon)
    air = fetch_air_quality(lat, lon)

    current_w = weather.get("current", {})
    current_a = air.get("current", {})

    row = {
        "place": place_name,
        "lat": lat,
        "lon": lon,
        "time": current_w.get("time"),
        "temperature_2m": current_w.get("temperature_2m"),
        "apparent_temperature": current_w.get("apparent_temperature"),
        "wind_speed_10m": current_w.get("wind_speed_10m"),
        "wind_direction_10m": current_w.get("wind_direction_10m"),
        "surface_pressure": current_w.get("surface_pressure"),
        "cloud_cover": current_w.get("cloud_cover"),
        "shortwave_radiation": current_w.get("shortwave_radiation"),
        "direct_radiation": current_w.get("direct_radiation"),
        "diffuse_radiation": current_w.get("diffuse_radiation"),
        "relative_humidity_2m": current_w.get("relative_humidity_2m"),
        "precipitation": current_w.get("precipitation"),
        "is_day": current_w.get("is_day"),
        "european_aqi": current_a.get("european_aqi"),
        "pm2_5": current_a.get("pm2_5"),
        "pm10": current_a.get("pm10"),
        "nitrogen_dioxide": current_a.get("nitrogen_dioxide"),
        "ozone": current_a.get("ozone"),
        "sulphur_dioxide": current_a.get("sulphur_dioxide"),
        "carbon_monoxide": current_a.get("carbon_monoxide"),
        "aerosol_optical_depth": current_a.get("aerosol_optical_depth"),
        "dust": current_a.get("dust"),
        "uv_index": current_a.get("uv_index"),
    }

    return {
        "weather_raw": weather,
        "air_raw": air,
        "current_row": row,
    }


def build_hourly_dataframe(weather_raw: Dict, air_raw: Dict) -> pd.DataFrame:
    w = pd.DataFrame(weather_raw.get("hourly", {}))
    a = pd.DataFrame(air_raw.get("hourly", {}))

    if w.empty and a.empty:
        return pd.DataFrame()

    if not w.empty and not a.empty and "time" in w.columns and "time" in a.columns:
        df = pd.merge(w, a, on="time", how="outer")
    elif not w.empty:
        df = w.copy()
    else:
        df = a.copy()

    for col in [
        "wind_speed_10m", "precipitation", "cloud_cover", "pm2_5",
        "nitrogen_dioxide", "european_aqi", "relative_humidity_2m", "temperature_2m"
    ]:
        if col not in df.columns:
            df[col] = 0

    def hourly_risk(r):
        temp_row = {
            "wind_speed_10m": r.get("wind_speed_10m", 0),
            "cloud_cover": r.get("cloud_cover", 0),
            "precipitation": r.get("precipitation", 0),
            "pm2_5": r.get("pm2_5", 0),
            "nitrogen_dioxide": r.get("nitrogen_dioxide", 0),
            "european_aqi": r.get("european_aqi", 0),
            "relative_humidity_2m": r.get("relative_humidity_2m", 0),
            "temperature_2m": r.get("temperature_2m", 0),
        }
        return compute_multilayer_risk(temp_row, outage_intensity=0)["risk_score"]

    df["predicted_risk_score"] = df.apply(hourly_risk, axis=1)
    return df


def build_place_dataframe(region_name: str, outages_df: pd.DataFrame, simulations: int = 30):
    rows = []
    raw_cache = {}

    # 🔥 faster iteration
    outage_points = []
    for r in outages_df.to_dict("records"):
        lat = safe_float(r.get("latitude"))
        lon = safe_float(r.get("longitude"))
        if lat is not None and lon is not None:
            outage_points.append((lat, lon))

    places_dict = REGIONS[region_name]["places"]

    # 🚀 paralel fetch
    all_data = fetch_all_places(places_dict)

    for place, (lat, lon) in places_dict.items():
        combined = all_data.get(place)

        # ❗ safety
        if combined is None:
            combined = {
                "weather_raw": {},
                "air_raw": {},
                "current_row": {
                    "place": place,
                    "lat": lat,
                    "lon": lon,
                }
            }

        # ❗ copy to avoid mutation bugs
        row = dict(combined["current_row"])

        nearby_outages = sum(
            1 for olat, olon in outage_points
            if haversine_km(lat, lon, olat, olon) <= 25
        )

        outage_intensity = clamp(nearby_outages / 20, 0, 1)

        risk = compute_multilayer_risk(row, outage_intensity)

        if simulations > 0:
            mc = monte_carlo_risk(row, outage_intensity, simulations=simulations)
        else:
            mc = {"risk_std": 0, "risk_p95": 0}

        row.update({
            "nearby_outages_25km": nearby_outages,
            **risk,
            **mc,
        })

        rows.append(row)
        raw_cache[place] = combined

    return pd.DataFrame(rows), raw_cache


# =========================================================
# GRID INTERPOLATION
# =========================================================
def interpolate_weather_value(grid_lat, grid_lon, places_df: pd.DataFrame, col: str) -> float:
    weights = []
    values = []

    for _, r in places_df.iterrows():
        lat = safe_float(r.get("lat"))
        lon = safe_float(r.get("lon"))
        value = safe_float(r.get(col))
        if lat is None or lon is None or value is None:
            continue
        d = haversine_km(grid_lat, grid_lon, lat, lon)
        w = 1 / max(d, 1.0)
        weights.append(w)
        values.append(value)

    if not weights:
        return 0.0

    return float(np.average(values, weights=weights))


def count_outages_near(grid_lat, grid_lon, outages_df: pd.DataFrame, radius_km=20) -> int:
    if outages_df.empty:
        return 0

    total = 0
    for _, r in outages_df.iterrows():
        olat = safe_float(r.get("latitude"))
        olon = safe_float(r.get("longitude"))
        if olat is None or olon is None:
            continue
        if haversine_km(grid_lat, grid_lon, olat, olon) <= radius_km:
            total += 1

    return total


def build_digital_twin_grid(region_name: str, places_df: pd.DataFrame, outages_df: pd.DataFrame, scenario_engine=None) -> pd.DataFrame:
    bbox = REGIONS[region_name]["bbox"]
    min_lon, min_lat, max_lon, max_lat = bbox

    lats = np.linspace(min_lat, max_lat, 12)
    lons = np.linspace(min_lon, max_lon, 12)

    cells = []

    for lat in lats:
        for lon in lons:
            wind = interpolate_weather_value(lat, lon, places_df, "wind_speed_10m")
            cloud = interpolate_weather_value(lat, lon, places_df, "cloud_cover")
            rain = interpolate_weather_value(lat, lon, places_df, "precipitation")
            pm25 = interpolate_weather_value(lat, lon, places_df, "pm2_5")
            no2 = interpolate_weather_value(lat, lon, places_df, "nitrogen_dioxide")
            aqi = interpolate_weather_value(lat, lon, places_df, "european_aqi")
            humidity = interpolate_weather_value(lat, lon, places_df, "relative_humidity_2m")
            temp = interpolate_weather_value(lat, lon, places_df, "temperature_2m")
            solar = interpolate_weather_value(lat, lon, places_df, "shortwave_radiation")

            outages_near = count_outages_near(lat, lon, outages_df, radius_km=20)
            outage_intensity = clamp(outages_near / 15, 0, 1)

            base_row = {
                "wind_speed_10m": wind,
                "cloud_cover": cloud,
                "precipitation": rain,
                "pm2_5": pm25,
                "nitrogen_dioxide": no2,
                "european_aqi": aqi,
                "relative_humidity_2m": humidity,
                "temperature_2m": temp,
                "shortwave_radiation": solar,
            }

            if scenario_engine is not None:
                enhanced = enhanced_risk_with_cascade(base_row, outage_intensity, scenario_engine)
            else:
                enhanced = compute_multilayer_risk(base_row, outage_intensity)
                enhanced["system_stress"] = 0
                enhanced["final_risk_score"] = enhanced["risk_score"]
                enhanced["cascade_power"] = 0
                enhanced["cascade_water"] = 0
                enhanced["cascade_telecom"] = 0
                enhanced["cascade_transport"] = 0

            final_risk = enhanced.get("final_risk_score", enhanced.get("risk_score", 0))
            final_risk = safe_float(final_risk) or 0
            label = get_risk_label(final_risk)

            cells.append({
                "lat": float(lat),
                "lon": float(lon),
                "wind_speed_10m": round(float(wind), 2),
                "cloud_cover": round(float(cloud), 2),
                "precipitation": round(float(rain), 2),
                "pm2_5": round(float(pm25), 2),
                "aqi": round(float(aqi), 2),
                "temperature_2m": round(float(temp), 2),
                "solar": round(float(solar), 2),
                "outages_near_20km": int(outages_near),
                "risk_score": safe_float(enhanced.get("risk_score", 0)) or 0,
                "final_risk_score": round(float(final_risk), 2),
                "system_stress": safe_float(enhanced.get("system_stress", 0)) or 0,
                "failure_probability": safe_float(enhanced.get("failure_probability", 0)) or 0,
                "cascade_power": safe_float(enhanced.get("cascade_power", 0)) or 0,
                "cascade_water": safe_float(enhanced.get("cascade_water", 0)) or 0,
                "cascade_telecom": safe_float(enhanced.get("cascade_telecom", 0)) or 0,
                "cascade_transport": safe_float(enhanced.get("cascade_transport", 0)) or 0,
                "risk_label": label,
            })

    return pd.DataFrame(cells)


# =========================================================
# TITLE + SIDEBAR
# =========================================================
st.title("North East & Yorkshire Grid Digital Twin")
st.caption("Live digital twin for weather, pollution, solar conditions, satellite layers, outage monitoring and predictive risk screening.")

with st.sidebar:
    st.header("Controls")
    region_name = st.selectbox("Region", list(REGIONS.keys()), index=0)
    satellite_name = st.selectbox("Satellite layer", list(SATELLITE_LAYERS.keys()), index=0)
    selected_place = st.selectbox("Detailed forecast site", list(REGIONS[region_name]["places"].keys()), index=0)
    outage_limit = st.slider("Maximum live outage records to request", 10, 100, 100, 10)
    risk_filter = st.slider("Minimum risk level", 0, 100, 0)
    
    
    use_scenario = st.checkbox("Enable scenario simulation", value=False)

    if use_scenario:
        scenario_choice = st.selectbox(
            "Scenario type",
            [
                "storm_cascade",
                "flood_infrastructure",
                "heatwave_peak",
                "pollution_event",
                "compound_extreme",
            ],
        )
    else:
        scenario_choice = "baseline"
        
        st.caption("Default mode is real-time (live digital twin). Enable scenarios for stress testing.")


    st.markdown("---")
    st.subheader("Auto refresh")
    st.write("This app refreshes automatically every 30 seconds.")

    st.markdown("---")
    simulated_wind = st.slider("Simulate wind increase (%)", 0, 100, 0)
    mc_runs = st.slider("Monte Carlo simulations", 10, 100, 30)

    st.subheader("Digital twin logic")
    st.write(
        """
        Risk score blends:
        - wind stress
        - precipitation and cloud burden
        - pollution stress
        - thermal deviation
        - humidity
        - nearby outage concentration
        """
    )
        
@st.cache_data(show_spinner=False)
def run_scenario_cached(places_df: pd.DataFrame, scenario_choice: str):
    scenario_engine = ScenarioEngine(scenario_choice or "baseline")

    enhanced_rows = []

    for _, r in places_df.iterrows():
        enhanced = enhanced_risk_with_cascade(
            r.to_dict(),
            clamp((safe_float(r.get("nearby_outages_25km")) or 0) / 25, 0, 1),
            scenario_engine,
        )
        enhanced_rows.append(enhanced)

    return pd.DataFrame(enhanced_rows).add_prefix("enh_")

# =========================================================
# DATA PREP (OPTIMISED + SCENARIO CACHE)
# =========================================================

if "base_data" not in st.session_state:
    st.session_state["base_data"] = None

try:
    # -----------------------------------------------------
    # STEP 1: LOAD HEAVY DATA ONLY ONCE
    # -----------------------------------------------------
    if st.session_state["base_data"] is None:

        raw_npg = fetch_npg_live_power_cuts(limit=outage_limit)
        raw_npg_df = payload_to_df(raw_npg)
        outages_df = standardise_outage_df(raw_npg_df, region_name)

        safe_mc = min(mc_runs, 15)

        places_df, raw_cache = build_place_dataframe(
            region_name,
            outages_df,
            simulations=safe_mc
        )

        st.session_state["base_data"] = {
            "outages_df": outages_df,
            "places_df": places_df,
            "raw_cache": raw_cache
        }

    # -----------------------------------------------------
    # STEP 2: LOAD FROM CACHE (FAST)
    # -----------------------------------------------------
    base = st.session_state["base_data"]

    outages_df = base["outages_df"].copy()
    places_df = base["places_df"].copy()
    raw_cache = base["raw_cache"]

    # -----------------------------------------------------
    # STEP 3: SCENARIO (NOW CACHED ⚡)
    # -----------------------------------------------------
    places_df_for_cache = places_df.copy().reset_index(drop=True)

    enhanced_df = run_scenario_cached(
        places_df_for_cache,
        scenario_choice
    )

    places_df = pd.concat(
        [places_df.reset_index(drop=True), enhanced_df.reset_index(drop=True)],
        axis=1,
    )

    # -----------------------------------------------------
    # STEP 4: FINAL METRICS
    # -----------------------------------------------------
    places_df["risk_score"] = places_df["enh_risk_score"]
    places_df["failure_probability"] = places_df["enh_failure_probability"]
    places_df["net_load"] = places_df["enh_net_load"]
    places_df["renewable_generation"] = places_df["enh_renewable_generation"]
    places_df["final_risk_score"] = places_df["enh_final_risk_score"]

    for col in ["risk_score", "final_risk_score", "failure_probability"]:
        if col in places_df.columns:
            places_df[col] = places_df[col].apply(force_scalar)

    if "risk_score" in places_df.columns:
        places_df["risk_label"] = places_df["risk_score"].apply(get_risk_label)

    # -----------------------------------------------------
    # STEP 5: USER CONTROLS (VERY LIGHT)
    # -----------------------------------------------------
    if "wind_speed_10m" in places_df.columns:
        places_df["wind_speed_10m"] = (
            pd.to_numeric(places_df["wind_speed_10m"], errors="coerce")
            .fillna(0)
            * (1 + simulated_wind / 100)
        )

    if "shortwave_radiation" not in places_df.columns:
        places_df["shortwave_radiation"] = 0

    places_df["renewable_score"] = (
        pd.to_numeric(places_df["shortwave_radiation"], errors="coerce").fillna(0) * 0.6
        + pd.to_numeric(places_df["wind_speed_10m"], errors="coerce").fillna(0) * 0.4
    )

    # -----------------------------------------------------
    # STEP 6: DIGITAL TWIN GRID
    # -----------------------------------------------------
    digital_twin_df = build_digital_twin_grid(
        region_name,
        places_df,
        outages_df,
        ScenarioEngine(scenario_choice)
    )

    digital_twin_df = digital_twin_df[
        digital_twin_df["risk_score"] >= risk_filter
    ]

    # -----------------------------------------------------
    # STEP 7: SELECTED SITE
    # -----------------------------------------------------
    selected_weather = raw_cache[selected_place]["weather_raw"]
    selected_air = raw_cache[selected_place]["air_raw"]
    selected_current = raw_cache[selected_place]["current_row"]

    hourly_df = build_hourly_dataframe(selected_weather, selected_air)

except Exception as e:
    st.error(f"Data fetch failed: {e}")
    st.stop()


# =========================================================
# TOP METRICS
# =========================================================

# 🔧 FIX: define scenario_engine here
scenario_engine = ScenarioEngine(scenario_choice or "baseline")

regional_risk = round(float(places_df["risk_score"].mean()), 1) if not places_df.empty and "risk_score" in places_df.columns else 0
regional_failure_prob = round(float(places_df["failure_probability"].mean()) * 100, 1) if not places_df.empty and "failure_probability" in places_df.columns else 0
live_outages = len(outages_df)

solar_value, solar_note = solar_interpretation(selected_current)

m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
m1.metric("Regional risk", regional_risk, delta=f"{round(regional_risk - 50, 1)} vs baseline")
m2.metric("Predicted failure probability", f"{regional_failure_prob}%")
m3.metric("Live outage records", live_outages)
m4.metric("Wind speed", f"{selected_current.get('wind_speed_10m', '—')} km/h")
m5.metric("Solar radiation", solar_value)
m6.metric("European AQI", f"{selected_current.get('european_aqi', '—')}")
m7.metric("Renewable potential", round(float(places_df["renewable_score"].mean()), 1) if "renewable_score" in places_df.columns else 0)
m8.metric("Grid net load", round(float(places_df["net_load"].mean()), 2) if "net_load" in places_df.columns else 0)

st.info(solar_note)

st.markdown("### 🔬 Scenario-adjusted system state")

scenario_row = scenario_engine.apply(selected_current)

scenario_risk = compute_multilayer_risk(
    scenario_row,
    outage_intensity=clamp(live_outages / 25, 0, 1)
)

scenario_wind = safe_float(scenario_row.get("wind_speed_10m")) or 0
scenario_solar = safe_float(scenario_row.get("shortwave_radiation")) or 0
scenario_aqi = safe_float(scenario_row.get("european_aqi")) or 0

sc1, sc2, sc3, sc4, sc5 = st.columns(5)
sc1.metric(
    "Scenario wind",
    f"{round(scenario_wind,1)} km/h",
    delta=round(scenario_wind - (safe_float(selected_current.get("wind_speed_10m")) or 0), 1)
)
sc2.metric("Scenario solar", f"{round(scenario_solar,1)} W/m²")
sc3.metric("Scenario AQI", round(scenario_aqi, 1))
sc4.metric("Scenario renewable", scenario_risk["renewable_generation"])
sc5.metric("Scenario net load", scenario_risk["net_load"])

st.markdown("### ⚡ System Status")
if regional_risk >= 70:
    st.error("CRITICAL GRID RISK — Immediate action required")
elif regional_risk >= 55:
    st.warning("Elevated grid stress detected")
elif regional_risk >= 35:
    st.info("Moderate operational conditions")
else:
    st.success("System stable")


# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Digital Twin Map",
    "Regional Intelligence",
    "Selected Site Forecast",
    "Live Outages",
    "Raw Twin Grid",
    "CReDo Network",
    "Satellite Analytics" 
])


# TAB 1 #

# =========================================================
with tab1:
    selected_node = st.session_state.get("credo_selected_node")
    selected_place_from_graph = get_place_from_node(selected_node) if selected_node else None

    if selected_place_from_graph and "place" in places_df.columns:
        selected_rows = places_df[places_df["place"] == selected_place_from_graph]
        if not selected_rows.empty:
            center = {
                "lat": safe_float(selected_rows.iloc[0].get("lat")) or REGIONS[region_name]["center"]["lat"],
                "lon": safe_float(selected_rows.iloc[0].get("lon")) or REGIONS[region_name]["center"]["lon"],
                "zoom": 10
            }
        else:
            center = REGIONS[region_name]["center"]
    else:
        center = REGIONS[region_name]["center"]

    st.subheader(f"{region_name} live digital twin map")

    base_tiles = {
        "Satellite-style dark": "CartoDB dark_matter",
        "Light": "CartoDB positron",
        "OpenStreetMap": "OpenStreetMap",
    }

    map_tile_choice = st.radio(
        "Map style",
        list(base_tiles.keys()),
        horizontal=True,
        index=0,
        key=f"map_style_{region_name}",
    )

    m = folium.Map(
        location=[center["lat"], center["lon"]],
        zoom_start=max(int(center["zoom"]), 6),
        tiles=base_tiles[map_tile_choice],
        control_scale=True,
    )

    try:
        tile_url = get_satellite_tile_url(satellite_name)
        folium.raster_layers.TileLayer(
            tiles=tile_url,
            attr="NASA GIBS",
            name=f"{satellite_name}",
            overlay=True,
            control=True,
            opacity=0.55,
        ).add_to(m)
    except Exception:
        st.warning("Satellite layer failed to load.")

    region_polygon_latlon = [[lat, lon] for lon, lat in REGIONS[region_name]["polygon"]]
    folium.Polygon(
        locations=region_polygon_latlon,
        color="orange",
        weight=2,
        fill=True,
        fill_opacity=0.05,
    ).add_to(m)

    # optional flood / water proxy tile
    try:
        water_mask_tile = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_WaterMask/default/{date}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.png".replace(
            "{date}", datetime.utcnow().strftime("%Y-%m-%d")
        )
        folium.raster_layers.TileLayer(
            tiles=water_mask_tile,
            attr="NASA GIBS Water Mask",
            name="Water / flood mask",
            overlay=True,
            control=True,
            opacity=0.30,
        ).add_to(m)
    except Exception:
        pass

    if not digital_twin_df.empty:
        twin_df = digital_twin_df.copy()
        max_risk = twin_df["risk_score"].max() if "risk_score" in twin_df else 1
        max_risk = max(max_risk, 1)

        heat_data = [
            [safe_float(r["lat"]), safe_float(r["lon"]), (safe_float(r["risk_score"]) or 0) / max_risk]
            for _, r in twin_df.iterrows()
            if safe_float(r["lat"]) and safe_float(r["lon"])
        ]

        if heat_data:
            HeatMap(
                heat_data,
                radius=30,
                blur=22,
                min_opacity=0.25,
                name="Risk heatmap"
            ).add_to(m)

    if not digital_twin_df.empty:
        for _, row in digital_twin_df.iterrows():
            lat = safe_float(row.get("lat"))
            lon = safe_float(row.get("lon"))
            if lat is None or lon is None:
                continue

            risk = safe_float(row.get("final_risk_score")) or 0
            stress = safe_float(row.get("system_stress")) or 0

            if stress > 0.7 or risk > 80:
                colour = "darkred"
            elif stress > 0.5 or risk > 60:
                colour = "red"
            elif risk > 40:
                colour = "orange"
            else:
                colour = "green"

            folium.CircleMarker(
                location=[lat, lon],
                radius=9,
                color=colour,
                fill=True,
                fill_opacity=0.9,
                popup=f"""
                <b>Digital Twin Cell</b><br>
                Risk: {round(risk,1)}<br>
                Stress: {round(stress,2)}<br>
                Wind: {row.get("wind_speed_10m")}<br>
                AQI: {row.get("aqi")}<br>
                """
            ).add_to(m)

    def generate_infra_nodes(df):
        nodes = []
        for _, r in df.iterrows():
            lat = safe_float(r.get("lat"))
            lon = safe_float(r.get("lon"))
            if lat is None or lon is None:
                continue
            nodes.append(("power", lat, lon))
            nodes.append(("water", lat + 0.02, lon + 0.02))
            nodes.append(("telecom", lat - 0.02, lon - 0.02))
        return nodes

    infra_nodes = generate_infra_nodes(places_df)
    colour_map = {"power": "yellow", "water": "blue", "telecom": "purple"}

    for t, lat, lon in infra_nodes:
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=colour_map.get(t, "white"),
            fill=True,
            fill_opacity=0.9,
            popup=f"{t} asset"
        ).add_to(m)

    for i in range(len(infra_nodes) - 1):
        a = infra_nodes[i]
        b = infra_nodes[i + 1]
        folium.PolyLine(
            locations=[[a[1], a[2]], [b[1], b[2]]],
            color="yellow",
            weight=1,
            opacity=0.3
        ).add_to(m)

    if not digital_twin_df.empty:
        for _, row in digital_twin_df.iterrows():
            risk = safe_float(row.get("final_risk_score")) or 0
            if risk > 65:
                folium.Circle(
                    location=[row["lat"], row["lon"]],
                    radius=2000,
                    color="blue",
                    fill=True,
                    fill_opacity=0.08
                ).add_to(m)

    for _, row in places_df.iterrows():
        lat = safe_float(row.get("lat"))
        lon = safe_float(row.get("lon"))
        if lat is None or lon is None:
            continue

        risk = safe_float(row.get("final_risk_score")) or 0
        is_selected_from_graph = selected_place_from_graph == row.get("place")

        if risk > 75:
            colour = "darkred"
        elif risk > 55:
            colour = "red"
        elif risk > 35:
            colour = "orange"
        else:
            colour = "green"

        marker_radius = 14 if is_selected_from_graph else 10
        marker_border = "red" if is_selected_from_graph else "white"

        folium.CircleMarker(
            location=[lat, lon],
            radius=marker_radius,
            color=marker_border,
            weight=2,
            fill=True,
            fill_color=colour,
            fill_opacity=0.95,
            tooltip=f"{row['place']} | Risk {risk}",
        ).add_to(m)

    if not outages_df.empty:
        for _, row in outages_df.iterrows():
            lat = safe_float(row.get("latitude"))
            lon = safe_float(row.get("longitude"))
            if lat is None or lon is None:
                continue

            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color="red", icon="flash"),
                tooltip="Outage"
            ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    st.caption("CReDo-style multi-layer digital twin (risk, infrastructure, cascade, hazard)")
    components.html(m._repr_html_(), height=720)

    c1, c2, c3 = st.columns(3)
    c1.metric("High-risk cells", len(digital_twin_df[digital_twin_df["risk_score"] > 55]))
    c2.metric("Severe cells", len(digital_twin_df[digital_twin_df["risk_score"] > 75]))
    if not places_df.empty:
        worst = places_df.sort_values("risk_score", ascending=False).iloc[0]["place"]
        c3.metric("Highest-risk location", worst)

# =========================================================
# TAB 2
# =========================================================
with tab2:
    st.subheader(f"{region_name} regional intelligence")

    k1, k2, k3, k4 = st.columns(4)
    avg_risk = round(float(places_df.get("risk_score", pd.Series([0])).mean()), 1)
    max_risk = round(float(places_df.get("risk_score", pd.Series([0])).max()), 1)
    avg_aqi = round(float(places_df.get("european_aqi", pd.Series([0])).mean()), 1)
    avg_wind = round(float(places_df.get("wind_speed_10m", pd.Series([0])).mean()), 1)

    k1.metric("Avg risk", avg_risk)
    k2.metric("Max risk", max_risk)
    k3.metric("Avg AQI", avg_aqi)
    k4.metric("Avg wind (km/h)", avg_wind)

    st.markdown("---")
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown("### Representative location risk table")
        display_cols = [
            "place", "temperature_2m", "wind_speed_10m",
            "shortwave_radiation", "cloud_cover", "precipitation",
            "european_aqi", "pm2_5",
            "nearby_outages_25km",
            "risk_score", "failure_probability",
            "net_load", "renewable_generation",
        ]
        df_safe = places_df.reindex(columns=display_cols)

        if "risk_score" in df_safe.columns:
            df_safe = df_safe.sort_values("risk_score", ascending=False)

        st.dataframe(df_safe, use_container_width=True, height=320)

    with right:
        st.markdown("### Top vulnerable areas")

        if "risk_score" in places_df.columns:
            top_places = places_df.sort_values("risk_score", ascending=False).head(5)
        else:
            top_places = places_df.head(5)

        for _, r in top_places.iterrows():
            r = r.to_dict()
            risk_score = safe_float(r.get("risk_score")) or 0
            risk_label = get_risk_label(risk_score)

            st.markdown(
                f"""
                **{r.get('place', 'Unknown')}**
                - Risk: **{risk_score} ({risk_label})**
                - Failure probability: **{round((safe_float(r.get('failure_probability')) or 0) * 100, 1)}%**
                - Wind: {r.get('wind_speed_10m', '—')} km/h  
                - AQI: {r.get('european_aqi', '—')}  
                - Nearby outages: {r.get('nearby_outages_25km', 0)}
                """
            )

    st.markdown("---")
    st.markdown("### Cascading infrastructure stress")
    if "system_stress" in places_df.columns:
        plot_df = places_df[["place", "system_stress"]].copy()
        plot_df["system_stress"] = pd.to_numeric(plot_df["system_stress"], errors="coerce").fillna(0)
        st.bar_chart(plot_df.set_index("place")["system_stress"])

    st.markdown("### Infrastructure breakdown")
    for _, r in places_df.iterrows():
        st.write(
            f"{r['place']} → "
            f"Power:{round(safe_float(r.get('cascade_power')) or 0, 2)} | "
            f"Water:{round(safe_float(r.get('cascade_water')) or 0, 2)} | "
            f"Telecom:{round(safe_float(r.get('cascade_telecom')) or 0, 2)}"
        )

    st.markdown("### Predicted risk trend (next 24h)")
    if not hourly_df.empty and "predicted_risk_score" in hourly_df.columns:
        st.line_chart(hourly_df["predicted_risk_score"])
    else:
        st.warning("No forecast data available.")

    st.markdown("---")
    st.markdown("### Renewable energy potential")
    if not places_df.empty:
        places_df["renewable_score"] = (
            pd.to_numeric(places_df.get("shortwave_radiation", 0), errors="coerce").fillna(0) * 0.6
            + pd.to_numeric(places_df.get("wind_speed_10m", 0), errors="coerce").fillna(0) * 0.4
        )
        st.metric("Regional renewable potential", round(float(places_df["renewable_score"].mean()), 1))

    st.markdown("---")
    st.markdown("### System interpretation")
    if avg_risk >= 70:
        st.error("Critical grid stress driven by extreme environmental conditions and outage clustering.")
    elif avg_risk >= 55:
        st.warning("Elevated grid stress due to combined wind, pollution and outage activity.")
    elif avg_risk >= 35:
        st.info("Moderate stress levels with localised vulnerabilities.")
    else:
        st.success("Grid operating under stable environmental conditions.")

    st.markdown("---")
    st.markdown("### AI interpretation")
    if regional_risk > 60:
        st.write("High wind and pollution levels are driving elevated grid stress.")
    elif regional_risk > 40:
        st.write("Moderate environmental stress observed across the region.")
    else:
        st.write("Grid operating under stable environmental conditions.")

    st.markdown("---")
    st.markdown("### Regional digital twin summary")
    twin_summary = pd.DataFrame({
        "Metric": [
            "Average risk score",
            "Maximum risk score",
            "Average predicted failure probability",
            "Mean PM2.5",
            "Mean wind speed",
            "Mean solar radiation",
            "Live outages",
        ],
        "Value": [
            round(float(digital_twin_df.get("risk_score", pd.Series([0])).mean()), 2),
            round(float(digital_twin_df.get("risk_score", pd.Series([0])).max()), 2),
            round(float(digital_twin_df.get("failure_probability", pd.Series([0])).mean()) * 100, 2),
            round(float(places_df.get("pm2_5", pd.Series([0])).mean()), 2),
            round(float(places_df.get("wind_speed_10m", pd.Series([0])).mean()), 2),
            round(float(places_df.get("shortwave_radiation", pd.Series([0])).mean()), 2),
            len(outages_df),
        ],
    })

    st.markdown("### ⏱️ Cascading failure timeline (CReDo style)")
    if not places_df.empty:
        sample_place = places_df.iloc[0].to_dict()
        timeline_df = run_time_simulation(sample_place, outage_intensity=0.3, scenario_engine=scenario_engine)
        st.line_chart(timeline_df.set_index("hour")[["risk"]])

    st.markdown("### 🔬 Uncertainty (Monte Carlo)")
    for _, r in places_df.iterrows():
        st.write(
            f"{r.get('place', 'Unknown')} → "
            f"σ={round(safe_float(r.get('risk_std')) or 0, 2)} | "
            f"P95={round(safe_float(r.get('risk_p95')) or 0, 1)}"
        )

    st.dataframe(twin_summary, use_container_width=True, hide_index=True)


# =========================================================
# TAB 3
# =========================================================
with tab3:
    st.subheader(f"{selected_place} live and forecast diagnostics")

    lc1, lc2 = st.columns([1.1, 1])

    with lc1:
        st.markdown("### Current conditions")
        st.json({
            "time": selected_current.get("time"),
            "temperature_2m": selected_current.get("temperature_2m"),
            "apparent_temperature": selected_current.get("apparent_temperature"),
            "wind_speed_10m": selected_current.get("wind_speed_10m"),
            "wind_direction_10m": selected_current.get("wind_direction_10m"),
            "cloud_cover": selected_current.get("cloud_cover"),
            "precipitation": selected_current.get("precipitation"),
            "shortwave_radiation": selected_current.get("shortwave_radiation"),
            "direct_radiation": selected_current.get("direct_radiation"),
            "diffuse_radiation": selected_current.get("diffuse_radiation"),
            "is_day": selected_current.get("is_day"),
            "AQI": selected_current.get("european_aqi"),
            "pm2_5": selected_current.get("pm2_5"),
            "pm10": selected_current.get("pm10"),
            "NO2": selected_current.get("nitrogen_dioxide"),
            "O3": selected_current.get("ozone"),
            "AOD": selected_current.get("aerosol_optical_depth"),
        })

    with lc2:
        st.markdown("### 48-hour predictive outlook")
        if not hourly_df.empty:
            preview_cols = [
                c for c in [
                    "time", "temperature_2m", "wind_speed_10m", "cloud_cover",
                    "precipitation", "shortwave_radiation", "european_aqi",
                    "pm2_5", "predicted_risk_score",
                ]
                if c in hourly_df.columns
            ]
            st.dataframe(hourly_df[preview_cols].head(24), use_container_width=True, height=360)
        else:
            st.warning("No hourly forecast available.")


# =========================================================
# TAB 4
# =========================================================
with tab4:
    st.subheader(f"Northern Powergrid live outage records — {region_name}")

    if outages_df.empty:
        st.warning("No outage records were returned for the current regional filter.")
    else:
        preferred_cols = [
            c for c in [
                "outage_reference", "outage_status", "outage_category",
                "affected_customers", "postcode_label", "estimated_restore",
                "time_label", "latitude", "longitude",
            ] if c in outages_df.columns
        ]
        st.dataframe(
            outages_df[preferred_cols] if preferred_cols else outages_df,
            use_container_width=True,
            height=460,
        )


# =========================================================
# TAB 5
# =========================================================
with tab5:
    csv = digital_twin_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download twin data",
        csv,
        "digital_twin.csv",
        "text/csv",
    )

    st.subheader("Digital twin grid cells")
    st.dataframe(
        digital_twin_df.sort_values("risk_score", ascending=False),
        use_container_width=True,
        height=500,
    )


# =========================================================
# TAB 6
# =========================================================
with tab6:
    st.subheader("CReDo-style cascading failure simulation")

    left_panel, centre_panel, right_panel = st.columns([1.1, 3.2, 1.35])

    with left_panel:
        st.markdown("### Controls")
        sim_steps = st.slider("Simulation steps", 6, 30, 14, 1, key="credo_steps")
        sim_speed = st.slider("Animation delay (seconds)", 0.1, 1.2, 0.45, 0.05, key="credo_speed")
        auto_run = st.checkbox("Run animation", value=True, key="credo_autorun")
        show_failure_waves = st.checkbox("Show failure waves", value=True, key="credo_waves")
        show_cross_region_links = st.checkbox("Show cross-place links", value=True, key="credo_crosslinks")
        manual_step = st.slider(
            "Manual step",
            0,
            sim_steps - 1,
            st.session_state.get("credo_last_step", 0),
            1,
            key="credo_manual_step",
        )
        st.caption("Node click opens detailed metadata and syncs the map below.")

    # -----------------------------------------------------
    # GRAPH BUILD
    # -----------------------------------------------------
    G = nx.DiGraph()
    place_lookup = {}
    place_names = list(places_df["place"]) if "place" in places_df.columns else []

    for _, r in places_df.iterrows():
        place = str(r.get("place"))
        row = r.to_dict()
        place_lookup[place] = row
        flood_depth = infer_flood_depth_for_place(row, scenario_choice)

        G.add_node(
            place,
            type="city",
            label=place,
            parent_place=place,
            state=0,
            flood_depth=flood_depth,
            direct_prob=0.0,
            indirect_prob=0.0,
        )

        for asset_type in ["power", "water", "telecom"]:
            node_name = f"{place}_{asset_type}"
            G.add_node(
                node_name,
                type=asset_type,
                label=node_name,
                parent_place=place,
                state=0,
                flood_depth=flood_depth,
                direct_prob=0.0,
                indirect_prob=0.0,
            )

        # local dependencies
        G.add_edge(f"{place}_power", f"{place}_telecom")
        G.add_edge(f"{place}_power", f"{place}_water")
        G.add_edge(f"{place}_telecom", f"{place}_water")
        G.add_edge(f"{place}_power", place)
        G.add_edge(f"{place}_water", place)
        G.add_edge(f"{place}_telecom", place)

    if show_cross_region_links:
        for i in range(len(place_names) - 1):
            a = place_names[i]
            b = place_names[i + 1]
            G.add_edge(f"{a}_power", f"{b}_power")
            G.add_edge(f"{a}_telecom", f"{b}_telecom")
            G.add_edge(f"{a}_water", f"{b}_water")

    pos = build_credo_layout(place_names)

    # -----------------------------------------------------
    # INITIAL SEEDING
    # -----------------------------------------------------
    for place, row in place_lookup.items():
        flood_depth = infer_flood_depth_for_place(row, scenario_choice)
        outages = safe_float(row.get("nearby_outages_25km")) or 0.0
        p_fail = direct_failure_probability("power", row, flood_depth, t=0)

        if outages > 3:
            p_fail = clamp(p_fail + 0.20, 0.0, 0.95)

        seed_rand = random.random()
        if seed_rand < p_fail * 0.75:
            G.nodes[f"{place}_power"]["state"] = 2
        elif seed_rand < p_fail:
            G.nodes[f"{place}_power"]["state"] = 1

    selected_node_name = st.session_state.get(
        "credo_selected_node",
        place_names[0] if place_names else None
    )

    # -----------------------------------------------------
    # FIGURE BUILDER
    # -----------------------------------------------------
    def make_figure(
        graph_obj: nx.DiGraph,
        step_idx: int,
        failed_wave_nodes: List[str],
        selected_node_name: Optional[str],
    ):
        # ---------------- edge trace
        edge_x, edge_y = [], []
        for src, dst in graph_obj.edges():
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            hoverinfo="none",
            line=dict(width=1.15, color="rgba(180,180,180,0.24)"),
            showlegend=False,
        )

        # ---------------- node collections
        node_records = []
        glow_traces = []

        for node_name, data in graph_obj.nodes(data=True):
            x, y = pos[node_name]
            node_type = data.get("type", "city")
            state = int(data.get("state", 0))
            parent_place = data.get("parent_place", node_name)

            row = place_lookup.get(parent_place, {})
            wind = safe_float(row.get("wind_speed_10m")) or 0.0
            rain = safe_float(row.get("precipitation")) or 0.0
            outages = safe_float(row.get("nearby_outages_25km")) or 0.0
            aqi = safe_float(row.get("european_aqi")) or 0.0

            is_selected = selected_node_name == node_name

            base_size = 20 if node_type == "city" else 16 if node_type == "power" else 13
            if state == 2:
                base_size += 4
            elif state == 1:
                base_size += 2
            if is_selected:
                base_size += 7

            label_text = f"{node_icon(node_type)} {node_name.replace('_', ' ')}"

            hover_text = "<br>".join([
                f"<b>{node_name}</b>",
                f"Type: {node_type}",
                f"State: {['Normal', 'Stressed', 'Failed'][state]}",
                f"Wind: {wind:.1f} km/h",
                f"Rain: {rain:.1f} mm",
                f"AQI: {aqi:.1f}",
                f"Nearby outages: {outages:.0f}",
                f"Flood depth proxy: {data.get('flood_depth', 0.0):.2f} m",
                f"Direct failure p: {data.get('direct_prob', 0.0):.3f}",
                f"Dependency failure p: {data.get('indirect_prob', 0.0):.3f}",
            ])

            node_records.append({
                "name": node_name,
                "x": x,
                "y": y,
                "type": node_type,
                "state": state,
                "size": base_size,
                "symbol": node_marker_symbol(node_type),
                "fill": node_display_colour(node_type, state),
                "text": label_text,
                "hover": hover_text,
                "selected": is_selected,
            })

            # failed red glow
            if state == 2:
                glow_traces.append(
                    go.Scatter(
                        x=[x],
                        y=[y],
                        mode="markers",
                        marker=dict(
                            size=base_size + 24,
                            color="rgba(255,59,48,0.14)",
                            line=dict(width=2, color="rgba(255,59,48,0.38)"),
                        ),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )

            # wave glow
            if show_failure_waves and node_name in failed_wave_nodes:
                glow_traces.append(
                    go.Scatter(
                        x=[x],
                        y=[y],
                        mode="markers",
                        marker=dict(
                            size=base_size + 36,
                            color="rgba(255,120,120,0.07)",
                            line=dict(width=1, color="rgba(255,120,120,0.24)"),
                        ),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )

            # selected node halo
            if is_selected:
                glow_traces.append(
                    go.Scatter(
                        x=[x],
                        y=[y],
                        mode="markers",
                        marker=dict(
                            size=base_size + 16,
                            color="rgba(255,255,255,0.05)",
                            line=dict(width=2, color="rgba(255,255,255,0.24)"),
                        ),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )

        def build_state_trace(records_subset, line_width, line_color, opacity_value, trace_name):
            if not records_subset:
                return None

            return go.Scatter(
                x=[r["x"] for r in records_subset],
                y=[r["y"] for r in records_subset],
                mode="markers+text",
                text=[r["text"] for r in records_subset],
                textposition="top center",
                hoverinfo="text",
                hovertext=[r["hover"] for r in records_subset],
                customdata=[r["name"] for r in records_subset],
                marker=dict(
                    size=[r["size"] for r in records_subset],
                    color=[r["fill"] for r in records_subset],
                    symbol=[r["symbol"] for r in records_subset],
                    opacity=opacity_value,
                    line=dict(width=line_width, color=line_color),
                ),
                textfont=dict(size=12, color="white"),
                name=trace_name,
                showlegend=False,
            )

        normal_records = [r for r in node_records if r["state"] == 0]
        stressed_records = [r for r in node_records if r["state"] == 1]
        failed_records = [r for r in node_records if r["state"] == 2]

        normal_trace = build_state_trace(
            normal_records,
            line_width=1.4,
            line_color="rgba(255,255,255,0.85)",
            opacity_value=1.0,
            trace_name="Normal",
        )

        stressed_trace = build_state_trace(
            stressed_records,
            line_width=2.3,
            line_color="#ffcc00",
            opacity_value=0.72,
            trace_name="Stressed",
        )

        failed_trace = build_state_trace(
            failed_records,
            line_width=3.0,
            line_color="#ff3b30",
            opacity_value=1.0,
            trace_name="Failed",
        )

        traces = [edge_trace] + glow_traces
        for tr in [normal_trace, stressed_trace, failed_trace]:
            if tr is not None:
                traces.append(tr)

        fig = go.Figure(data=traces)

        fig.update_layout(
            title=f"⚡ CReDo Cascade Simulation — Step {step_idx}",
            showlegend=False,
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="white"),
            margin=dict(l=10, r=10, t=48, b=10),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            clickmode="event+select",
        )

        return fig

    # -----------------------------------------------------
    # PRECOMPUTE STEP HISTORY
    # -----------------------------------------------------
    history = []
    wave_history = []
    sim_graph = G.copy()

    for t in range(sim_steps):
        next_states = {}
        failed_wave_nodes = []

        for node_name, data in sim_graph.nodes(data=True):
            node_type = data.get("type", "city")
            place = data.get("parent_place", node_name)
            row = place_lookup.get(place, {})
            flood_depth = data.get("flood_depth", 0.0)
            current_state = int(data.get("state", 0))

            d_prob = direct_failure_probability(node_type, row, flood_depth, t)
            sim_graph.nodes[node_name]["direct_prob"] = d_prob

            incoming_neighbors = list(sim_graph.predecessors(node_name))
            failed_upstream = sum(sim_graph.nodes[n].get("state", 0) == 2 for n in incoming_neighbors)
            stressed_upstream = sum(sim_graph.nodes[n].get("state", 0) == 1 for n in incoming_neighbors)

            if node_type == "city":
                if failed_upstream >= 2:
                    i_prob = 0.55
                elif failed_upstream == 1 or stressed_upstream >= 2:
                    i_prob = 0.28
                elif stressed_upstream == 1:
                    i_prob = 0.10
                else:
                    i_prob = 0.02
            else:
                total_in = max(len(incoming_neighbors), 1)
                i_prob = (failed_upstream * 1.0 + stressed_upstream * 0.45) / total_in

                if node_type == "water":
                    i_prob *= 1.35
                elif node_type == "telecom":
                    i_prob *= 1.15
                elif node_type == "power":
                    i_prob *= 1.05

            i_prob = clamp(i_prob, 0.0, 0.95)
            sim_graph.nodes[node_name]["indirect_prob"] = i_prob

            combined_fail_prob = clamp(0.55 * d_prob + 0.45 * i_prob, 0.0, 0.75)
            recover_prob = clamp(recovery_probability(node_type, flood_depth, t) * 1.5, 0.0, 0.45)

            if current_state == 2:
                if random.random() < recover_prob:
                    next_states[node_name] = 1
                else:
                    next_states[node_name] = 2
                continue

            if current_state == 1:
                escalate_prob = clamp(combined_fail_prob + 0.10, 0.0, 0.90)
                recover_to_normal_prob = clamp(recover_prob * 0.75, 0.0, 0.55)

                u = random.random()
                if u < escalate_prob:
                    next_states[node_name] = 2
                    failed_wave_nodes.append(node_name)
                elif u < escalate_prob + recover_to_normal_prob:
                    next_states[node_name] = 0
                else:
                    next_states[node_name] = 1
                continue

            if node_type == "city":
                fail_prob = combined_fail_prob * 0.25
                stress_prob = combined_fail_prob * 0.65
            else:
                fail_prob = combined_fail_prob * 0.55
                stress_prob = combined_fail_prob * 0.90

            u = random.random()
            if u < fail_prob:
                next_states[node_name] = 2
                failed_wave_nodes.append(node_name)
            elif u < fail_prob + stress_prob:
                next_states[node_name] = 1
            else:
                next_states[node_name] = 0

        for node_name, state in next_states.items():
            sim_graph.nodes[node_name]["state"] = state

        history.append(sim_graph.copy())
        wave_history.append(list(failed_wave_nodes))

    # -----------------------------------------------------
    # PLAYBACK
    # -----------------------------------------------------
    if auto_run:
        active_step = sim_steps - 1

        for t in range(sim_steps):
            current_graph = history[t]
            fig = make_figure(current_graph, t, wave_history[t], selected_node_name)

            with centre_panel:
                clicked = plotly_events(
                    fig,
                    click_event=True,
                    select_event=False,
                    hover_event=False,
                    override_height=640,
                    key=f"credo_plot_autoplay_{t}",
                )

            if clicked:
                point_index = clicked[0].get("pointIndex")
                if point_index is not None:
                    try:
                        selected_trace = fig.data[-1]
                        clicked_node = selected_trace.customdata[point_index]
                        if clicked_node in current_graph.nodes:
                            selected_node_name = clicked_node
                            st.session_state["credo_selected_node"] = clicked_node
                    except Exception:
                        pass

            st.session_state["credo_last_step"] = t
            active_step = t
            time.sleep(sim_speed)

    else:
        active_step = manual_step
        st.session_state["credo_last_step"] = active_step

        current_graph = history[active_step]
        fig = make_figure(current_graph, active_step, wave_history[active_step], selected_node_name)

        with centre_panel:
            clicked = plotly_events(
                fig,
                click_event=True,
                select_event=False,
                hover_event=False,
                override_height=640,
                key=f"credo_plot_manual_{active_step}",
            )

        if clicked:
            point_index = clicked[0].get("pointIndex")
            if point_index is not None:
                try:
                    # trace sonuncu olmak zorunda değil, customdata olan son trace'i bul
                    clicked_node = None
                    for trace in reversed(fig.data):
                        if hasattr(trace, "customdata") and trace.customdata is not None:
                            if point_index < len(trace.customdata):
                                clicked_node = trace.customdata[point_index]
                                break

                    if clicked_node and clicked_node in current_graph.nodes:
                        selected_node_name = clicked_node
                        st.session_state["credo_selected_node"] = clicked_node
                except Exception:
                    pass

    # -----------------------------------------------------
    # ACTIVE GRAPH + METADATA
    # -----------------------------------------------------
    active_graph = history[active_step] if history else G

    if not selected_node_name and place_names:
        selected_node_name = place_names[0]

    selected_place_name = get_place_from_node(selected_node_name) if selected_node_name else None
    selected_data = active_graph.nodes[selected_node_name] if selected_node_name in active_graph.nodes else {}

    with right_panel:
        st.markdown("### Asset metadata")

        if selected_node_name in active_graph.nodes:
            row = place_lookup.get(selected_place_name, {})
            node_state = int(selected_data.get("state", 0))

            st.markdown(
                f"""
                **Selected node**  
                {selected_node_name}

                **Type**  
                {selected_data.get('type', 'Unknown')}

                **State**  
                {['Normal', 'Stressed', 'Failed'][node_state]}

                **Flood depth proxy**  
                {selected_data.get('flood_depth', 0.0):.2f} m

                **Direct failure probability**  
                {selected_data.get('direct_prob', 0.0):.3f}

                **Dependency failure probability**  
                {selected_data.get('indirect_prob', 0.0):.3f}

                **Wind**  
                {safe_float(row.get('wind_speed_10m')) or 0:.1f} km/h

                **Rain**  
                {safe_float(row.get('precipitation')) or 0:.1f} mm

                **AQI**  
                {safe_float(row.get('european_aqi')) or 0:.1f}

                **Nearby outages**  
                {safe_float(row.get('nearby_outages_25km')) or 0:.0f}
                """
            )

            st.markdown("### 🎨 Legend")

            st.markdown("""
            **Infrastructure type (fill colour)**  
            - 🟡 Power  
            - 🟣 Telecom  
            - 🔵 Water  
            - 🟢 City  

            **System state (outline / style)**  
            - ⚪ White outline = Normal  
            - 🟡 Amber outline + reduced opacity = Stressed  
            - 🔴 Red outline + red glow = Failed  

            **Visual rules**  
            - Fill colour shows infrastructure class  
            - Border colour shows operational state  
            - Larger node = higher importance / stronger emphasis  
            - Outer glow = active failure or selected focus
            """)

            st.markdown("### Interpretation")
            st.info(
                "The network visualisation follows a cascading failure logic inspired by CReDo-style interdependent infrastructure modelling. "
                "Power disruptions can propagate into telecom, water and city-demand nodes through dependency chains."
            )

            if node_state == 2:
                st.error("This asset is currently failed in the simulation.")
            elif node_state == 1:
                st.warning("This asset is currently under stress and may escalate.")
            else:
                st.success("This asset is currently operating in a stable state.")

    # -----------------------------------------------------
    # SYNCED MAP
    # -----------------------------------------------------
    st.markdown("### Synced flood and outage map")

    synced_map = build_synced_credo_map(
        region_name=region_name,
        selected_place_name=selected_place_name,
        places_df=places_df,
        outages_df=outages_df,
        scenario_choice=scenario_choice,
    )
    components.html(synced_map._repr_html_(), height=640)


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    """
    **System notes**
    - The app refreshes every 30 seconds.
    - Solar radiation values close to zero are physically expected during night-time conditions.
    - The digital twin risk layer is a predictive screening layer, not a formal operator-grade protection model.
    - Regional outage matching is robust but still depends on the live schema exposed by the Northern Powergrid dataset.
    - The flood layer used in the CReDo network tab is currently a proxy depth layer. Replace it with a real raster or model output for higher-fidelity flood analytics.
    """
)

# =========================================================
# SATELLITE STORYBOARD HELPERS
# =========================================================
def get_story_satellite_url(layer_mode: str, date_str: str) -> str:
    """
    Returns a dated GIBS tile URL for the selected storytelling mode.
    """
    layer_catalog = {
        "True Colour (MODIS Terra)": (
            "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
            "MODIS_Terra_CorrectedReflectance_TrueColor/default/"
            f"{date_str}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg"
        ),
        "True Colour (VIIRS NOAA-20)": (
            "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
            "VIIRS_NOAA20_CorrectedReflectance_TrueColor/default/"
            f"{date_str}/GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg"
        ),
        "Night Lights (NOAA-20 DNB Radiance)": (
            "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
            "VIIRS_NOAA20_DayNightBand_At_Sensor_Radiance/default/"
            f"{date_str}/GoogleMapsCompatible_Level8/{{z}}/{{y}}/{{x}}.png"
        ),
        # legacy fallback if needed
        "Night Lights (Legacy Black Marble)": (
            "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
            "BlackMarble_2016/default/"
            f"{date_str}/GoogleMapsCompatible_Level8/{{z}}/{{y}}/{{x}}.png"
        ),
    }
    return layer_catalog[layer_mode]

from datetime import datetime, timedelta

def get_story_phase_dates():
    today = datetime.utcnow().date()

    return {
        "pre": (today - timedelta(days=5)).strftime("%Y-%m-%d"),
        "during": (today - timedelta(days=4)).strftime("%Y-%m-%d"),
        "peak": (today - timedelta(days=3)).strftime("%Y-%m-%d"),
        "recovery": (today - timedelta(days=2)).strftime("%Y-%m-%d"),
    }

def create_satellite_story_map(
    region_name: str,
    phase_title: str,
    layer_mode: str,
    date_str: str,
    places_df: pd.DataFrame,
    outages_df: pd.DataFrame,
    scenario_choice: str,
    show_places: bool = True,
    show_outages: bool = True,
    show_risk_overlay: bool = True,
):
    center = REGIONS[region_name]["center"]
    bbox = REGIONS[region_name]["bbox"]  # [min_lon, min_lat, max_lon, max_lat]
    
    m = folium.Map(
        location=[center["lat"], center["lon"]],
        zoom_start=max(int(center["zoom"]), 6),
        tiles="CartoDB dark_matter",
        control_scale=True,
        prefer_canvas=True,
    )

    m.fit_bounds([
        [bbox[1], bbox[0]],
        [bbox[3], bbox[2]]
    ])
    

    # Fit tightly to the selected region so Newcastle/Yorkshire is actually visible
    m.fit_bounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]])

    # base satellite layer
    tile_url = safe_tile_url(layer_mode, date_str)
    folium.raster_layers.TileLayer(
        tiles=tile_url,
        attr="NASA GIBS",
        name=f"{layer_mode} | {date_str}",
        overlay=True,
        control=False,
        opacity=0.92,
    ).add_to(m)

    # optional region boundary
    region_polygon_latlon = [[lat, lon] for lon, lat in REGIONS[region_name]["polygon"]]
    folium.Polygon(
        locations=region_polygon_latlon,
        color="#ffb000",
        weight=2,
        fill=True,
        fill_opacity=0.03,
        popup=f"{region_name} boundary",
    ).add_to(m)

    # risk / flood proxy overlay by scenario phase
    if show_risk_overlay and not places_df.empty:
        for _, r in places_df.iterrows():
            lat = safe_float(r.get("lat"))
            lon = safe_float(r.get("lon"))
            if lat is None or lon is None:
                continue

            flood_depth = infer_flood_depth_for_place(r.to_dict(), scenario_choice)
            risk_score = safe_float(r.get("risk_score")) or 0

            # make phases visually different
            phase_multiplier = {
                "Pre-event": 0.45,
                "Disturbance": 0.85,
                "Peak failure": 1.20,
                "Recovery": 0.60,
            }.get(phase_title, 0.80)

            adj_depth = flood_depth * phase_multiplier
            adj_risk = risk_score * phase_multiplier

            if adj_depth > 0.05 or adj_risk > 35:
                folium.Circle(
                    location=[lat, lon],
                    radius=3000 + adj_depth * 9000 + adj_risk * 18,
                    color="#53b7ff",
                    weight=1,
                    fill=True,
                    fill_color="#53b7ff",
                    fill_opacity=min(0.05 + adj_depth * 0.08, 0.28),
                    popup=folium.Popup(
                        f"""
                        <b>{r.get('place')}</b><br>
                        Phase: {phase_title}<br>
                        Date: {date_str}<br>
                        Flood-depth proxy: {adj_depth:.2f} m<br>
                        Risk score proxy: {adj_risk:.1f}
                        """,
                        max_width=300,
                    ),
                ).add_to(m)

    if show_places and not places_df.empty:
        for _, r in places_df.iterrows():
            lat = safe_float(r.get("lat"))
            lon = safe_float(r.get("lon"))
            if lat is None or lon is None:
                continue

            risk = safe_float(r.get("risk_score")) or 0
            if phase_title == "Pre-event":
                risk *= 0.55
            elif phase_title == "Disturbance":
                risk *= 0.85
            elif phase_title == "Peak failure":
                risk *= 1.10
            elif phase_title == "Recovery":
                risk *= 0.70

            if risk >= 75:
                colour = "darkred"
            elif risk >= 55:
                colour = "red"
            elif risk >= 35:
                colour = "orange"
            else:
                colour = "green"

            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color="white",
                weight=1.5,
                fill=True,
                fill_color=colour,
                fill_opacity=0.95,
                tooltip=f"{r.get('place')} | {phase_title}",
                popup=folium.Popup(
                    f"""
                    <b>{r.get('place')}</b><br>
                    Phase: {phase_title}<br>
                    Date: {date_str}<br>
                    Wind: {safe_float(r.get('wind_speed_10m')) or 0:.1f} km/h<br>
                    Rain: {safe_float(r.get('precipitation')) or 0:.1f} mm<br>
                    AQI: {safe_float(r.get('european_aqi')) or 0:.1f}<br>
                    Risk score: {risk:.1f}<br>
                    Nearby outages: {safe_float(r.get('nearby_outages_25km')) or 0:.0f}
                    """,
                    max_width=320,
                ),
            ).add_to(m)

    if show_outages and not outages_df.empty and phase_title in ["Disturbance", "Peak failure", "Recovery"]:
        for _, r in outages_df.iterrows():
            lat = safe_float(r.get("latitude"))
            lon = safe_float(r.get("longitude"))
            if lat is None or lon is None:
                continue

            folium.Marker(
                location=[lat, lon],
                tooltip=f"Outage | {r.get('outage_status', 'Unknown')}",
                popup=folium.Popup(
                    f"""
                    <b>Power outage</b><br>
                    Phase: {phase_title}<br>
                    Date: {date_str}<br>
                    Reference: {r.get('outage_reference', 'N/A')}<br>
                    Status: {r.get('outage_status', 'Unknown')}<br>
                    Category: {r.get('outage_category', 'Unknown')}<br>
                    Customers affected: {r.get('affected_customers', '')}<br>
                    Estimated restore: {r.get('estimated_restore', '')}
                    """,
                    max_width=340,
                ),
                icon=folium.Icon(color="red", icon="flash"),
            ).add_to(m)

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 18px;
        left: 18px;
        z-index: 9999;
        background-color: rgba(8,12,20,0.88);
        color: white;
        padding: 10px 12px;
        border-radius: 8px;
        font-size: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35);
    ">
        <b>{phase_title}</b><br>
        Date: {date_str}<br>
        Layer: {layer_mode}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m
# =========================================================
# TAB 7 - SATELLITE ANALYTICS (IMPROVED)
# =========================================================
with tab7:
    st.subheader("🛰️ Advanced Satellite Storyboard for Newcastle & Yorkshire")

    st.markdown(
        "This panel is designed to look more like a paper figure: pre-event, disturbance, peak failure, and recovery."
    )

    story_left, story_right = st.columns([1.15, 1])

    with story_left:
        layer_mode = st.selectbox(
            "Satellite visualisation mode",
            [
                "True Colour (VIIRS NOAA-20)",
                "True Colour (MODIS Terra)",
                "Night Lights (NOAA-20 DNB Radiance)",
                "Night Lights (Legacy Black Marble)",
            ],
            index=0,
            key="sat_story_layer_mode"
        )

        use_fixed_story_dates = st.checkbox(
            "Use recommended event-style dates",
            value=True,
            key="sat_story_fixed_dates"
        )

        show_places = st.checkbox("Show place markers", value=True, key="sat_story_places")
        show_outages = st.checkbox("Show outage markers", value=True, key="sat_story_outages")
        show_risk_overlay = st.checkbox("Show hazard/risk overlay", value=True, key="sat_story_risk")

    with story_right:
        st.markdown("### Notes")
        st.caption(
            "For a cleaner UK-focused result, this figure now fits the selected regional bounds instead of opening at a generic global view."
        )
        st.caption(
            "For event-style storytelling, true-colour layers usually look better for clouds/flood context; night-lights layers are better for blackout/recovery narratives."
        )

    fixed_dates = get_story_phase_dates()

    if use_fixed_story_dates:
        pre_date = datetime.strptime(fixed_dates["pre"], "%Y-%m-%d").date()
        during_date = datetime.strptime(fixed_dates["during"], "%Y-%m-%d").date()
        peak_date = datetime.strptime(fixed_dates["peak"], "%Y-%m-%d").date()
        recovery_date = datetime.strptime(fixed_dates["recovery"], "%Y-%m-%d").date()
        st.info(
            f"Using storyboard dates: pre={pre_date}, disturbance={during_date}, peak={peak_date}, recovery={recovery_date}"
        )
    else:
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        pre_date = dcol1.date_input("Pre-event date", datetime.utcnow(), key="sat_pre_date")
        during_date = dcol2.date_input("Disturbance date", datetime.utcnow(), key="sat_during_date")
        peak_date = dcol3.date_input("Peak failure date", datetime.utcnow(), key="sat_peak_date")
        recovery_date = dcol4.date_input("Recovery date", datetime.utcnow(), key="sat_recovery_date")

    st.markdown("### Figure-style 2 × 2 satellite panel")

    top_left, top_right = st.columns(2)
    bottom_left, bottom_right = st.columns(2)

    with top_left:
        st.markdown("#### (a) Pre-event")
        pre_map = create_satellite_story_map(
            region_name=region_name,
            phase_title="Pre-event",
            layer_mode=layer_mode,
            date_str=pre_date.strftime("%Y-%m-%d"),
            places_df=places_df,
            outages_df=outages_df,
            scenario_choice=scenario_choice,
            show_places=show_places,
            show_outages=False,
            show_risk_overlay=show_risk_overlay,
        )
        components.html(pre_map._repr_html_(), height=360)

    with top_right:
        st.markdown("#### (b) Disturbance")
        during_map = create_satellite_story_map(
            region_name=region_name,
            phase_title="Disturbance",
            layer_mode=layer_mode,
            date_str=during_date.strftime("%Y-%m-%d"),
            places_df=places_df,
            outages_df=outages_df,
            scenario_choice=scenario_choice,
            show_places=show_places,
            show_outages=show_outages,
            show_risk_overlay=show_risk_overlay,
        )
        components.html(during_map._repr_html_(), height=360)

    with bottom_left:
        st.markdown("#### (c) Peak failure")
        peak_map = create_satellite_story_map(
            region_name=region_name,
            phase_title="Peak failure",
            layer_mode=layer_mode,
            date_str=peak_date.strftime("%Y-%m-%d"),
            places_df=places_df,
            outages_df=outages_df,
            scenario_choice=scenario_choice,
            show_places=show_places,
            show_outages=show_outages,
            show_risk_overlay=show_risk_overlay,
        )
        components.html(peak_map._repr_html_(), height=360)

    with bottom_right:
        st.markdown("#### (d) Recovery")
        recovery_map = create_satellite_story_map(
            region_name=region_name,
            phase_title="Recovery",
            layer_mode=layer_mode,
            date_str=recovery_date.strftime("%Y-%m-%d"),
            places_df=places_df,
            outages_df=outages_df,
            scenario_choice=scenario_choice,
            show_places=show_places,
            show_outages=show_outages,
            show_risk_overlay=show_risk_overlay,
        )
        components.html(recovery_map._repr_html_(), height=360)

    st.markdown("---")
    st.markdown("### Figure caption")
    st.markdown(
        f"""
        **Spatiotemporal evolution over {region_name}** showing  
        (a) pre-event baseline,  
        (b) disturbance phase,  
        (c) peak failure conditions, and  
        (d) recovery phase,  
        rendered using **{layer_mode}** with region-focused overlays for outages and risk.
        """
    )

    st.markdown("### Interpretation")
    st.write(
        "If you want a paper-style blackout narrative, use **Night Lights (NOAA-20 DNB Radiance)**. "
        "If you want flood/cloud/context interpretation, use **True Colour (VIIRS NOAA-20)**."
    )
