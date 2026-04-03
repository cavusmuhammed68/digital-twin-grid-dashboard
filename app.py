import json
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import random

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import urllib3
import folium
from folium.plugins import HeatMap

st.set_page_config(
    page_title="North East & Yorkshire Grid Digital Twin",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
</style>
""",
    unsafe_allow_html=True,
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

session = requests.Session()
session.verify = False
session.headers.update({
    "User-Agent": "north-east-yorkshire-digital-twin/3.0"
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
    return {k: force_scalar(v) if isinstance(v, (pd.Series, list, tuple, np.ndarray, int, float, np.integer, np.floating, str, type(None))) else v for k, v in row.items()}


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


@st.cache_data(ttl=25, show_spinner=False)
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


@st.cache_data(ttl=25, show_spinner=False)
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


@st.cache_data(ttl=25, show_spinner=False)
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

        "storm_cascade": {
            "wind": 2.2,
            "rain": 1.6,
            "temp": -1,
            "infra": 1.8,
        },

        "flood_infrastructure": {
            "wind": 1.3,
            "rain": 3.5,
            "temp": 0,
            "infra": 2.2,
        },

        "heatwave_peak": {
            "wind": 0.6,
            "rain": 0.2,
            "temp": +10,
            "infra": 1.5,
        },

        "pollution_event": {
            "wind": 0.4,
            "rain": 0.1,
            "temp": +5,
            "infra": 1.2,
        },

        "compound_extreme": {
            "wind": 1.8,
            "rain": 2.8,
            "temp": +6,
            "infra": 3.0,
        },
    }
        return scenarios.get(name, scenarios["baseline"])

    def apply(self, row):
        if isinstance(row, pd.Series):
            row = row.to_dict()
        row = dict(row)
        row["wind_speed_10m"] = (safe_float(row.get("wind_speed_10m")) or 0) * self.params["wind"]
        row["precipitation"] = (safe_float(row.get("precipitation")) or 0) * self.params["rain"]
        row["temperature_2m"] = (safe_float(row.get("temperature_2m")) or 0) + self.params["temp"]
        row["shortwave_radiation"] *= (1 - 0.3 * self.params["rain"])
        row["european_aqi"] *= (1 + 0.4 * self.params["infra"])
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

            dependency_failures = [
                system_state.get(dep, 0) for dep in config["depends_on"]
            ]

            if dependency_failures:
                combined = np.mean(dependency_failures)
                system_state[infra] = clamp(
                    (combined ** 1.5) * config["impact_factor"],
                    0,
                    1
                )
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
    system_stress, system_breakdown = graph.propagate_failure(
        base["failure_probability"]
    )
 
    infra_multiplier = scenario_engine.params.get("infra", 1.0)
    scenario_name = scenario_engine.scenario_name

    # baseline vs hazard cascade sensitivity
    cascade_factor = 0.9 if scenario_name == "baseline" else 1.3

    # base cascade interaction
    final_risk = base["risk_score"] * (1 + system_stress * infra_multiplier * cascade_factor)

    # nonlinear disaster amplification (ONLY for hazard scenarios)
    if scenario_name != "baseline":
        severity = infra_multiplier
    
        # exponential amplification (high risk → much stronger boost)
        scenario_boost = (final_risk / 100) ** 1.7 * severity * 30
    
        final_risk += scenario_boost

    # soft cap (avoid flat saturation at 100)
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

def run_time_simulation(row, outage_intensity, scenario_engine, hours=24):
    timeline = []

    for h in range(hours):
        modified = dict(row)

        modified["wind_speed_10m"] *= random.uniform(0.9, 1.2)
        modified["precipitation"] *= random.uniform(0.8, 1.3)

        risk = enhanced_risk_with_cascade(
            modified,
            outage_intensity * (1 + h / 24),
            scenario_engine
        )

        timeline.append({
            "hour": h,
            "risk": risk["final_risk_score"],
            "failure_prob": risk["failure_probability"],
        })

    return pd.DataFrame(timeline)

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

def compound_hazard_index(row):
    wind = safe_float(row.get("wind_speed_10m")) or 0
    rain = safe_float(row.get("precipitation")) or 0
    temp = safe_float(row.get("temperature_2m")) or 0

    return (wind * rain) / 50 + abs(temp - 18) / 10

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

    outage_points = []
    for _, r in outages_df.iterrows():
        lat = safe_float(r.get("latitude"))
        lon = safe_float(r.get("longitude"))
        if lat is not None and lon is not None:
            outage_points.append((lat, lon))

    for place, (lat, lon) in REGIONS[region_name]["places"].items():
        combined = combine_weather_air(place, lat, lon)
        row = combined["current_row"]

        nearby_outages = sum(
            1 for olat, olon in outage_points
            if haversine_km(lat, lon, olat, olon) <= 25
        )

        outage_intensity = clamp(nearby_outages / 20, 0, 1)

        risk = compute_multilayer_risk(row, outage_intensity)
        mc = monte_carlo_risk(row, outage_intensity, simulations=simulations)

        row.update({
            "nearby_outages_25km": nearby_outages,
            **risk,
            **mc,
        })

        rows.append(row)
        raw_cache[place] = combined

    return pd.DataFrame(rows), raw_cache


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


def build_digital_twin_grid(
    region_name: str,
    places_df: pd.DataFrame,
    outages_df: pd.DataFrame,
    scenario_engine=None,
) -> pd.DataFrame:
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
                enhanced = enhanced_risk_with_cascade(
                    base_row,
                    outage_intensity,
                    scenario_engine,
                )
            else:
                enhanced = compute_multilayer_risk(
                    base_row,
                    outage_intensity,
                )
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


st.title("North East & Yorkshire Grid Digital Twin")
st.caption("Live digital twin for weather, pollution, solar conditions, satellite layers, outage monitoring and predictive risk screening.")

with st.sidebar:
    st.header("Controls")
    region_name = st.selectbox("Region", list(REGIONS.keys()), index=0)
    satellite_name = st.selectbox("Satellite layer", list(SATELLITE_LAYERS.keys()), index=0)
    selected_place = st.selectbox("Detailed forecast site", list(REGIONS[region_name]["places"].keys()), index=0)
    outage_limit = st.slider("Maximum live outage records to request", 10, 100, 100, 10)
    risk_filter = st.slider("Minimum risk level", 0, 100, 0)
    scenario_choice = st.selectbox(
        "Simulation scenario",
        [
            "baseline",
            "storm_cascade",
            "flood_infrastructure",
            "heatwave_peak",
            "pollution_event",
            "compound_extreme",
        ],
    )

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

try:
    raw_npg = fetch_npg_live_power_cuts(limit=outage_limit)
    raw_npg_df = payload_to_df(raw_npg)
    outages_df = standardise_outage_df(raw_npg_df, region_name)

    places_df, raw_cache = build_place_dataframe(region_name, outages_df, simulations=mc_runs)
    scenario_engine = ScenarioEngine(scenario_choice)

    enhanced_rows = []
    for _, r in places_df.iterrows():
        enhanced = enhanced_risk_with_cascade(
            r.to_dict(),
            clamp((safe_float(r.get("nearby_outages_25km")) or 0) / 25, 0, 1),
            scenario_engine,
        )
        enhanced_rows.append(enhanced)

    enhanced_df = pd.DataFrame(enhanced_rows)

    enhanced_df = enhanced_df.add_prefix("enh_")

    places_df = pd.concat(
        [places_df.reset_index(drop=True), enhanced_df.reset_index(drop=True)],
        axis=1,
    )
    
    # overwrite with enhanced model outputs
    places_df["risk_score"] = places_df["enh_risk_score"]
    places_df["failure_probability"] = places_df["enh_failure_probability"]
    places_df["net_load"] = places_df["enh_net_load"]
    places_df["renewable_generation"] = places_df["enh_renewable_generation"]

    for col in ["risk_score", "final_risk_score", "failure_probability"]:
        if col in places_df.columns:
            places_df[col] = places_df[col].apply(force_scalar)

    if "risk_score" in places_df.columns:
        places_df["risk_label"] = places_df["risk_score"].apply(get_risk_label)

    if "wind_speed_10m" in places_df.columns:
        places_df["wind_speed_10m"] = pd.to_numeric(places_df["wind_speed_10m"], errors="coerce").fillna(0)
        places_df["wind_speed_10m"] *= (1 + simulated_wind / 100)

    if "shortwave_radiation" not in places_df.columns:
        places_df["shortwave_radiation"] = 0

    places_df["renewable_score"] = (
        pd.to_numeric(places_df["shortwave_radiation"], errors="coerce").fillna(0) * 0.6
        + pd.to_numeric(places_df["wind_speed_10m"], errors="coerce").fillna(0) * 0.4
    )

    digital_twin_df = build_digital_twin_grid(
        region_name,
        places_df,
        outages_df,
        scenario_engine,
    )
    digital_twin_df = digital_twin_df[digital_twin_df["risk_score"] >= risk_filter]

    selected_weather = raw_cache[selected_place]["weather_raw"]
    selected_air = raw_cache[selected_place]["air_raw"]
    selected_current = raw_cache[selected_place]["current_row"]
    hourly_df = build_hourly_dataframe(selected_weather, selected_air)

except Exception as e:
    st.error(f"Data fetch failed: {e}")
    st.stop()

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

# Apply scenario to selected location
scenario_row = scenario_engine.apply(selected_current)

# recompute renewable + load + risk under scenario
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
    delta=round(scenario_wind - (safe_float(selected_current.get("wind_speed_10m")) or 0),1)
)

sc2.metric(
    "Scenario solar",
    f"{round(scenario_solar,1)} W/m²"
)

sc3.metric(
    "Scenario AQI",
    round(scenario_aqi,1)
)

sc4.metric(
    "Scenario renewable",
    scenario_risk["renewable_generation"]
)

sc5.metric(
    "Scenario net load",
    scenario_risk["net_load"]
)

st.markdown("### ⚡ System Status")

if regional_risk >= 70:
    st.error("CRITICAL GRID RISK — Immediate action required")
elif regional_risk >= 55:
    st.warning("Elevated grid stress detected")
elif regional_risk >= 35:
    st.info("Moderate operational conditions")
else:
    st.success("System stable")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Digital Twin Map",
    "Regional Intelligence",
    "Selected Site Forecast",
    "Live Outages",
    "Raw Twin Grid",
])

with tab1:
    st.subheader(f"{region_name} live digital twin map")

    center = REGIONS[region_name]["center"]

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
            name=f"{satellite_name} overlay",
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
        weight=3,
        fill=True,
        fill_color="orange",
        fill_opacity=0.06,
        popup=f"{region_name} region boundary",
    ).add_to(m)

    if not digital_twin_df.empty:
        twin_df = digital_twin_df.copy()

        if "risk_score" in twin_df.columns:
            max_risk = twin_df["risk_score"].max()
            max_risk = max_risk if pd.notna(max_risk) and max_risk > 0 else 1.0
            twin_df["risk_weight"] = twin_df["risk_score"] / max_risk
        else:
            twin_df["risk_weight"] = 0.2

        heat_data = []
        for _, row in twin_df.iterrows():
            lat = safe_float(row.get("lat"))
            lon = safe_float(row.get("lon"))
            weight = safe_float(row.get("risk_weight"))
            if lat is not None and lon is not None and weight is not None:
                heat_data.append([lat, lon, weight])

        if heat_data:
            HeatMap(
                heat_data,
                name="Digital twin risk heatmap",
                min_opacity=0.25,
                radius=28,
                blur=20,
                max_zoom=12,
            ).add_to(m)

    if not digital_twin_df.empty:
        twin_df = digital_twin_df.copy()
        for _, row in twin_df.iterrows():
            row = row.to_dict()

            lat = safe_float(row.get("lat"))
            lon = safe_float(row.get("lon"))
            if lat is None or lon is None:
                continue

            risk_score = safe_float(row.get("final_risk_score")) or 0
            risk_label = row.get("risk_label", "Unknown")
            wind = safe_float(row.get("wind_speed_10m"))
            aqi = safe_float(row.get("aqi"))
            solar = safe_float(row.get("solar"))
            outages_near = row.get("outages_near_20km", 0)

            if risk_score >= 75:
                colour = "darkred"
            elif risk_score >= 55:
                colour = "red"
            elif risk_score >= 35:
                colour = "orange"
            else:
                colour = "green"

            stress = safe_float(row.get("system_stress")) or 0

            if stress > 0.7:
                colour = "darkred"
            elif stress > 0.5:
                colour = "red"
            elif stress > 0.3:
                colour = "orange"
            else:
                colour = colour

            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                color=colour,
                weight=2,
                fill=True,
                fill_color=colour,
                fill_opacity=0.95,
                popup=folium.Popup(
                    f"""
                    <b>Digital Twin Cell</b><br>
                    Risk score: {risk_score}<br>
                    System stress: {round(stress, 2)}<br>
                    Risk label: {risk_label}<br>
                    Wind: {wind}<br>
                    AQI: {aqi}<br>
                    Solar: {solar}<br>
                    Nearby outages (20 km): {outages_near}
                    """,
                    max_width=320,
                ),
            ).add_to(m)

    if not places_df.empty:
        place_map_df = places_df.copy()
        for _, row in place_map_df.iterrows():
            row = row.to_dict()
            lat = safe_float(row.get("lat"))
            lon = safe_float(row.get("lon"))
            if lat is None or lon is None:
                continue

            risk_score = safe_float(row.get("final_risk_score")) or 0
            failure_prob = safe_float(row.get("failure_probability")) or 0
            wind = safe_float(row.get("wind_speed_10m"))
            aqi = safe_float(row.get("european_aqi"))
            solar = safe_float(row.get("shortwave_radiation"))
            nearby_outages = row.get("nearby_outages_25km", 0)
            place = row.get("place", "Unknown place")

            if risk_score >= 75:
                colour = "darkred"
            elif risk_score >= 55:
                colour = "red"
            elif risk_score >= 35:
                colour = "orange"
            else:
                colour = "green"

            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                color="white",
                weight=2,
                fill=True,
                fill_color=colour,
                fill_opacity=0.95,
                popup=folium.Popup(
                    f"""
                    <b>{place}</b><br>
                    Risk score: {risk_score}<br>
                    Failure probability: {round(failure_prob * 100, 1)}%<br>
                    Wind speed: {wind} km/h<br>
                    AQI: {aqi}<br>
                    Solar radiation: {solar} W/m²<br>
                    Nearby outages (25 km): {nearby_outages}
                    """,
                    max_width=320,
                ),
                tooltip=f"{place} | Risk {risk_score}",
            ).add_to(m)

    outage_points_df = pd.DataFrame()
    if (
        not outages_df.empty
        and "latitude" in outages_df.columns
        and "longitude" in outages_df.columns
    ):
        outage_points_df = outages_df.copy()
        outage_points_df["latitude"] = pd.to_numeric(outage_points_df["latitude"], errors="coerce")
        outage_points_df["longitude"] = pd.to_numeric(outage_points_df["longitude"], errors="coerce")
        outage_points_df = outage_points_df.dropna(subset=["latitude", "longitude"]).copy()

    if not outage_points_df.empty:
        outage_layer = folium.FeatureGroup(name="Live outages", show=True)

        for _, row in outage_points_df.iterrows():
            lat = safe_float(row.get("latitude"))
            lon = safe_float(row.get("longitude"))
            if lat is None or lon is None:
                continue

            popup_html = f"""
            <b>Power outage</b><br>
            Reference: {row.get('outage_reference', 'N/A')}<br>
            Status: {row.get('outage_status', 'Unknown')}<br>
            Category: {row.get('outage_category', 'Unknown')}<br>
            Customers affected: {row.get('affected_customers', '')}<br>
            Postcode: {row.get('postcode_label', '')}<br>
            Estimated restore: {row.get('estimated_restore', '')}
            """

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=340),
                tooltip=f"Outage | {row.get('outage_status', 'Unknown')}",
                icon=folium.Icon(color="red", icon="flash", prefix="glyphicon"),
            ).add_to(outage_layer)

        outage_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background-color: rgba(20, 20, 20, 0.88);
        color: white;
        padding: 12px 14px;
        border-radius: 8px;
        font-size: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35);
    ">
        <b>Digital Twin Risk</b><br>
        <span style="color:green;">●</span> Low<br>
        <span style="color:orange;">●</span> Moderate<br>
        <span style="color:red;">●</span> High<br>
        <span style="color:darkred;">●</span> Severe<br>
        <span style="color:#ff4d4d;">⬤</span> Live outage marker
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    st.caption("Click map markers to inspect local grid conditions")
    components.html(m._repr_html_(), height=700)

    c1, c2, c3 = st.columns(3)

    with c1:
        risky_cells = (
            digital_twin_df[digital_twin_df["risk_score"] >= 55]
            if not digital_twin_df.empty and "risk_score" in digital_twin_df.columns
            else pd.DataFrame()
        )
        st.metric("High-risk twin cells", len(risky_cells))

    with c2:
        severe_cells = (
            digital_twin_df[digital_twin_df["risk_score"] >= 75]
            if not digital_twin_df.empty and "risk_score" in digital_twin_df.columns
            else pd.DataFrame()
        )
        st.metric("Severe-risk twin cells", len(severe_cells))

    with c3:
        if not places_df.empty and "risk_score" in places_df.columns:
            worst_place = places_df.sort_values("risk_score", ascending=False).iloc[0]["place"]
            st.metric("Highest-risk place", worst_place)
        else:
            st.metric("Highest-risk place", "N/A")

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

        st.dataframe(
            df_safe,
            use_container_width=True,
            height=320,
        )

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

        st.metric(
            "Regional renewable potential",
            round(float(places_df["renewable_score"].mean()), 1),
        )

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

        timeline_df = run_time_simulation(
            sample_place,
            outage_intensity=0.3,
            scenario_engine=scenario_engine
        )

        st.line_chart(timeline_df.set_index("hour")[["risk"]])    
    st.markdown("### 🔬 Uncertainty (Monte Carlo)")

    for _, r in places_df.iterrows():
        st.write(
            f"{r.get('place', 'Unknown')} → "
            f"σ={round(safe_float(r.get('risk_std')) or 0, 2)} | "
            f"P95={round(safe_float(r.get('risk_p95')) or 0, 1)}"
        )

    st.dataframe(twin_summary, use_container_width=True, hide_index=True)

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

st.markdown("---")
st.markdown(
    """
    **System notes**
    - The app refreshes every 30 seconds.
    - Solar radiation values close to zero are physically expected during night-time conditions.
    - The digital twin risk layer is a predictive screening layer, not a formal operator-grade protection model.
    - Regional outage matching is robust but still depends on the live schema exposed by the Northern Powergrid dataset.
    """
)
