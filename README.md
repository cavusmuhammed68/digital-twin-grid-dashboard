# North East & Yorkshire Energy Dashboard

This starter project creates a Streamlit dashboard for:

- North East and Yorkshire
- live Northern Powergrid power cut data
- weather, wind, solar radiation and temperature
- air-quality / pollution metrics
- NASA satellite basemaps for MODIS, VIIRS and Black Marble night lights

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Main APIs used

- Northern Powergrid Open Data
- Open-Meteo Weather API
- Open-Meteo Air Quality API
- NASA GIBS satellite tile services

## Next upgrades

- add postcode or polygon filtering for exact service area boundaries
- add historical outage analysis
- add UK station-level pollution feeds
- add LAADS token-based raw file downloader
