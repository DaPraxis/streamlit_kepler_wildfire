import time
import json
import os
import streamlit as st
from keplergl import keplergl
import geopandas as gpd
import numpy as np
from shapely.geometry import mapping
import plotly.graph_objects as go
import rasterio
import pandas as pd

# Mapbox
MY_TOKEN = "pk.eyJ1IjoibWF4ZG9taW5pYyIsImEiOiJjbWhyejVvY2owMmNsMmtwdmEwNHd3YjRmIn0.4mJfybYpE2oWZc7iy1hiHA"

st.set_page_config(page_title="Wildfire Progression Viewer", layout="wide")

if "datasets" not in st.session_state:
    st.session_state.datasets = []

# Sidebar
st.sidebar.title("Wildfire Selection")

yyyy = [2018, 2019]

select_yyyy = st.sidebar.selectbox("Select the Year", yyyy)

weather_dfs = {}
for yy in yyyy:
    fd = f"data/{yy}"
    weather_dfs[yy] = pd.read_csv(os.path.join(fd, f"fire_daily_weather_{yy}.csv"))

FIRE_DIR = f"data/{select_yyyy}"
fire_files = []
for f in os.listdir(FIRE_DIR):
    if '.json' in f:
        fire_files.append(f.split(".json")[0])
fire_files = sorted(fire_files)

selected_fire = st.sidebar.selectbox("Select Fire ID", fire_files)

# Meta data for day fire

meta_data = ""

placeholder = st.empty()
st.dataframe(weather_dfs[select_yyyy][weather_dfs[select_yyyy]['fire_id']==int(selected_fire.split('_')[1])].drop(['fire_id', 'year', 'jd'],axis=1))



col_map, col_viz = st.columns([1.5, 1]) 

if selected_fire:

    # ---------------------------
    # Kepler fire polygons (UNCHANGED)
    # ---------------------------
    with open(os.path.join(FIRE_DIR, f"{selected_fire}.json"), "r") as f:
        fire_json = json.load(f)

    fire_gdf = gpd.GeoDataFrame.from_features(fire_json["features"])
    fire_gdf.label = selected_fire
    fire_gdf.id = selected_fire
    fire_gdf["_geojson"] = fire_gdf.geometry.apply(mapping)

    # compute JD range
    jds = fire_gdf["JD"].dropna()
    if len(jds) > 0:
        jd_min, jd_max = int(jds.min()), int(jds.max())
        jd_range = jd_max - jd_min
    else:
        jd_min, jd_max, jd_range = 0, 1, 1

    placeholder.markdown(f"""
        **Fire ID:** `{selected_fire}`  
        **JD Range:** {jd_min} → {jd_max} ({jd_range} days)  
        **Weather Code:** Checkout [WMO weather interpretation codes](https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM) for details
        """)

    # color scale granularity
    color_scale_type = "linear" if jd_range > 10 else "quantize"

    color_range = {
        "name": "Custom40_YlOrRd",
        "type": "sequential",
        "category": "Custom",
        "colors": [
            "#ffffcc","#fffac2","#fff5b8","#fff0ae","#ffeaa4",
            "#ffe59a","#ffe090","#ffdb86","#ffd67d","#ffd173",
            "#ffcc69","#fec75f","#fbc255","#f8bd4c","#f5b842",
            "#f2b339","#efae30","#eba926","#e8a41d","#e59f14",
            "#e29a0b","#df9500","#db8f00","#d88900","#d58300",
            "#d27d00","#cf7700","#cc7100","#c96b00","#c66500",
            "#c35f00","#c05900","#bd5300","#ba4d00","#b74700",
            "#b44100","#b13b00","#ae3500","#ab2f00","#a82900"
        ]
    }


    st.session_state.datasets = [fire_gdf]

    # make layer id unique to force full reload
    layer_id = f"wildfire-polygons-{selected_fire}"

    # --- Build config ---
    kepler_config = {
        "version": "v1",
        "config": {
            "visState": {
                "filters": [],
                "layers": [
                    {
                        "id": layer_id,
                        "type": "geojson",
                        "config": {
                            "dataId": selected_fire,
                            "label": selected_fire,
                            "color": [255, 153, 31],
                            "highlightColor": [252, 242, 26, 255],
                            "columns": {"geojson": "_geojson"},
                            "isVisible": True,
                            "hidden": False,
                            "visConfig": {
                                "opacity": 0.8,
                                "thickness": 0.5,
                                "strokeColor": [0, 0, 0],
                                "colorRange": color_range,
                                "filled": True,
                                "stroked": False,
                                "enable3d": False,
                                "wireframe": False,
                            },
                            "textLabel": [],
                        },
                        "visualChannels": {
                            "colorField": {"name": "JD", "type": "integer"},
                            "colorScale": color_scale_type,
                        },
                    }
                ],
                "interactionConfig": {
                    "tooltip": {
                        "fieldsToShow": {
                            selected_fire: [
                                {"name": "UID_Fire", "format": None},
                                {"name": "JD", "format": None},
                                {"name": "Map_Date", "format": None},
                            ]
                        },
                        "enabled": True,
                    },
                    "legend": {"enabled": True},
                },
                "layerBlending": "normal",
                "splitMaps": [],
            },
            "mapState": {
                "bearing": 0,
                "latitude": float(fire_gdf.geometry.centroid.y.mean()),
                "longitude": float(fire_gdf.geometry.centroid.x.mean()),
                "zoom": 6,
                "pitch": 0,
            },
            "mapStyle": {
                "styleType": "dark-matter",
                # "mapStyles": {
                #     "Satellite Streets": {
                #         "accessToken": MY_TOKEN,
                #         "url": "mapbox://styles/mapbox/satellite-streets-v12",
                #         "label": "Mapbox Satellite Streets",
                #     }
                # },
            },
        },
    }

    # --- Render map ---
    options = {"keepExistingConfig": False}

    # ✅ key fix: unique per selection, avoids caching but no flicker
    map_key = f"kepler_map_{selected_fire}"

    with col_map:
        st.subheader("🔥 Wildfire Progression Visualization")
        map_instance = keplergl(
            st.session_state.datasets,
            options=options,
            config=kepler_config,
            height=650,
            key=map_key,  # ensures fresh map per fire, no overlap
        )

    # ---------------------------------------------------------
    # BELOW: Terrain Visualization for the same fire
    # ---------------------------------------------------------
    with col_viz:
        st.subheader("⛰ Terrain & DEM-based Layers for this Fire")

        DEM_PATH = os.path.join(FIRE_DIR, f"{selected_fire}_dem.tif")
        HILL_PATH = os.path.join(FIRE_DIR, f"{selected_fire}_hillshade.npy")
        SLOPE_PATH = os.path.join(FIRE_DIR, f"{selected_fire}_slope_deg.npy")
        ASPECT_PATH = os.path.join(FIRE_DIR, f"{selected_fire}_aspect_deg.npy")

        # Tabs for separate interactive views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["🌍 DEM 3D Surface", "🌗 Hillshade (2D)", "📈 Slope (2D)", "🧭 Aspect (2D)", "⛰ DEM (2D)"]
        )

        # Small helpers
        def safe_load_dem(path):
            if not os.path.exists(path):
                return None, None
            try:
                with rasterio.open(path) as src:
                    arr = src.read(1).astype(float)
                    transform = src.transform
                return arr, transform
            except Exception as e:
                st.warning(f"Failed to load DEM: {e}")
                return None, None

        def safe_load_npy(path, label):
            if not os.path.exists(path):
                st.info(f"{label} file not found for this fire.")
                return None
            try:
                return np.load(path)[::-1, :]
            except Exception as e:
                st.warning(f"Failed to load {label}: {e}")
                return None

        # ---------------- DEM 3D ----------------
        with tab1:
            dem, transform = safe_load_dem(DEM_PATH)
            if dem is not None:
                fig = go.Figure(
                    data=[
                        go.Surface(
                            z=dem,
                            colorscale="Viridis",
                            colorbar=dict(title="Elevation"),
                        )
                    ]
                )
                fig.update_layout(
                    height=500,
                    scene=dict(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        zaxis=dict(title="Elevation"),
                    ),
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.warning("DEM not available for this fire.")

        # ---------------- Hillshade 2D ----------------
        with tab2:
            hill = safe_load_npy(HILL_PATH, "Hillshade")
            if hill is not None:
                fig = go.Figure(
                    data=[
                        go.Heatmap(
                            z=hill,
                            colorscale="gray",
                            showscale=False,
                        )
                    ]
                )
                fig.update_layout(
                    height=500,
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, width='stretch')

        # ---------------- Slope 2D ----------------
        with tab3:
            slope = safe_load_npy(SLOPE_PATH, "Slope (deg)")
            if slope is not None:
                fig = go.Figure(
                    data=[
                        go.Heatmap(
                            z=slope,
                            colorscale="Turbo",
                            colorbar=dict(title="Slope (°)"),
                        )
                    ]
                )
                fig.update_layout(
                    height=500,
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, width='stretch')

        # ---------------- Aspect 2D ----------------
        with tab4:
            aspect = safe_load_npy(ASPECT_PATH, "Aspect (deg)")
            if aspect is not None:
                fig = go.Figure(
                    data=[
                        go.Heatmap(
                            z=aspect,
                            colorscale="HSV",
                            colorbar=dict(title="Aspect (°)"),
                        )
                    ]
                )
                fig.update_layout(
                    height=500,
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, width='stretch')

        # ---------------- DEM 2D ----------------
        with tab5:
            dem2d, _ = safe_load_dem(DEM_PATH)
            if dem2d is not None:
                fig = go.Figure(
                    data=[
                        go.Heatmap(
                            z=dem2d,
                            colorscale="gray",
                            colorbar=dict(title="Elevation"),
                        )
                    ]
                )
                fig.update_layout(
                    height=500,
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("DEM not available for this fire.")