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
import joblib
from graphviz import Digraph
# Mapbox
MY_TOKEN = "pk.eyJ1IjoibWF4ZG9taW5pYyIsImEiOiJjbWhyejVvY2owMmNsMmtwdmEwNHd3YjRmIn0.4mJfybYpE2oWZc7iy1hiHA"

st.set_page_config(page_title="Wildfire Progression Viewer", layout="wide")
MODEL_PATH = "model/gaussian_hmm_model.pkl"
SCALER_PATH = "model/scaler.pkl"

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_hmm_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)

model = load_hmm_model()
scaler = load_scaler()

# Extract HMM info
A = model.transmat_
pi = model.startprob_
means = model.means_
n_states = model.n_components

# OPTIONAL: Give readable state names
state_names = [
    "Mildly Suppressed / Stable",
    "Rain-Driven Collapse",
    "Heat-Driven Extreme Growth",
    "Flat-Terrain Moderate Growth",
    "Chaotic Breakdown (Heavy Rain + High Wind)"

][:n_states]

feature_names = ['slope_mean', 'slope_median', 'slope_circular', 'aspect_mean',
       'aspect_median', 'aspect_circular', 'hillshade_mean',
       'hillshade_median', 'delta_area', 'delta_perimeter', 'delta_cx',
       'delta_cy', 'centroid_shift_m', 'temperature_2m_max',
       'temperature_2m_min', 'temperature_2m_mean', 'precipitation_sum',
       'rain_sum', 'snowfall_sum', 'wind_speed_10m_max', 'wind_gusts_10m_max',
       'wind_direction_10m_dominant', 'shortwave_radiation_sum',
       'et0_fao_evapotranspiration']

# -------------------------
# Build GraphViz HMM Diagram
# -------------------------
def build_hmm_graph(model, state_names=None, feature_names=None, max_features=3):

    g = Digraph("Wildfire_HMM", format="png")
    g.attr(rankdir="LR", size="12,6")

    # Start Node
    g.node("Start", shape="doublecircle", fontsize="14", style="bold")

    A = model.transmat_
    pi = model.startprob_
    means = model.means_
    n_states = model.n_components

    # -------------------------
    # Hidden State Nodes
    # -------------------------
    for i in range(n_states):
        label = state_names[i] if state_names else f"State {i}"

        # show top absolute-mean features
        if feature_names is not None:
            idx = np.argsort(np.abs(means[i]))[::-1][:max_features]
            emissions_text = "\\n".join(
                f"{feature_names[j]}: {means[i][j]:.1f}" for j in idx
            )
        else:
            # default: show 3 highest mean values
            idx = np.argsort(np.abs(means[i]))[::-1][:max_features]
            emissions_text = "\\n".join(
                f"μ[{j}] = {means[i][j]:.1f}" for j in idx
            )

        g.node(
            f"S{i}",
            label=f"{label}\n---\n{emissions_text}",
            shape="circle",
            fontsize="12"
        )

        # Start transition
        g.edge("Start", f"S{i}", label=f"{pi[i]:.2f}", fontsize="10")

    # -------------------------
    # Transitions
    # -------------------------
    for i in range(n_states):
        for j in range(n_states):
            if A[i, j] > 0.05:  # show only meaningful transitions
                thickness = str(1 + A[i, j] * 5)
                g.edge(
                    f"S{i}", f"S{j}",
                    label=f"{A[i, j]:.2f}",
                    penwidth=thickness,
                    fontsize="10"
                )

    return g

if "datasets" not in st.session_state:
    st.session_state.datasets = []

# Sidebar
st.sidebar.title("Wildfire Selection")

yyyy = [2018, 2019]

select_yyyy = st.sidebar.selectbox("Select the Year", yyyy)

weather_dfs = {}
for yy in yyyy:
    fd = f"data/{yy}"
    weather_dfs[yy] = pd.read_csv(os.path.join(fd, f"fire_hmm_ready_{yy}.csv"))

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
data_df = weather_dfs[select_yyyy][weather_dfs[select_yyyy]['fire_id']==int(selected_fire.split('_')[1])].drop(['fire_id', 'year', 'jd'],axis=1)
st.dataframe(data_df[feature_names])

data_df_scl = scaler.transform(data_df[feature_names].values)
logprob, decoded_states = model.decode(data_df_scl)
print(logprob)
print(decoded_states)
decoded_state_names = [state_names[s] for s in decoded_states]

tab_1, tab_2, tab_3 = st.tabs(['🔎Wildfire Data Exploration', '📍Hidden Markov Model', '📖Fire Interpretation'])
with tab_1: 
    col_map, col_viz = st.columns([1.5, 1]) 
with tab_2:
    st.markdown('''
                | **State**                                                | **Spread Intensity (ΔArea/ΔPerimeter)** | **Wind Influence** | **Temperature**  | **Moisture / Rain** | **Terrain Influence (Slope/Aspect)** | **Interpretation (Corrected)**                                                         |
| -------------------------------------------------------- | --------------------------------------- | ------------------ | ---------------- | ------------------- | ------------------------------------ | -------------------------------------------------------------------------------------- |
| **State 1 — Mildly Suppressed / Stable**                 | 🔵 *Small negative spread*              | Low–Moderate       | Moderate         | Low–Moderate        | **Moderate slope but stable**        | Fire is slowing but not collapsing; light suppression, stable boundary.                |
| **State 2 — Rain-Driven Collapse**                       | 🔵 **Large negative spread**            | Low–Moderate       | Moderate         | **Moderate rain**   | Moderate slope                       | Fire boundary collapses primarily due to moisture; moderate centroid drift.            |
| **State 3 — Chaotic Breakdown (Heavy Rain + High Wind)** | 🔴 **Extremely large negative spread**  | **Highest winds**  | Moderate         | **Highest rain**    | **Flattest terrain**                 | Violent fire collapse; very large centroid movement; influenced by wind + rainfall.    |
| **State 4 — Flat-Terrain Moderate Growth**               | 🟠 *Moderate positive spread*           | Moderate           | Moderate         | Low                 | **Very low slope**                   | Fire spreads steadily on flat terrain; not wind-dominated, not heat-dominated.         |
| **State 5 — Heat-Driven Extreme Growth**                 | 🟢 **Very large positive spread**       | Low–Moderate       | **Highest temp** | **No rain**         | Low slope                            | Rapid expansion due to hot, dry conditions; energy-driven, not terrain or wind-driven. |
                ''')
    max_features = st.slider("Number of Emission Features", min_value=3, max_value=len(feature_names), value=5)
    graph = build_hmm_graph(model, state_names=state_names, feature_names=feature_names, max_features=max_features)
    st.graphviz_chart(graph)
with tab_3:
    st.subheader("🔥 State Timeline Across Days")

    jd_values = weather_dfs[select_yyyy][weather_dfs[select_yyyy]['fire_id']==int(selected_fire.split('_')[1])]['jd'].values

    fig_timeline = go.Figure()

    fig_timeline.add_trace(
        go.Scatter(
            x=jd_values,
            y=decoded_states,
            mode='lines+markers',
            line=dict(width=3),
            marker=dict(size=8),
            name="State"
        )
    )

    fig_timeline.update_layout(
        height=300,
        xaxis_title="Julian Day (JD)",
        yaxis_title="State",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_states)),
            ticktext=state_names
        ),
        margin=dict(l=40, r=20, t=40, b=40)
    )

    st.plotly_chart(fig_timeline, use_container_width=True)

    st.subheader("🔁 State Transition Graph for This Fire")

    # compute transitions
    T = np.zeros((n_states, n_states))
    for i in range(len(decoded_states) - 1):
        a = decoded_states[i]
        b = decoded_states[i+1]
        T[a, b] += 1

    # build graphviz circular layout
    g2 = Digraph("State_Transitions", format="png")
    g2.attr(rankdir="LR", size="10,5")

    for i in range(n_states):
        g2.node(f"S{i}", label=state_names[i], shape="circle")

    for i in range(n_states):
        for j in range(n_states):
            if T[i,j] > 0:
                g2.edge(f"S{i}", f"S{j}", label=str(int(T[i,j])), penwidth=str(1+T[i,j]/2))

    st.graphviz_chart(g2)


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