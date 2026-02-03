import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd

import streamlit as st
import pydeck as pdk

import folium
from streamlit_folium import st_folium

import osmnx as ox
import networkx as nx

from shapely.geometry import Point


# =========================================================
# 0) PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

GRID_SHP = os.path.join(DATA_DIR, "nlsp_021001001.shp")      # 전수 격자(여기서 남현동만 clip)
UNCOVERED_GPKG = os.path.join(DATA_DIR, "demo_uncovered.gpkg")  # 선택(없어도 됨)

# ✅ 행정구역 파일(남현동 포함된 행정동/법정동 경계)
# - 둘 중 하나만 있으면 됨
ADMIN_GPKG = os.path.join(DATA_DIR, "demo_admin.gpkg")

GRID_ID_COL = "gid"
GRID_POP_COL = "val"

TARGET_CRS = 5179
MAP_CRS = 4326

# =========================================================
# 1) 고정 파라미터(일반값 고정)
# =========================================================
KPI_RADIUS_M = 1250
WALK_SPEED_MPS = 1.4
CUTOFF_MIN = 5
CUTOFF_SEC = CUTOFF_MIN * 60

GRAPH_DIST_M = 3500
NEW_STATION_BUFFER_M = 1250

CARTO_POSITRON_GL = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

MAPBOX_TOKEN = os.getenv("MAPBOX_API_KEY") or os.getenv("MAPBOX_TOKEN")
if MAPBOX_TOKEN:
    pdk.settings.mapbox_api_key = MAPBOX_TOKEN


# =========================================================
# 2) Streamlit Page
# =========================================================
st.set_page_config(page_title="5강 | 남현동만", layout="wide")

st.title("🚲 5강 | 남현동만: 격자 선택 → KPI 즉석 계산 → 좌(Pydeck) / 우(남현동 경계 + 커버효과 + 5분 네트워크)")
st.caption(
    f"고정값: KPI반경={KPI_RADIUS_M}m | 보행속도={WALK_SPEED_MPS}m/s | 컷오프={CUTOFF_MIN}분 | 그래프반경={GRAPH_DIST_M}m | 신규 커버반경={NEW_STATION_BUFFER_M}m"
)


# =========================================================
# 3) Load (경로만 캐시)
# =========================================================
@st.cache_data(show_spinner=True)
def load_grid(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"GRID_SHP not found: {path}")

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError("GRID_SHP CRS is None. (.prj 확인)")
    gdf = gdf.to_crs(TARGET_CRS)

    if GRID_ID_COL not in gdf.columns:
        raise ValueError(f"GRID_ID_COL='{GRID_ID_COL}' not found in grid")

    gdf[GRID_ID_COL] = gdf[GRID_ID_COL].astype(str)

    if GRID_POP_COL in gdf.columns:
        gdf["pop"] = pd.to_numeric(gdf[GRID_POP_COL], errors="coerce").fillna(0).astype(float)
    elif "pop" in gdf.columns:
        gdf["pop"] = pd.to_numeric(gdf["pop"], errors="coerce").fillna(0).astype(float)
    else:
        gdf["pop"] = 0.0

    gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf[[GRID_ID_COL, "pop", "geometry"]].copy()


@st.cache_data(show_spinner=False)
def load_uncovered(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=TARGET_CRS)

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError("UNCOVERED_GPKG CRS is None.")
    gdf = gdf.to_crs(TARGET_CRS)
    gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf[["geometry"]].copy()


@st.cache_data(show_spinner=True)
def load_admin() -> gpd.GeoDataFrame:
    path = None
    if os.path.exists(ADMIN_GPKG):
        path = ADMIN_GPKG

    if path is None:
        raise FileNotFoundError("남현동 행정구역 파일이 필요합니다. data/admin_dong.gpkg 또는 data/admin_dong.shp를 넣어주세요.")

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError("ADMIN CRS is None.")
    gdf = gdf.to_crs(TARGET_CRS)
    gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf


@st.cache_data(show_spinner=True)
def pick_namhyeon_polygon(gdf_admin_5179: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # 남현동 찾을 때 사용할 후보 컬럼들(너 데이터에 맞춰 추가 가능)
    NAME_COL_CANDIDATES = [
        "ADM_NM", "adm_nm", "ADMNM",
        "region_nm", "REGION_NM",
        "emd_nm", "EMD_NM",
        "dong_nm", "DONG_NM",
        "법정동명", "행정동명"
    ]

    name_col = None
    for c in NAME_COL_CANDIDATES:
        if c in gdf_admin_5179.columns:
            name_col = c
            break

    if name_col is None:
        # 컬럼을 못 찾으면 그냥 첫 폴리곤을 남현동으로 간주(최후 fallback)
        return gdf_admin_5179.iloc[[0]].copy()

    mask = gdf_admin_5179[name_col].astype(str).str.contains("남현", na=False)
    if mask.sum() == 0:
        # "남현동"으로도 한 번 더 시도
        mask = gdf_admin_5179[name_col].astype(str).str.contains("남현동", na=False)

    if mask.sum() == 0:
        # 그래도 없으면 첫 폴리곤 fallback
        return gdf_admin_5179.iloc[[0]].copy()

    return gdf_admin_5179.loc[mask].copy()


@st.cache_data(show_spinner=True)
def clip_grid_to_polygon(grid_path: str, admin_path_marker: str = "admin"):
    # admin_path_marker는 캐시 키 안정화용 더미(실제론 load_admin 내부에서 파일을 읽음)
    gdf_grid = load_grid(grid_path)
    gdf_admin = load_admin()
    gdf_nam = pick_namhyeon_polygon(gdf_admin)

    nam_union = gdf_nam.geometry.union_all()
    # intersects로 먼저 줄이고 clip
    gdf_sub = gdf_grid[gdf_grid.geometry.intersects(nam_union)].copy()
    gdf_clip = gpd.clip(gdf_sub, gdf_nam)
    gdf_clip["geometry"] = gdf_clip.geometry.buffer(0)
    return gdf_clip, gdf_nam


@st.cache_data(show_spinner=False)
def attach_is_uncovered(grid_gdf_5179: gpd.GeoDataFrame, unc_path: str) -> gpd.GeoDataFrame:
    gdf_unc = load_uncovered(unc_path)
    g = grid_gdf_5179.copy()
    if len(gdf_unc) == 0:
        g["is_uncovered"] = False
        return g
    unc_union = gdf_unc.geometry.union_all()
    g["is_uncovered"] = g.geometry.intersects(unc_union)
    return g


@st.cache_resource(show_spinner=True)
def build_osm_graph_from_point(lat: float, lon: float, dist_m: int, network_type: str = "walk"):
    ox.settings.log_console = False
    G = ox.graph_from_point((lat, lon), dist=int(dist_m), network_type=network_type, simplify=True)
    try:
        G = ox.distance.add_edge_lengths(G)
    except Exception:
        try:
            G = ox.add_edge_lengths(G)
        except Exception:
            pass
    return G


# =========================================================
# 4) Data Load: 남현동 격자만
# =========================================================
with st.spinner("남현동 격자만 로딩/클립 중..."):
    gdf_grid_nam, gdf_namhyeon = clip_grid_to_polygon(GRID_SHP, admin_path_marker="admin")
    gdf_grid_nam = attach_is_uncovered(gdf_grid_nam, UNCOVERED_GPKG)
    gdf_unc = load_uncovered(UNCOVERED_GPKG)

if len(gdf_grid_nam) == 0:
    st.error("남현동으로 clip된 격자가 0개입니다. 행정구역 파일/CRS/남현동 명칭 컬럼을 확인하세요.")
    st.stop()

all_gids = gdf_grid_nam[GRID_ID_COL].astype(str).tolist()


# =========================================================
# 5) KPI 아래 gid 선택(남현동 gid만)
# =========================================================
st.subheader("KPI")
sel_gid = st.selectbox("남현동 격자 gid 선택", options=all_gids, index=0, key="gid_select")


# =========================================================
# 6) KPI + 신규 커버 효과(남현동 내부만 대상으로)
# =========================================================
row = gdf_grid_nam.loc[gdf_grid_nam[GRID_ID_COL] == str(sel_gid)]
sel_poly = row.geometry.iloc[0]
sel_center_5179 = sel_poly.centroid

kpi_circle_5179 = sel_center_5179.buffer(float(KPI_RADIUS_M))
station_buffer_5179 = sel_center_5179.buffer(float(NEW_STATION_BUFFER_M))

in_circle = gdf_grid_nam.geometry.intersects(kpi_circle_5179)
gdf_in = gdf_grid_nam.loc[in_circle, [GRID_ID_COL, "pop", "is_uncovered", "geometry"]].copy()

total_pop = float(gdf_in["pop"].sum())
unc_pop = float(gdf_in.loc[gdf_in["is_uncovered"] == True, "pop"].sum())
cov_pop = total_pop - unc_pop
unc_rate = (unc_pop / total_pop) if total_pop > 0 else 0.0

newly_covered_geom = None
if len(gdf_unc) > 0:
    # 남현동 영역 내부 비커버만 대상으로 보고 싶으면: unc ∩ namhyeon 먼저
    nam_union = gdf_namhyeon.geometry.union_all()
    unc_union = gdf_unc.geometry.union_all().intersection(nam_union)
    newly_covered_geom = unc_union.intersection(station_buffer_5179)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("선택 gid", str(sel_gid))
c2.metric("KPI 반경 내 격자 수", f"{len(gdf_in):,}")
c3.metric("총 인구", f"{total_pop:,.0f}")
c4.metric("비커버 인구", f"{unc_pop:,.0f}")
c5.metric("비커버 비율", f"{unc_rate*100:.2f}%")


# =========================================================
# 7) 좌(Pydeck) / 우(Folium)
# =========================================================
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("좌측: Pydeck | 남현동 격자 + 신규 정류장 + 커버 버퍼")

    gdf_ll = gdf_in.to_crs(MAP_CRS).copy()

    pop = gdf_ll["pop"].clip(lower=0).astype(float)
    cap_val = float(pop.quantile(0.995)) if len(pop) > 0 else 0.0
    pop_capped = np.minimum(pop, cap_val) if cap_val > 0 else pop
    gdf_ll["elev"] = (np.power(pop_capped, 1.80) * 0.02).astype(float)

    records = []
    for gid, popv, is_unc, elev, geom in zip(
        gdf_ll[GRID_ID_COL].astype(str).tolist(),
        gdf_ll["pop"].tolist(),
        gdf_ll["is_uncovered"].tolist(),
        gdf_ll["elev"].tolist(),
        gdf_ll.geometry.tolist(),
    ):
        if geom is None or geom.is_empty:
            continue
        polys = [geom] if geom.geom_type == "Polygon" else (list(geom.geoms) if geom.geom_type == "MultiPolygon" else [])
        for poly in polys:
            records.append(
                {"gid": gid, "pop": float(popv), "is_uncovered": bool(is_unc), "elev": float(elev), "polygon": list(poly.exterior.coords)}
            )

    kpi_circle_ll = gpd.GeoSeries([kpi_circle_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
    station_buf_ll = gpd.GeoSeries([station_buffer_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
    center_ll = gpd.GeoSeries([sel_center_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]

    layer_grid = pdk.Layer(
        "PolygonLayer",
        data=records,
        get_polygon="polygon",
        extruded=True,
        filled=True,
        stroked=False,
        get_elevation="elev",
        elevation_scale=1,
        get_fill_color="[240, 240, 240, 160]",
        pickable=True,
    )

    layer_kpi = pdk.Layer(
        "PolygonLayer",
        data=[{"polygon": list(kpi_circle_ll.exterior.coords)}],
        get_polygon="polygon",
        filled=False,
        stroked=True,
        get_line_color=[30, 30, 30, 220],
        get_line_width=140,
    )

    layer_station_buf = pdk.Layer(
        "PolygonLayer",
        data=[{"polygon": list(station_buf_ll.exterior.coords)}],
        get_polygon="polygon",
        filled=False,
        stroked=True,
        get_line_color=[0, 120, 0, 220],
        get_line_width=140,
    )

    layer_station = pdk.Layer(
        "ScatterplotLayer",
        data=[{"lon": float(center_ll.x), "lat": float(center_ll.y)}],
        get_position="[lon, lat]",
        get_radius=50,
        pickable=True,
    )

    view = pdk.ViewState(
        latitude=float(center_ll.y),
        longitude=float(center_ll.x),
        zoom=14,
        pitch=55,
        bearing=20,
    )

    deck = pdk.Deck(
        layers=[layer_grid, layer_kpi, layer_station_buf, layer_station],
        initial_view_state=view,
        map_style=CARTO_POSITRON_GL if not MAPBOX_TOKEN else "mapbox://styles/mapbox/light-v11",
        tooltip={"text": "gid: {gid}\npop: {pop}\nuncovered: {is_uncovered}"},
    )

    st.pydeck_chart(deck, width="stretch")


with right:
    st.subheader("우측: Folium | 남현동 경계 + 비커버 + 신규 커버 + 5분 네트워크")

    center_ll = gpd.GeoSeries([sel_center_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
    lon, lat = float(center_ll.x), float(center_ll.y)

    with st.spinner(f"OSM 그래프 다운로드/캐시 확인... (dist={GRAPH_DIST_M}m)"):
        G = build_osm_graph_from_point(lat=lat, lon=lon, dist_m=int(GRAPH_DIST_M), network_type="walk")

    with st.spinner("그래프 투영(project) + travel_time 세팅..."):
        Gp = ox.project_graph(G)

        gdf_center_proj = gpd.GeoSeries([Point(lon, lat)], crs=MAP_CRS).to_crs(Gp.graph["crs"])
        px, py = float(gdf_center_proj.iloc[0].x), float(gdf_center_proj.iloc[0].y)

        for u, v, k, data in Gp.edges(keys=True, data=True):
            length_m = float(data.get("length", 0.0))
            data["travel_time"] = (length_m / float(WALK_SPEED_MPS)) if WALK_SPEED_MPS > 0 else np.inf

        source_node = ox.distance.nearest_nodes(Gp, X=px, Y=py)

    with st.spinner(f"{CUTOFF_MIN}분 네트워크 계산 중..."):
        lengths = nx.single_source_dijkstra_path_length(Gp, int(source_node), cutoff=float(CUTOFF_SEC), weight="travel_time")
        reachable_nodes = set(lengths.keys())
        SG = Gp.subgraph(reachable_nodes).copy()

        gdf_edges = ox.graph_to_gdfs(SG, nodes=False, edges=True, fill_edge_geometry=True)
        if gdf_edges.crs is None:
            gdf_edges = gdf_edges.set_crs(Gp.graph["crs"])
        gdf_edges_ll = gdf_edges.to_crs(MAP_CRS).reset_index(drop=True)

    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="cartodbpositron")

    # ✅ 남현동 경계만 표시
    gdf_nam_ll = gdf_namhyeon.to_crs(MAP_CRS)
    folium.GeoJson(
        gdf_nam_ll,
        name="남현동 경계",
        style_function=lambda x: {"color": "#000000", "weight": 3, "fillOpacity": 0.02},
    ).add_to(m)

    # 비커버(남현동 내부만 보고 싶으면 gdf_unc ∩ namhyeon으로 먼저 잘라서 표시)
    if len(gdf_unc) > 0:
        nam_union = gdf_namhyeon.geometry.union_all()
        unc_union = gdf_unc.geometry.union_all().intersection(nam_union)
        unc_ll = gpd.GeoSeries([unc_union], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
        if not unc_ll.is_empty:
            folium.GeoJson(
                {"type": "Feature", "properties": {}, "geometry": unc_ll.__geo_interface__},
                name="비커버(남현동)",
                style_function=lambda x: {"color": "#ff0000", "weight": 2, "fillOpacity": 0.10},
            ).add_to(m)

    # 신규 정류장(격자 중심점)
    folium.Marker(
        location=[lat, lon],
        tooltip=f"신규 따릉이 정류장(가정): gid={sel_gid}",
        icon=folium.Icon(color="green", icon="bicycle", prefix="fa"),
    ).add_to(m)

    # 신규 커버 버퍼
    station_buf_ll = gpd.GeoSeries([station_buffer_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
    folium.GeoJson(
        {"type": "Feature", "properties": {}, "geometry": station_buf_ll.__geo_interface__},
        name="신규 커버 반경",
        style_function=lambda x: {"color": "#00aa00", "weight": 2, "fillOpacity": 0.03},
    ).add_to(m)

    # 새로 커버된 비커버(교집합)
    if newly_covered_geom is not None and (not newly_covered_geom.is_empty):
        newly_ll = gpd.GeoSeries([newly_covered_geom], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
        folium.GeoJson(
            {"type": "Feature", "properties": {}, "geometry": newly_ll.__geo_interface__},
            name="새로 커버된 비커버",
            style_function=lambda x: {"color": "#008800", "weight": 2, "fillOpacity": 0.25},
        ).add_to(m)

    # 5분 네트워크
    if len(gdf_edges_ll) > 0:
        folium.GeoJson(
            gdf_edges_ll,
            name=f"5분 네트워크({CUTOFF_MIN}min)",
            style_function=lambda x: {"color": "#0055ff", "weight": 3, "opacity": 0.85},
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=680)


with st.expander("진단"):
    st.write("GRID_SHP:", GRID_SHP, "(exists:", os.path.exists(GRID_SHP), ")")
    st.write("ADMIN_GPKG:", ADMIN_GPKG, "(exists:", os.path.exists(ADMIN_GPKG), ")")
    st.write("ADMIN_SHP :", ADMIN_SHP,  "(exists:", os.path.exists(ADMIN_SHP), ")")
    st.write("UNCOVERED_GPKG:", UNCOVERED_GPKG, "(exists:", os.path.exists(UNCOVERED_GPKG), ")")
    st.write("남현동 격자 수:", len(gdf_grid_nam))
    st.write("남현동 admin rows:", len(gdf_namhyeon))
    st.write("admin columns:", list(load_admin().columns))

