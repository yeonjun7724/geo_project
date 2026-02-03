# =========================================================
# 5강 완성형 app.py
# - gid 선택: KPI 아래(본문)로 이동
# - 파라미터(반경/속도/컷오프/그래프반경): 일반적인 값으로 고정
# - 좌: Pydeck (격자+비커버/커버+신규정류장+커버버퍼)
# - 우: Folium (행정구역+비커버+신규정류장+새로 커버된 영역+5분 네트워크)
# - 라우팅: OSMnx+NetworkX 즉석 계산 (project graph로 nearest_nodes 안정화)
# =========================================================

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
from shapely.ops import unary_union


# =========================================================
# 0) PATHS (GitHub 기준)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 필수: 전수 격자(SHP 세트: .shp/.shx/.dbf/.prj 모두)
GRID_SHP = os.path.join(DATA_DIR, "nlsp_021001001.shp")

# 선택: 비커버 폴리곤(없으면 전체 uncovered=False 처리)
UNCOVERED_GPKG = os.path.join(DATA_DIR, "demo_uncovered.gpkg")

# 선택: 행정구역(없으면 스킵) - 파일명만 맞추면 자동으로 그림
#  - 예: data/admin_dong.gpkg 또는 data/admin_dong.shp
ADMIN_GPKG = os.path.join(DATA_DIR, "admin_dong.gpkg")
ADMIN_SHP  = os.path.join(DATA_DIR, "admin_dong.shp")

GRID_ID_COL = "gid"
GRID_POP_COL = "val"   # 없으면 pop=0

TARGET_CRS = 5179      # 분석용(미터)
MAP_CRS = 4326         # 지도용(WGS84)

# =========================================================
# 1) 고정 파라미터 (일반적인 수준)
# =========================================================
KPI_RADIUS_M = 1250      # KPI(반경 내 인구/비커버) 계산용 반경
WALK_SPEED_MPS = 1.4     # 보행 속도 (약 5km/h)
CUTOFF_MIN = 5           # 네트워크 컷오프(분)
CUTOFF_SEC = CUTOFF_MIN * 60

GRAPH_DIST_M = 3500      # OSM 그래프 다운로드 반경(중심점 기준)
NEW_STATION_BUFFER_M = 1250  # "신규 따릉이가 커버"한다고 가정하는 커버 반경(강의 컨셉 맞춤)

# Pydeck basemap (토큰 없어도 뜨게: Carto GL Style)
CARTO_POSITRON_GL = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"


# =========================================================
# 2) Streamlit Page
# =========================================================
st.set_page_config(page_title="5강 | Streamlit + Pydeck + OSMnx", layout="wide")

st.title("🚲 5강 | 대시보드: 격자 선택 → KPI 즉석 계산 → 좌(Pydeck) / 우(커버효과 + 5분 네트워크)")
st.caption(
    f"고정값: KPI반경={KPI_RADIUS_M}m | 보행속도={WALK_SPEED_MPS}m/s | 컷오프={CUTOFF_MIN}분 | 그래프반경={GRAPH_DIST_M}m | 신규 커버반경={NEW_STATION_BUFFER_M}m"
)

# Mapbox 토큰이 있으면 자동 적용(없어도 Carto GL로 뜨게 설계)
MAPBOX_TOKEN = os.getenv("MAPBOX_API_KEY") or os.getenv("MAPBOX_TOKEN")
if MAPBOX_TOKEN:
    pdk.settings.mapbox_api_key = MAPBOX_TOKEN


# =========================================================
# 3) Load (캐시) - 입력은 "path(str)"만 받기
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


@st.cache_data(show_spinner=False)
def load_admin() -> gpd.GeoDataFrame:
    # admin은 없을 수 있으니 optional
    path = None
    if os.path.exists(ADMIN_GPKG):
        path = ADMIN_GPKG
    elif os.path.exists(ADMIN_SHP):
        path = ADMIN_SHP

    if path is None:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=TARGET_CRS)

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError("ADMIN CRS is None.")
    gdf = gdf.to_crs(TARGET_CRS)
    gdf["geometry"] = gdf.geometry.buffer(0)
    # 컬럼은 그냥 다 두되 geometry만 보장
    return gdf


@st.cache_data(show_spinner=False)
def attach_is_uncovered(grid_path: str, unc_path: str) -> gpd.GeoDataFrame:
    gdf_grid = load_grid(grid_path)
    gdf_unc = load_uncovered(unc_path)

    g = gdf_grid.copy()
    if len(gdf_unc) == 0:
        g["is_uncovered"] = False
        return g

    unc_union = gdf_unc.geometry.union_all()
    g["is_uncovered"] = g.geometry.intersects(unc_union)
    return g


# =========================================================
# 4) OSMnx graph (캐시) - point 기반
# =========================================================
@st.cache_resource(show_spinner=True)
def build_osm_graph_from_point(lat: float, lon: float, dist_m: int, network_type: str = "walk"):
    ox.settings.log_console = False
    G = ox.graph_from_point((lat, lon), dist=int(dist_m), network_type=network_type, simplify=True)

    # edge length 호환
    try:
        G = ox.distance.add_edge_lengths(G)  # osmnx 2.x
    except Exception:
        try:
            G = ox.add_edge_lengths(G)       # osmnx 1.x
        except Exception:
            pass

    return G


# =========================================================
# 5) 데이터 로드
# =========================================================
with st.spinner("데이터 로딩 중..."):
    gdf_grid = attach_is_uncovered(GRID_SHP, UNCOVERED_GPKG)
    gdf_unc = load_uncovered(UNCOVERED_GPKG)
    gdf_admin = load_admin()

all_gids = gdf_grid[GRID_ID_COL].astype(str).tolist()
if len(all_gids) == 0:
    st.error("전수 격자를 불러오지 못했습니다. data 폴더 및 SHP 세트(.shp/.shx/.dbf/.prj)를 확인하세요.")
    st.stop()


# =========================================================
# 6) KPI 아래에 gid 선택 UI 배치 (사이드바 X)
# =========================================================
kpi_box = st.container()
with kpi_box:
    st.subheader("KPI")
    sel_gid = st.selectbox("전수 격자 gid 선택", options=all_gids, index=0, key="gid_select")


# =========================================================
# 7) KPI 즉석 계산 + 신규 커버 효과 계산
#    - KPI: 선택 gid 중심점 기준 KPI_RADIUS_M 내 격자(pop, uncovered)
#    - 신규 커버: 신규 정류장(중심점) 버퍼가 비커버 폴리곤을 얼마나 깎는지(교집합 면적)
# =========================================================
row = gdf_grid.loc[gdf_grid[GRID_ID_COL] == str(sel_gid)]
if len(row) == 0:
    st.error("선택 gid를 찾지 못했습니다.")
    st.stop()

sel_poly = row.geometry.iloc[0]
sel_center_5179 = sel_poly.centroid

kpi_circle_5179 = sel_center_5179.buffer(float(KPI_RADIUS_M))
station_buffer_5179 = sel_center_5179.buffer(float(NEW_STATION_BUFFER_M))

in_circle = gdf_grid.geometry.intersects(kpi_circle_5179)
gdf_in = gdf_grid.loc[in_circle, [GRID_ID_COL, "pop", "is_uncovered", "geometry"]].copy()

total_pop = float(gdf_in["pop"].sum())
unc_pop = float(gdf_in.loc[gdf_in["is_uncovered"] == True, "pop"].sum())
cov_pop = total_pop - unc_pop
unc_rate = (unc_pop / total_pop) if total_pop > 0 else 0.0

# 비커버 폴리곤이 있을 때 "새로 커버되는 비커버 영역"
newly_covered_geom = None
remaining_unc_geom = None

if len(gdf_unc) > 0:
    unc_union = gdf_unc.geometry.union_all()
    newly_covered_geom = unc_union.intersection(station_buffer_5179)
    remaining_unc_geom = unc_union.difference(station_buffer_5179)

# KPI 카드
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("선택 gid", str(sel_gid))
c2.metric("KPI 반경 내 격자 수", f"{len(gdf_in):,}")
c3.metric("총 인구", f"{total_pop:,.0f}")
c4.metric("비커버 인구", f"{unc_pop:,.0f}")
c5.metric("비커버 비율", f"{unc_rate*100:.2f}%")


# =========================================================
# 8) 좌(Pydeck) / 우(Folium) 레이아웃
# =========================================================
left, right = st.columns([1, 1], gap="large")

# ---------------------------------------------------------
# LEFT: Pydeck
#   - KPI 반경 내 격자 3D
#   - 신규 정류장(중심점) + 신규 커버 버퍼
# ---------------------------------------------------------
with left:
    st.subheader("좌측: Pydeck | KPI 반경 내 격자 + 신규 정류장 + 커버 버퍼")

    gdf_ll = gdf_in.to_crs(MAP_CRS).copy()

    # 높이: pop 기반 (클리핑)
    pop = gdf_ll["pop"].clip(lower=0).astype(float)
    cap_val = float(pop.quantile(0.995)) if len(pop) > 0 else 0.0
    pop_capped = np.minimum(pop, cap_val) if cap_val > 0 else pop
    gdf_ll["elev"] = (np.power(pop_capped, 1.80) * 0.02).astype(float)

    # PolygonLayer 입력 레코드
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

    # KPI 원 + 신규 커버 버퍼 + 중심점
    kpi_circle_ll = gpd.GeoSeries([kpi_circle_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
    station_buf_ll = gpd.GeoSeries([station_buffer_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
    center_ll = gpd.GeoSeries([sel_center_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]

    kpi_circle_coords = list(kpi_circle_ll.exterior.coords)
    station_buf_coords = list(station_buf_ll.exterior.coords)

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

    layer_kpi_circle = pdk.Layer(
        "PolygonLayer",
        data=[{"polygon": kpi_circle_coords}],
        get_polygon="polygon",
        filled=False,
        stroked=True,
        get_line_color=[30, 30, 30, 220],
        get_line_width=140,
    )

    layer_station_buffer = pdk.Layer(
        "PolygonLayer",
        data=[{"polygon": station_buf_coords}],
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
        get_radius=40,
        pickable=True,
    )

    view = pdk.ViewState(
        latitude=float(center_ll.y),
        longitude=float(center_ll.x),
        zoom=14,
        pitch=55,
        bearing=20,
    )

    # map_style: 토큰이 없어도 뜨게 Carto GL 먼저 사용
    deck = pdk.Deck(
        layers=[layer_grid, layer_kpi_circle, layer_station_buffer, layer_station],
        initial_view_state=view,
        map_style=CARTO_POSITRON_GL if not MAPBOX_TOKEN else "mapbox://styles/mapbox/light-v11",
        tooltip={"text": "gid: {gid}\npop: {pop}\nuncovered: {is_uncovered}"},
    )

    st.pydeck_chart(deck, width="stretch")


# ---------------------------------------------------------
# RIGHT: Folium
#   - 행정구역(있으면)
#   - 비커버 폴리곤(있으면)
#   - 신규 정류장(중심점)
#   - 신규 커버 버퍼
#   - 새로 커버된 비커버 영역(교집합)
#   - 5분 네트워크 edge(즉석 계산)
# ---------------------------------------------------------
with right:
    st.subheader("우측: Folium | 커버 효과(행정/비커버/신규) + 5분 네트워크")

    # 중심점(4326)
    center_ll = gpd.GeoSeries([sel_center_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
    lon, lat = float(center_ll.x), float(center_ll.y)

    # OSM 그래프 로딩 (중심점 기반)
    with st.spinner(f"OSM 그래프 다운로드/캐시 확인... (dist={GRAPH_DIST_M}m)"):
        G = build_osm_graph_from_point(lat=lat, lon=lon, dist_m=int(GRAPH_DIST_M), network_type="walk")

    # graph project(미터 좌표계) → sklearn 없이 nearest_nodes 안정화
    with st.spinner("그래프 투영(project) + travel_time 세팅..."):
        Gp = ox.project_graph(G)
        # project CRS로 중심점 변환
        gdf_center_proj = gpd.GeoSeries([Point(lon, lat)], crs=MAP_CRS).to_crs(Gp.graph["crs"])
        px, py = float(gdf_center_proj.iloc[0].x), float(gdf_center_proj.iloc[0].y)

        # travel_time(초)
        sp = float(WALK_SPEED_MPS)
        for u, v, k, data in Gp.edges(keys=True, data=True):
            length_m = float(data.get("length", 0.0))
            data["travel_time"] = (length_m / sp) if sp > 0 else np.inf

        try:
            source_node = ox.distance.nearest_nodes(Gp, X=px, Y=py)
        except Exception as e:
            st.error(f"nearest_nodes 실패: {e}")
            st.stop()

    # reachable subgraph → edges gdf
    with st.spinner(f"{CUTOFF_MIN}분 네트워크 계산 중..."):
        lengths = nx.single_source_dijkstra_path_length(Gp, int(source_node), cutoff=float(CUTOFF_SEC), weight="travel_time")
        reachable_nodes = set(lengths.keys())
        SG = Gp.subgraph(reachable_nodes).copy()

        gdf_edges = ox.graph_to_gdfs(SG, nodes=False, edges=True, fill_edge_geometry=True)
        if gdf_edges.crs is None:
            gdf_edges = gdf_edges.set_crs(Gp.graph["crs"])

        # 표시용 4326으로
        gdf_edges_ll = gdf_edges.to_crs(MAP_CRS).reset_index(drop=True)
        if "length" in gdf_edges_ll.columns:
            gdf_edges_ll["length_m"] = gdf_edges_ll["length"].astype(float)

    # 네트워크 KPI
    n_edges = int(len(gdf_edges_ll))
    total_len_km = float(gdf_edges_ll["length_m"].sum() / 1000.0) if "length_m" in gdf_edges_ll.columns else np.nan
    k6, k7 = st.columns(2)
    k6.metric("네트워크 edge 수", f"{n_edges:,}")
    k7.metric("네트워크 총 길이(km)", f"{total_len_km:,.2f}" if not np.isnan(total_len_km) else "-")

    # Folium 지도
    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="cartodbpositron")

    # (A) 행정구역(있으면)
    if len(gdf_admin) > 0:
        gdf_admin_ll = gdf_admin.to_crs(MAP_CRS)
        folium.GeoJson(
            gdf_admin_ll,
            name="행정구역",
            style_function=lambda x: {"color": "#777777", "weight": 2, "fillOpacity": 0.02},
        ).add_to(m)

    # (B) 비커버 폴리곤(있으면)
    if len(gdf_unc) > 0:
        gdf_unc_ll = gdf_unc.to_crs(MAP_CRS)
        folium.GeoJson(
            gdf_unc_ll,
            name="비커버(기존)",
            style_function=lambda x: {"color": "#ff0000", "weight": 2, "fillOpacity": 0.10},
        ).add_to(m)

    # (C) 신규 정류장(중심점) 마커
    folium.Marker(
        location=[lat, lon],
        tooltip=f"신규 따릉이 정류장(가정): gid={sel_gid}",
        icon=folium.Icon(color="green", icon="bicycle", prefix="fa"),
    ).add_to(m)

    # (D) 신규 커버 버퍼(원)
    station_buf_ll = gpd.GeoSeries([station_buffer_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
    folium.GeoJson(
        {"type": "Feature", "properties": {}, "geometry": station_buf_ll.__geo_interface__},
        name="신규 커버 반경",
        style_function=lambda x: {"color": "#00aa00", "weight": 2, "fillOpacity": 0.03},
    ).add_to(m)

    # (E) 새로 커버된 비커버 영역(교집합)
    if newly_covered_geom is not None and (not newly_covered_geom.is_empty):
        newly_ll = gpd.GeoSeries([newly_covered_geom], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
        folium.GeoJson(
            {"type": "Feature", "properties": {}, "geometry": newly_ll.__geo_interface__},
            name="새로 커버된 비커버(교집합)",
            style_function=lambda x: {"color": "#008800", "weight": 2, "fillOpacity": 0.25},
        ).add_to(m)

    # (F) 5분 네트워크 edge
    if len(gdf_edges_ll) > 0:
        folium.GeoJson(
            gdf_edges_ll,
            name=f"5분 네트워크({CUTOFF_MIN}min)",
            style_function=lambda x: {"color": "#0055ff", "weight": 3, "opacity": 0.85},
        ).add_to(m)
    else:
        st.info("5분 네트워크가 비었습니다. 그래프반경/데이터/OSM 상태를 확인하세요.")

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=680)


# =========================================================
# 9) 진단
# =========================================================
with st.expander("진단(파일/CRS/데이터)"):
    st.write("GRID_SHP:", GRID_SHP, "(exists:", os.path.exists(GRID_SHP), ")")
    st.write("UNCOVERED_GPKG:", UNCOVERED_GPKG, "(exists:", os.path.exists(UNCOVERED_GPKG), ")")
    st.write("ADMIN_GPKG:", ADMIN_GPKG, "(exists:", os.path.exists(ADMIN_GPKG), ")")
    st.write("ADMIN_SHP :", ADMIN_SHP,  "(exists:", os.path.exists(ADMIN_SHP), ")")
    st.write("grid crs:", str(gdf_grid.crs))
    st.write("grid cols:", list(gdf_grid.columns))
    st.write("grid rows:", len(gdf_grid))
    st.write("uncovered polys:", len(gdf_unc))
    st.write("admin rows:", len(gdf_admin))
