# app.py
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


# =========================================================
# 0) PATHS (GitHub 기준: app.py와 같은 폴더)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ✅ 전수 격자 SHP (폴더 내 .shp/.shx/.dbf/.prj 모두 있어야 함)
GRID_SHP = os.path.join(DATA_DIR, "nlsp_021001001.shp")

# ✅ 비커버 폴리곤(선택) - 없으면 자동으로 전체 uncovered=False 처리
UNCOVERED_GPKG = os.path.join(DATA_DIR, "demo_uncovered.gpkg")

GRID_ID_COL = "gid"
GRID_POP_COL = "val"     # 없으면 pop=0 처리
TARGET_CRS = 5179        # 분석용(거리 m)
MAP_CRS = 4326           # 지도용(WGS84)


# =========================================================
# 1) Streamlit Page
# =========================================================
st.set_page_config(page_title="5강 | Streamlit + Pydeck + OSMnx", layout="wide")

st.title("🚲 5강 | Streamlit 대시보드: 격자 선택 → KPI 즉석 계산 → 좌(Pydeck) / 우(5분 네트워크)")
st.caption("우측은 선택 격자 중심점에서 시작해 OSMnx+NetworkX로 5분(300초) 내 도달 가능한 네트워크 라인을 즉석 계산해 표시한다.")


# =========================================================
# 2) Loaders (캐시)
#    - GeoDataFrame / Shapely 객체는 cache key hashing에서 문제 날 수 있어
#      입력 인자는 'path(str)' 같이 해시 가능한 값만 넣는 구조로 구성
# =========================================================
@st.cache_data(show_spinner=True)
def load_grid_shp(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"GRID_SHP not found: {path}")

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError("GRID_SHP CRS is None. (.prj 확인)")

    gdf = gdf.to_crs(TARGET_CRS)

    if GRID_ID_COL not in gdf.columns:
        raise ValueError(f"GRID_ID_COL='{GRID_ID_COL}' not found in grid shapefile")

    gdf[GRID_ID_COL] = gdf[GRID_ID_COL].astype(str)

    if GRID_POP_COL in gdf.columns:
        gdf["pop"] = pd.to_numeric(gdf[GRID_POP_COL], errors="coerce").fillna(0).astype(float)
    elif "pop" in gdf.columns:
        gdf["pop"] = pd.to_numeric(gdf["pop"], errors="coerce").fillna(0).astype(float)
    else:
        gdf["pop"] = 0.0

    # geometry 안정화
    gdf["geometry"] = gdf.geometry.buffer(0)

    keep_cols = [GRID_ID_COL, "pop", "geometry"]
    return gdf[keep_cols].copy()


@st.cache_data(show_spinner=True)
def load_uncovered(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=TARGET_CRS)

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError("UNCOVERED_GPKG CRS is None.")

    gdf = gdf.to_crs(TARGET_CRS)
    gdf["geometry"] = gdf.geometry.buffer(0)

    # geometry만 쓰면 됨
    return gdf[["geometry"]].copy()


@st.cache_data(show_spinner=False)
def attach_is_uncovered(_grid_path: str, _unc_path: str) -> gpd.GeoDataFrame:
    """
    캐시 키에 GeoDataFrame이 들어가면 UnhashableParamError 나기 쉬움.
    그래서 path(str)만 입력으로 받고, 내부에서 로딩 후 계산해서 반환.
    """
    gdf_grid_5179 = load_grid_shp(_grid_path)
    gdf_unc_5179 = load_uncovered(_unc_path)

    g = gdf_grid_5179.copy()
    if len(gdf_unc_5179) == 0:
        g["is_uncovered"] = False
        return g

    unc_union = gdf_unc_5179.geometry.union_all()
    g["is_uncovered"] = g.geometry.intersects(unc_union)
    return g


def to_polygon_records(gdf_4326: gpd.GeoDataFrame):
    """
    pydeck PolygonLayer 입력용 레코드 생성
    """
    records = []
    for gid, popv, is_unc, elev, geom in zip(
        gdf_4326[GRID_ID_COL].astype(str).tolist(),
        gdf_4326["pop"].tolist(),
        gdf_4326["is_uncovered"].tolist(),
        gdf_4326["elev"].tolist(),
        gdf_4326.geometry.tolist(),
    ):
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue

        for poly in polys:
            records.append(
                {
                    "gid": gid,
                    "pop": float(popv),
                    "is_uncovered": bool(is_unc),
                    "elev": float(elev),
                    "polygon": list(poly.exterior.coords),
                }
            )
    return records


# =========================================================
# 3) OSMnx Graph Builder (캐시)
#    - shapely polygon 같은 객체를 cache key로 넣으면 해시 에러/피클 에러 나기 쉬움
#    - 그래서 "선택 중심점(lat/lon) + dist"만으로 그래프를 만든다
# =========================================================
@st.cache_resource(show_spinner=True)
def build_osm_graph_from_point(lat: float, lon: float, dist_m: int, network_type: str = "walk"):
    ox.settings.log_console = False

    G = ox.graph_from_point(
        (lat, lon),
        dist=int(dist_m),
        network_type=network_type,
        simplify=True,
    )

    # ✅ OSMnx 버전 호환: 2.x는 ox.distance.add_edge_lengths
    try:
        G = ox.distance.add_edge_lengths(G)
    except Exception:
        # 구버전 대비
        G = ox.add_edge_lengths(G)

    return G


def add_travel_time(G, speed_m_per_s: float):
    # edge travel_time(초) 추가 (in-place)
    sp = float(speed_m_per_s)
    for u, v, k, data in G.edges(keys=True, data=True):
        length_m = float(data.get("length", 0.0))
        data["travel_time"] = (length_m / sp) if sp > 0 else np.inf
    return G


def compute_reachable_edges_gdf(G, source_node: int, cutoff_sec: int) -> gpd.GeoDataFrame:
    # 5분 내 도달 가능한 노드 집합
    lengths = nx.single_source_dijkstra_path_length(
        G, source_node, cutoff=float(cutoff_sec), weight="travel_time"
    )
    reachable_nodes = set(lengths.keys())

    # induced subgraph
    SG = G.subgraph(reachable_nodes).copy()

    # edge gdf로 변환
    gdf_edges = ox.graph_to_gdfs(SG, nodes=False, edges=True, fill_edge_geometry=True)
    if gdf_edges.crs is None:
        gdf_edges = gdf_edges.set_crs(MAP_CRS)
    else:
        gdf_edges = gdf_edges.to_crs(MAP_CRS)

    # 보기용 컬럼
    if "length" in gdf_edges.columns:
        gdf_edges["length_m"] = gdf_edges["length"].astype(float)

    if "travel_time" in gdf_edges.columns:
        gdf_edges["time_s"] = gdf_edges["travel_time"].astype(float)

    return gdf_edges.reset_index(drop=True)


# =========================================================
# 4) Data Load
# =========================================================
with st.spinner("데이터 로딩 중..."):
    gdf_grid = attach_is_uncovered(GRID_SHP, UNCOVERED_GPKG)

all_gids = gdf_grid[GRID_ID_COL].astype(str).tolist()
if len(all_gids) == 0:
    st.error("GRID_SHP에서 격자를 불러오지 못했습니다. 파일/경로/CRS를 확인하세요.")
    st.stop()


# =========================================================
# 5) Sidebar Controls
# =========================================================
st.sidebar.header("설정")

sel_gid = st.sidebar.selectbox("전수 격자 gid 선택", options=all_gids, index=0)

RADIUS_M = st.sidebar.slider("KPI 반경(m) (좌측/상단 KPI용)", 300, 3000, 1250, 50)

speed_mps = st.sidebar.slider("보행 속도(m/s) (우측 네트워크 시간 계산)", 0.8, 2.0, 1.4, 0.1)
cutoff_min = st.sidebar.slider("네트워크 컷오프(분)", 1, 15, 5, 1)
cutoff_sec = int(cutoff_min * 60)

graph_dist_m = st.sidebar.slider("그래프 다운로드 반경(m) (우측 OSMnx)", 1000, 8000, 3500, 250)

st.sidebar.caption("우측 네트워크는 travel_time=length/speed(초)로 계산한다.")


# =========================================================
# 6) KPI 즉석 계산 (선택 gid 중심점 반경)
# =========================================================
def compute_kpi_for_gid(gdf_grid_5179: gpd.GeoDataFrame, sel_gid: str, radius_m: float):
    row = gdf_grid_5179.loc[gdf_grid_5179[GRID_ID_COL] == str(sel_gid)]
    if len(row) == 0:
        return None

    sel_poly = row.geometry.iloc[0]
    sel_center = sel_poly.centroid
    circle = sel_center.buffer(float(radius_m))

    in_circle = gdf_grid_5179.geometry.intersects(circle)
    gdf_in = gdf_grid_5179.loc[in_circle, [GRID_ID_COL, "pop", "is_uncovered", "geometry"]].copy()

    total_pop = float(gdf_in["pop"].sum())
    unc_pop = float(gdf_in.loc[gdf_in["is_uncovered"] == True, "pop"].sum())
    cov_pop = total_pop - unc_pop
    unc_rate = (unc_pop / total_pop) if total_pop > 0 else 0.0

    return {
        "sel_center_5179": sel_center,
        "circle_5179": circle,
        "cells": int(len(gdf_in)),
        "total_pop": total_pop,
        "uncovered_pop": unc_pop,
        "covered_pop": cov_pop,
        "uncovered_rate": unc_rate,
        "gdf_in_5179": gdf_in,
    }


kpi = compute_kpi_for_gid(gdf_grid, sel_gid, RADIUS_M)
if kpi is None:
    st.error("선택 gid를 grid에서 찾지 못했습니다. gid 컬럼/형식을 확인하세요.")
    st.stop()


# KPI cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("선택 gid", str(sel_gid))
c2.metric("반경 내 격자 수", f"{kpi['cells']:,}")
c3.metric("총 인구", f"{kpi['total_pop']:,.0f}")
c4.metric("비커버 인구", f"{kpi['uncovered_pop']:,.0f}")
c5.metric("비커버 비율", f"{kpi['uncovered_rate']*100:.2f}%")


# =========================================================
# 7) Layout: 좌(Pydeck) / 우(즉석 네트워크)
# =========================================================
left, right = st.columns([1, 1])


# -------------------------
# LEFT: Pydeck (선택 반경 내 격자 3D)
# -------------------------
with left:
    st.subheader("좌측: Pydeck 3D 격자 + KPI 반경")

    gdf_ll = kpi["gdf_in_5179"].to_crs(MAP_CRS).copy()

    pop = gdf_ll["pop"].clip(lower=0).astype(float)
    cap_val = float(pop.quantile(0.995)) if len(pop) > 0 else 0.0
    pop_capped = np.minimum(pop, cap_val) if cap_val > 0 else pop
    gdf_ll["elev"] = (np.power(pop_capped, 1.80) * 0.02).astype(float)

    records = to_polygon_records(gdf_ll)

    # KPI 원(circle) + 중심점
    circle_ll = gpd.GeoSeries([kpi["circle_5179"]], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
    circle_coords = list(circle_ll.exterior.coords)

    sel_center_ll = gpd.GeoSeries([kpi["sel_center_5179"]], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]

    layer_blocks = pdk.Layer(
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

    layer_circle = pdk.Layer(
        "PolygonLayer",
        data=[{"polygon": circle_coords}],
        get_polygon="polygon",
        filled=False,
        stroked=True,
        get_line_color=[30, 30, 30, 220],
        get_line_width=120,
    )

    view = pdk.ViewState(
        latitude=float(sel_center_ll.y),
        longitude=float(sel_center_ll.x),
        zoom=14,
        pitch=65,
        bearing=20,
    )

    deck = pdk.Deck(
        layers=[layer_blocks, layer_circle],
        initial_view_state=view,
        map_style="carto-positron",
        tooltip={"text": "gid: {gid}\npop: {pop}\nuncovered: {is_uncovered}"},
    )

    # ✅ Streamlit 경고 대응: use_container_width 대신 width="stretch"
    st.pydeck_chart(deck, width="stretch")


# -------------------------
# RIGHT: Folium (즉석 OSMnx+NetworkX 5분 네트워크)
# -------------------------
with right:
    st.subheader("우측: OSMnx+NetworkX 즉석 계산 5분 네트워크")

    # 선택 중심점(4326) 계산
    sel_center_ll = gpd.GeoSeries([kpi["sel_center_5179"]], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
    lon, lat = float(sel_center_ll.x), float(sel_center_ll.y)

    with st.spinner(f"OSM 그래프 다운로드/캐시 확인... (dist={graph_dist_m}m)"):
        G = build_osm_graph_from_point(lat=lat, lon=lon, dist_m=int(graph_dist_m), network_type="walk")
        G = add_travel_time(G, speed_m_per_s=float(speed_mps))

    try:
        source_node = ox.distance.nearest_nodes(G, X=lon, Y=lat)
    except Exception as e:
        st.error(f"nearest_nodes 실패: {e}")
        st.stop()

    with st.spinner(f"{cutoff_min}분 네트워크 계산 중... (cutoff={cutoff_sec}s)"):
        gdf_edges = compute_reachable_edges_gdf(G, source_node=int(source_node), cutoff_sec=int(cutoff_sec))

    # KPI: 네트워크 규모 요약
    n_edges = int(len(gdf_edges))
    total_len_km = float(gdf_edges["length_m"].sum() / 1000.0) if "length_m" in gdf_edges.columns else np.nan
    c6, c7 = st.columns(2)
    c6.metric("네트워크 edge 수", f"{n_edges:,}")
    c7.metric("네트워크 총 길이(km)", f"{total_len_km:,.2f}" if not np.isnan(total_len_km) else "-")

    # Folium 지도
    m = folium.Map(
        location=[lat, lon],
        zoom_start=14,
        tiles="cartodbpositron",
    )

    # 시작점
    folium.Marker(
        location=[lat, lon],
        tooltip=f"gid={sel_gid} (nearest node: {source_node})",
        icon=folium.Icon(color="red", icon="play", prefix="fa"),
    ).add_to(m)

    # 네트워크 edge
    if len(gdf_edges) > 0:
        tooltip_fields = []
        aliases = []
        if "length_m" in gdf_edges.columns:
            tooltip_fields.append("length_m")
            aliases.append("length(m)")
        if "time_s" in gdf_edges.columns:
            tooltip_fields.append("time_s")
            aliases.append("time(s)")

        folium.GeoJson(
            gdf_edges,
            name=f"reachable_network_{cutoff_min}min",
            style_function=lambda x: {"color": "#0055ff", "weight": 3, "opacity": 0.85},
            tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=aliases) if len(tooltip_fields) > 0 else None,
        ).add_to(m)
    else:
        st.info("도달 가능한 네트워크가 비어 있습니다. 그래프 반경(dist) / 위치 / 속도 / cutoff를 확인하세요.")

    # KPI 반경 원도 같이 표시
    circle_ll = gpd.GeoSeries([kpi["circle_5179"]], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
    folium.GeoJson(
        {"type": "Feature", "properties": {}, "geometry": circle_ll.__geo_interface__},
        name="kpi_radius",
        style_function=lambda x: {"color": "#111111", "weight": 2, "opacity": 0.8},
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=650)


# =========================================================
# 8) 디버그(필요시)
# =========================================================
with st.expander("데이터/그래프 진단"):
    st.write("GRID_SHP:", GRID_SHP, "(exists:", os.path.exists(GRID_SHP), ")")
    st.write("UNCOVERED_GPKG:", UNCOVERED_GPKG, "(exists:", os.path.exists(UNCOVERED_GPKG), ")")
    st.write("grid CRS:", str(gdf_grid.crs))
    st.write("grid columns:", list(gdf_grid.columns))

    # 그래프 정보는 우측이 실행되어야 생성됨
    try:
        st.write("OSM graph nodes:", len(G.nodes), "edges:", len(G.edges))
    except Exception:
        st.write("OSM graph: (우측 실행 전)")
