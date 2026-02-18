import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import streamlit as st
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx
from shapely.ops import unary_union
from shapely.geometry import mapping

# =========================================================
# 0) 상수 / 경로
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

ADMIN_SHP  = os.path.join(DATA_DIR, "BND_ADM_DONG_PG.gpkg")
BUS_XLSX   = os.path.join(DATA_DIR, "서울시버스정류소위치정보(20260108).xlsx")
SUBWAY_CSV = os.path.join(DATA_DIR, "서울교통공사_1_8호선 역사 좌표(위경도) 정보_20250814.csv")
GRID_SHP   = os.path.join(DATA_DIR, "nlsp_021001001.shp")

NAMHYEON_ID = "11210630"

TARGET_CRS = 5179
MAP_CRS    = 4326

BUS_BUFFER_M   = 300.0
SUB_BUFFER_M   = 500.0
GRAPH_BUFFER_M = 1500.0
EDGE_BUFFER_M  = 25.0

MAP_HEIGHT_PX = 600

# =========================================================
# 1) 페이지 설정
# =========================================================

st.set_page_config(
    page_title="남현동 대중교통 커버리지 비교: 직선 버퍼 vs 네트워크",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding: 1.2rem 1.0rem 1.6rem 1.0rem; max-width: none; }
      h1, h2, h3 { letter-spacing: -0.02em; }
      .stApp h1, div[data-testid="stMarkdownContainer"] h1 { text-align: center; width: 100%; }
      div[data-testid="stMarkdownContainer"] h1 { margin-top: 0.2rem; margin-bottom: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("남현동 대중교통 커버리지 분석: 직선 버퍼 vs 네트워크 기반")
st.caption("버스정류장 300 m / 지하철역 500 m 도보 기준 — 남현동(11210630)")

# =========================================================
# 2) 데이터 로드 + 분석 (캐싱)
# =========================================================

def _to_ll(geom):
    """EPSG:5179 → 4326 변환 헬퍼"""
    return gpd.GeoSeries([geom], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]


@st.cache_resource(show_spinner=False)
def compute_all():
    # ── 행정동 ──
    gdf_admin = gpd.read_file(ADMIN_SHP)
    gdf_admin["region_id"] = gdf_admin["ADM_CD"].astype(str).str.strip()
    gdf_admin = gdf_admin.to_crs(TARGET_CRS)
    gdf_nam = gdf_admin[gdf_admin["region_id"] == NAMHYEON_ID].copy()
    if len(gdf_nam) == 0:
        raise ValueError(f"행정동 {NAMHYEON_ID} 을 찾을 수 없습니다.")
    nam_union = unary_union(gdf_nam.geometry)

    # ── 버스정류장 ──
    bus_raw = pd.read_excel(BUS_XLSX)
    bus_raw["X좌표"] = pd.to_numeric(bus_raw["X좌표"], errors="coerce")
    bus_raw["Y좌표"] = pd.to_numeric(bus_raw["Y좌표"], errors="coerce")
    bus_raw = bus_raw.dropna(subset=["X좌표", "Y좌표"])
    gdf_bus = gpd.GeoDataFrame(
        bus_raw, geometry=gpd.points_from_xy(bus_raw["X좌표"], bus_raw["Y좌표"]), crs=MAP_CRS
    ).to_crs(TARGET_CRS)
    gdf_bus_nam = gdf_bus[gdf_bus.geometry.within(nam_union)].copy()
    gdf_bus_nam["stop_type"] = "bus"

    # ── 지하철 ──
    try:
        sub_raw = pd.read_csv(SUBWAY_CSV, encoding="utf-8")
    except UnicodeDecodeError:
        sub_raw = pd.read_csv(SUBWAY_CSV, encoding="cp949")
    sub_raw["경도"] = pd.to_numeric(sub_raw["경도"], errors="coerce")
    sub_raw["위도"] = pd.to_numeric(sub_raw["위도"], errors="coerce")
    sub_raw = sub_raw.dropna(subset=["경도", "위도"])
    gdf_sub = gpd.GeoDataFrame(
        sub_raw, geometry=gpd.points_from_xy(sub_raw["경도"], sub_raw["위도"]), crs=MAP_CRS
    ).to_crs(TARGET_CRS)
    gdf_sub_nam = gdf_sub[gdf_sub.geometry.within(nam_union)].copy()
    gdf_sub_nam["stop_type"] = "subway"

    # ── 인구 격자 ──
    gdf_grid = gpd.read_file(GRID_SHP).to_crs(TARGET_CRS)
    gdf_grid["gid"] = gdf_grid["gid"].astype(str)
    gdf_grid["pop"] = pd.to_numeric(gdf_grid.get("val", 0), errors="coerce").fillna(0.0)
    gdf_grid_nam = gpd.clip(
        gdf_grid[gdf_grid.geometry.intersects(nam_union)], gdf_nam
    )[["gid", "pop", "geometry"]].copy()

    # ── (A) 직선 버퍼 커버리지 ──
    bufs = []
    if len(gdf_bus_nam):
        bufs.append(unary_union(gdf_bus_nam.geometry.buffer(BUS_BUFFER_M)))
    if len(gdf_sub_nam):
        bufs.append(unary_union(gdf_sub_nam.geometry.buffer(SUB_BUFFER_M)))
    cover_buf = unary_union(bufs) if bufs else None
    uncov_buf = nam_union.difference(cover_buf) if cover_buf else nam_union

    # ── (B) Isochrone(네트워크) 커버리지 ──
    poly_graph_ll = (
        gpd.GeoSeries([nam_union.buffer(GRAPH_BUFFER_M)], crs=TARGET_CRS)
        .to_crs(MAP_CRS).iloc[0]
    )
    ox.settings.log_console = False
    G = ox.graph_from_polygon(poly_graph_ll, network_type="walk", simplify=True)

    gdf_stops = gpd.GeoDataFrame(
        pd.concat([gdf_bus_nam, gdf_sub_nam], ignore_index=True),
        geometry="geometry", crs=TARGET_CRS,
    )
    gdf_stops_ll = gdf_stops.to_crs(MAP_CRS).copy()
    gdf_stops_ll["v_node"] = ox.distance.nearest_nodes(
        G, X=gdf_stops_ll.geometry.x.values, Y=gdf_stops_ll.geometry.y.values
    )

    iso_polys = []
    for _, r in gdf_stops_ll.iterrows():
        v = int(r["v_node"])
        cutoff = BUS_BUFFER_M if r["stop_type"] == "bus" else SUB_BUFFER_M
        try:
            Gsub = nx.ego_graph(G, v, radius=cutoff, distance="length", undirected=True)
        except Exception:
            continue
        if Gsub.number_of_edges() == 0:
            continue
        _, gdf_edges = ox.graph_to_gdfs(Gsub, nodes=True, edges=True, fill_edge_geometry=True)
        poly_m = unary_union(gdf_edges.to_crs(TARGET_CRS).geometry.buffer(EDGE_BUFFER_M))
        if poly_m and not poly_m.is_empty:
            iso_polys.append(poly_m)

    cover_iso = unary_union(iso_polys) if iso_polys else None
    uncov_iso = nam_union.difference(cover_iso) if cover_iso else nam_union

    # ── KPI 계산 ──
    admin_area = nam_union.area
    buf_area = uncov_buf.area if uncov_buf and not uncov_buf.is_empty else 0
    iso_area = uncov_iso.area if uncov_iso and not uncov_iso.is_empty else 0

    centroids = gdf_grid_nam.geometry.centroid
    buf_mask = centroids.within(uncov_buf) if (uncov_buf and not uncov_buf.is_empty) else pd.Series(False, index=gdf_grid_nam.index)
    iso_mask = centroids.within(uncov_iso) if (uncov_iso and not uncov_iso.is_empty) else pd.Series(False, index=gdf_grid_nam.index)

    buf_pop = float(gdf_grid_nam.loc[buf_mask, "pop"].sum())
    iso_pop = float(gdf_grid_nam.loc[iso_mask, "pop"].sum())
    total_pop = float(gdf_grid_nam["pop"].sum())

    # 추가 발견 비커버 인구: 버퍼로는 커버인데 isochrone으로는 비커버
    false_covered = (~buf_mask) & iso_mask
    additional_pop = float(gdf_grid_nam.loc[false_covered, "pop"].sum())

    kpi = {
        "buf_uncov_km2": buf_area / 1e6,
        "iso_uncov_km2": iso_area / 1e6,
        "buf_uncov_pop": buf_pop,
        "iso_uncov_pop": iso_pop,
        "buf_ratio": buf_area / admin_area if admin_area > 0 else 0,
        "iso_ratio": iso_area / admin_area if admin_area > 0 else 0,
        "additional_pop": additional_pop,
        "total_pop": total_pop,
    }

    # ── 표시용 4326 변환 ──
    nam_ll = gdf_nam.to_crs(MAP_CRS)
    bounds = nam_ll.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    cover_buf_ll  = _to_ll(cover_buf.intersection(nam_union)) if cover_buf else None
    uncov_buf_ll  = _to_ll(uncov_buf) if (uncov_buf and not uncov_buf.is_empty) else None
    cover_iso_cl  = cover_iso.intersection(nam_union) if cover_iso else None
    cover_iso_ll  = _to_ll(cover_iso_cl.simplify(5)) if cover_iso_cl else None
    uncov_iso_ll  = _to_ll(uncov_iso.simplify(5)) if (uncov_iso and not uncov_iso.is_empty) else None

    bus_ll = gdf_bus_nam.to_crs(MAP_CRS)
    sub_ll = gdf_sub_nam.to_crs(MAP_CRS)

    return {
        "kpi": kpi,
        "nam_ll": nam_ll,
        "bounds": bounds,
        "center": center,
        "cover_buf_ll": cover_buf_ll,
        "uncov_buf_ll": uncov_buf_ll,
        "cover_iso_ll": cover_iso_ll,
        "uncov_iso_ll": uncov_iso_ll,
        "bus_ll": bus_ll,
        "sub_ll": sub_ll,
    }

# =========================================================
# 3) 지도 생성 함수
# =========================================================

def _add_stops(m, bus_ll, sub_ll):
    """버스/지하철 마커를 지도에 추가"""
    for _, r in bus_ll.iterrows():
        folium.CircleMarker(
            location=[r.geometry.y, r.geometry.x],
            radius=4, color="#0066ff", fill=True, fill_opacity=0.8,
            tooltip=f"버스정류장 | {r.get('정류소명', '')}",
        ).add_to(m)
    for _, r in sub_ll.iterrows():
        folium.CircleMarker(
            location=[r.geometry.y, r.geometry.x],
            radius=6, color="#ff6600", fill=True, fill_opacity=0.9,
            tooltip="지하철역",
        ).add_to(m)


def create_buffer_map(d):
    m = folium.Map(location=d["center"], zoom_start=14, tiles="cartodbpositron")

    folium.GeoJson(
        d["nam_ll"], name="행정동 경계",
        style_function=lambda x: {"fillOpacity": 0.03, "color": "#444", "weight": 3},
    ).add_to(m)

    if d["cover_buf_ll"]:
        folium.GeoJson(
            mapping(d["cover_buf_ll"]), name="커버 영역 (직선 버퍼)",
            style_function=lambda x: {"fillOpacity": 0.25, "fillColor": "#28a745", "color": "#28a745", "weight": 1},
        ).add_to(m)

    if d["uncov_buf_ll"]:
        folium.GeoJson(
            mapping(d["uncov_buf_ll"]), name="비커버 영역 (직선 버퍼)",
            style_function=lambda x: {"fillOpacity": 0.35, "fillColor": "#cc0000", "color": "#cc0000", "weight": 2},
        ).add_to(m)

    _add_stops(m, d["bus_ll"], d["sub_ll"])
    folium.LayerControl(collapsed=False).add_to(m)
    b = d["bounds"]
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    return m


def create_isochrone_map(d):
    m = folium.Map(location=d["center"], zoom_start=14, tiles="cartodbpositron")

    folium.GeoJson(
        d["nam_ll"], name="행정동 경계",
        style_function=lambda x: {"fillOpacity": 0.03, "color": "#444", "weight": 3},
    ).add_to(m)

    if d["cover_iso_ll"]:
        folium.GeoJson(
            mapping(d["cover_iso_ll"]), name="커버 영역 (네트워크 Isochrone)",
            style_function=lambda x: {"fillOpacity": 0.20, "fillColor": "#0066ff", "color": "#0066ff", "weight": 1},
        ).add_to(m)

    if d["uncov_iso_ll"]:
        folium.GeoJson(
            mapping(d["uncov_iso_ll"]), name="비커버 영역 (네트워크 Isochrone)",
            style_function=lambda x: {"fillOpacity": 0.30, "fillColor": "#7a00cc", "color": "#7a00cc", "weight": 2},
        ).add_to(m)

    _add_stops(m, d["bus_ll"], d["sub_ll"])
    folium.LayerControl(collapsed=False).add_to(m)
    b = d["bounds"]
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    return m

# =========================================================
# 4) 메인 레이아웃
# =========================================================

with st.spinner("OSM 도보 네트워크 다운로드 및 Isochrone 계산 중..."):
    data = compute_all()

k = data["kpi"]

# ── KPI ──
st.markdown("---")
st.subheader("KPI 비교")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        label="비커버 면적",
        value=f"{k['iso_uncov_km2']:.3f} km\u00b2",
        delta=f"{k['iso_uncov_km2'] - k['buf_uncov_km2']:+.3f} km\u00b2 (네트워크 \u2212 버퍼)",
        delta_color="inverse",
    )

with c2:
    st.metric(
        label="비커버 인구",
        value=f"{k['iso_uncov_pop']:,.0f} 명",
        delta=f"{k['iso_uncov_pop'] - k['buf_uncov_pop']:+,.0f} 명",
        delta_color="inverse",
    )

with c3:
    st.metric(
        label="비커버 비율",
        value=f"{k['iso_ratio']:.1%}",
        delta=f"{(k['iso_ratio'] - k['buf_ratio'])*100:+.1f} %p",
        delta_color="inverse",
    )

with c4:
    st.metric(
        label="추가 발견 비커버 인구",
        value=f"{k['additional_pop']:,.0f} 명",
        help="직선 버퍼로는 커버된 것처럼 보이지만, 실제 도보 네트워크로는 도달 불가능한 인구",
    )

# ── 좌/우 지도 ──
st.markdown("---")
col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.subheader("직선 버퍼 기반 분석")
    st_folium(create_buffer_map(data), width=None, height=MAP_HEIGHT_PX, key="map_buf", returned_objects=[])

with col_r:
    st.subheader("네트워크(Isochrone) 기반 분석")
    st_folium(create_isochrone_map(data), width=None, height=MAP_HEIGHT_PX, key="map_iso", returned_objects=[])

# ── 방법론 비교 ──
with st.expander("분석 방법론 비교"):
    st.markdown(
        """
| 항목 | 직선 버퍼 | 네트워크 기반 (Isochrone) |
|------|-----------|--------------------------|
| **방식** | 정류장 중심 원형 버퍼 (300 m / 500 m) | OSMnx 도보 네트워크 ego_graph + 도로 폭 25 m 버퍼 |
| **장점** | 계산 빠름, 직관적 | 실제 보행 가능 경로 반영 |
| **단점** | 건물/하천/도로 등 장애물 미반영 | OSM 네트워크 다운로드 필요, 계산 시간 |
| **비커버 판단** | 원 바깥 = 비커버 | 도보로 도달 불가 = 비커버 |
        """
    )

