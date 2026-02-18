import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
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

TARGET_IDS = {"11210630": "남현동", "11210540": "청림동"}

TARGET_CRS = 5179
MAP_CRS    = 4326

BUS_BUFFER_M   = 300.0
SUB_BUFFER_M   = 500.0
GRAPH_BUFFER_M = 1500.0
EDGE_BUFFER_M  = 25.0
WALK_5MIN_M    = 5 * 60 * 1.4   # 420m

MAP_HEIGHT_PX = 650
MAX_ROUTES_PER_TOPGRID = 10

# =========================================================
# 1) 페이지 설정
# =========================================================
st.set_page_config(page_title="대중교통 커버리지 비교", layout="wide")

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

st.title("대중교통 커버리지 분석: 직선 버퍼 vs 네트워크 기반")
st.caption("버스 300 m / 지하철 500 m 기준 · TOP 격자→버스정류장 5분 경로 표시")

# =========================================================
# 2) 드롭다운(행정동 선택)
# =========================================================
st.markdown("---")
rid = st.selectbox(
    "행정동 선택",
    options=list(TARGET_IDS.keys()),
    format_func=lambda x: f"{TARGET_IDS.get(x, x)} ({x})",
    index=0,
)
st.caption(f"선택 행정동: {TARGET_IDS.get(rid)}")

# =========================================================
# 3) 데이터 로드 (스크립트 흐름)
# =========================================================
with st.spinner("데이터 로드/분석 중... (OSM 네트워크 다운로드 포함)"):
    # ── (1) 행정동 ──
    gdf_admin = gpd.read_file(ADMIN_SHP)
    gdf_admin["region_id"] = gdf_admin["ADM_CD"].astype(str).str.strip()
    gdf_admin["region_nm"] = gdf_admin["ADM_NM"].astype(str).str.strip()
    gdf_admin = gdf_admin.to_crs(TARGET_CRS)

    gdf_sel = gdf_admin[gdf_admin["region_id"] == rid].copy()
    if len(gdf_sel) == 0:
        st.stop()
    region_nm = gdf_sel["region_nm"].iloc[0]
    sel_union = unary_union(gdf_sel.geometry)

    # 표시용(4326)
    sel_ll = gdf_sel.to_crs(MAP_CRS)
    bounds = sel_ll.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    # ── (2) 버스정류장 ──
    bus_raw = pd.read_excel(BUS_XLSX)
    bus_raw["X좌표"] = pd.to_numeric(bus_raw["X좌표"], errors="coerce")
    bus_raw["Y좌표"] = pd.to_numeric(bus_raw["Y좌표"], errors="coerce")
    bus_raw = bus_raw.dropna(subset=["X좌표", "Y좌표"])

    gdf_bus = gpd.GeoDataFrame(
        bus_raw,
        geometry=gpd.points_from_xy(bus_raw["X좌표"], bus_raw["Y좌표"]),
        crs=MAP_CRS,
    ).to_crs(TARGET_CRS)
    gdf_bus_sel = gdf_bus[gdf_bus.geometry.within(sel_union)].copy()

    # ── (3) 지하철역 ──
    try:
        sub_raw = pd.read_csv(SUBWAY_CSV, encoding="utf-8")
    except UnicodeDecodeError:
        sub_raw = pd.read_csv(SUBWAY_CSV, encoding="cp949")

    sub_raw["경도"] = pd.to_numeric(sub_raw["경도"], errors="coerce")
    sub_raw["위도"] = pd.to_numeric(sub_raw["위도"], errors="coerce")
    sub_raw = sub_raw.dropna(subset=["경도", "위도"])

    gdf_sub = gpd.GeoDataFrame(
        sub_raw,
        geometry=gpd.points_from_xy(sub_raw["경도"], sub_raw["위도"]),
        crs=MAP_CRS,
    ).to_crs(TARGET_CRS)
    gdf_sub_sel = gdf_sub[gdf_sub.geometry.within(sel_union)].copy()

    # ── (4) 인구격자 ──
    gdf_grid = gpd.read_file(GRID_SHP).to_crs(TARGET_CRS)
    gdf_grid["gid"] = gdf_grid["gid"].astype(str)
    gdf_grid["pop"] = pd.to_numeric(gdf_grid.get("val", 0), errors="coerce").fillna(0.0)

    gdf_grid_sel = gpd.clip(
        gdf_grid[gdf_grid.geometry.intersects(sel_union)],
        gdf_sel
    )[["gid", "pop", "geometry"]].copy()

    gdf_grid_sel["centroid_m"] = gdf_grid_sel.geometry.centroid  # 5179 centroid

    # =========================================================
    # 4) (A) 직선 버퍼 커버/비커버
    # =========================================================
    bufs = []
    if len(gdf_bus_sel) > 0:
        bufs.append(unary_union(gdf_bus_sel.geometry.buffer(BUS_BUFFER_M)))
    if len(gdf_sub_sel) > 0:
        bufs.append(unary_union(gdf_sub_sel.geometry.buffer(SUB_BUFFER_M)))

    cover_buf = unary_union(bufs) if len(bufs) > 0 else None
    uncov_buf = sel_union.difference(cover_buf) if cover_buf else sel_union

    # =========================================================
    # 5) (B) 네트워크(Isochrone) 커버/비커버 + TOP격자→버스정류장 5분 최단경로
    # =========================================================
    poly_graph_ll = (
        gpd.GeoSeries([sel_union.buffer(GRAPH_BUFFER_M)], crs=TARGET_CRS)
        .to_crs(MAP_CRS)
        .iloc[0]
    )

    ox.settings.log_console = False
    G = ox.graph_from_polygon(poly_graph_ll, network_type="walk", simplify=True)

    # 버스/지하철 노드 매핑(4326)
    bus_ll = gdf_bus_sel.to_crs(MAP_CRS).copy()
    sub_ll = gdf_sub_sel.to_crs(MAP_CRS).copy()

    bus_nodes = []
    if len(bus_ll) > 0:
        bus_nodes = list(ox.distance.nearest_nodes(G, X=bus_ll.geometry.x.values, Y=bus_ll.geometry.y.values))

    # (B-1) isochrone 커버리지(정류장 기준 300/500m)
    # stops 합치기
    gdf_bus_sel2 = gdf_bus_sel.copy()
    gdf_bus_sel2["stop_type"] = "bus"
    gdf_sub_sel2 = gdf_sub_sel.copy()
    gdf_sub_sel2["stop_type"] = "subway"

    gdf_stops = gpd.GeoDataFrame(
        pd.concat([gdf_bus_sel2, gdf_sub_sel2], ignore_index=True),
        geometry="geometry",
        crs=TARGET_CRS,
    )
    gdf_stops_ll = gdf_stops.to_crs(MAP_CRS).copy()

    if len(gdf_stops_ll) > 0:
        gdf_stops_ll["v_node"] = ox.distance.nearest_nodes(
            G, X=gdf_stops_ll.geometry.x.values, Y=gdf_stops_ll.geometry.y.values
        )

    iso_polys = []
    for _, r in gdf_stops_ll.iterrows():
        v = int(r["v_node"])
        iso_cutoff = BUS_BUFFER_M if r["stop_type"] == "bus" else SUB_BUFFER_M
        try:
            Gsub_iso = nx.ego_graph(G, v, radius=float(iso_cutoff), distance="length", undirected=True)
        except Exception:
            continue

        if Gsub_iso.number_of_edges() == 0:
            continue

        _, gdf_edges = ox.graph_to_gdfs(Gsub_iso, nodes=True, edges=True, fill_edge_geometry=True)
        poly_m = unary_union(gdf_edges.to_crs(TARGET_CRS).geometry.buffer(EDGE_BUFFER_M))
        if poly_m is not None and (not poly_m.is_empty):
            iso_polys.append(poly_m)

    cover_iso = unary_union(iso_polys) if len(iso_polys) > 0 else None
    uncov_iso = sel_union.difference(cover_iso) if cover_iso else sel_union

    # =========================================================
    # 6) KPI + TOP 격자(버퍼/네트워크 각각)
    # =========================================================
    admin_area = sel_union.area

    buf_mask = gdf_grid_sel["centroid_m"].within(uncov_buf) if (uncov_buf and not uncov_buf.is_empty) else pd.Series(False, index=gdf_grid_sel.index)
    iso_mask = gdf_grid_sel["centroid_m"].within(uncov_iso) if (uncov_iso and not uncov_iso.is_empty) else pd.Series(False, index=gdf_grid_sel.index)

    buf_pop = float(gdf_grid_sel.loc[buf_mask, "pop"].sum())
    iso_pop = float(gdf_grid_sel.loc[iso_mask, "pop"].sum())
    total_pop = float(gdf_grid_sel["pop"].sum())

    buf_area = float(uncov_buf.area) if (uncov_buf and not uncov_buf.is_empty) else 0.0
    iso_area = float(uncov_iso.area) if (uncov_iso and not uncov_iso.is_empty) else 0.0

    false_covered = (~buf_mask) & iso_mask
    additional_pop = float(gdf_grid_sel.loc[false_covered, "pop"].sum())

    # TOP 격자
    top_buf = None
    top_iso = None

    if (uncov_buf is not None) and (not uncov_buf.is_empty):
        cands = gdf_grid_sel[gdf_grid_sel["centroid_m"].within(uncov_buf)]
        if len(cands) > 0:
            top_buf = cands.loc[cands["pop"].idxmax()].copy()

    if (uncov_iso is not None) and (not uncov_iso.is_empty):
        cands = gdf_grid_sel[gdf_grid_sel["centroid_m"].within(uncov_iso)]
        if len(cands) > 0:
            top_iso = cands.loc[cands["pop"].idxmax()].copy()

    # =========================================================
    # 7) TOP 격자 중심점 → 버스정류장 5분 최단경로(버퍼/네트워크 각각)
    # =========================================================
    routes_top_buf = []
    routes_top_iso = []

    # helper 없이 스크립트로 직접 작성(중복을 감수)
    if top_buf is not None and len(bus_nodes) > 0:
        top_buf_cent_ll = gpd.GeoSeries([top_buf["centroid_m"]], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
        src = ox.distance.nearest_nodes(G, X=float(top_buf_cent_ll.x), Y=float(top_buf_cent_ll.y))

        try:
            lengths = nx.single_source_dijkstra_path_length(G, src, cutoff=float(WALK_5MIN_M), weight="length")
            reachable_bus = [(bn, lengths[bn]) for bn in bus_nodes if bn in lengths]
            reachable_bus.sort(key=lambda x: x[1])
            reachable_bus = reachable_bus[:MAX_ROUTES_PER_TOPGRID]
        except Exception:
            reachable_bus = []

        for bn, _dist in reachable_bus:
            try:
                path_nodes = nx.shortest_path(G, source=src, target=bn, weight="length")
                line = ox.utils_graph.route_to_gdf(G, path_nodes, weight="length")["geometry"].unary_union
                if line is None or line.is_empty:
                    continue
                if line.geom_type == "LineString":
                    routes_top_buf.append(line)
                else:
                    parts = list(line.geoms)
                    parts.sort(key=lambda g: g.length if g else 0, reverse=True)
                    if parts and (not parts[0].is_empty):
                        routes_top_buf.append(parts[0])
            except Exception:
                continue

    if top_iso is not None and len(bus_nodes) > 0:
        top_iso_cent_ll = gpd.GeoSeries([top_iso["centroid_m"]], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
        src = ox.distance.nearest_nodes(G, X=float(top_iso_cent_ll.x), Y=float(top_iso_cent_ll.y))

        try:
            lengths = nx.single_source_dijkstra_path_length(G, src, cutoff=float(WALK_5MIN_M), weight="length")
            reachable_bus = [(bn, lengths[bn]) for bn in bus_nodes if bn in lengths]
            reachable_bus.sort(key=lambda x: x[1])
            reachable_bus = reachable_bus[:MAX_ROUTES_PER_TOPGRID]
        except Exception:
            reachable_bus = []

        for bn, _dist in reachable_bus:
            try:
                path_nodes = nx.shortest_path(G, source=src, target=bn, weight="length")
                line = ox.utils_graph.route_to_gdf(G, path_nodes, weight="length")["geometry"].unary_union
                if line is None or line.is_empty:
                    continue
                if line.geom_type == "LineString":
                    routes_top_iso.append(line)
                else:
                    parts = list(line.geoms)
                    parts.sort(key=lambda g: g.length if g else 0, reverse=True)
                    if parts and (not parts[0].is_empty):
                        routes_top_iso.append(parts[0])
            except Exception:
                continue

    # =========================================================
    # 8) 표시용 4326 geometry 만들기
    # =========================================================
    cover_buf_ll = None
    uncov_buf_ll = None
    cover_iso_ll = None
    uncov_iso_ll = None

    if cover_buf is not None:
        cover_buf_ll = gpd.GeoSeries([cover_buf.intersection(sel_union)], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
    if uncov_buf is not None and (not uncov_buf.is_empty):
        uncov_buf_ll = gpd.GeoSeries([uncov_buf], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]

    if cover_iso is not None:
        cover_iso_ll = gpd.GeoSeries([cover_iso.intersection(sel_union).simplify(5)], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
    if uncov_iso is not None and (not uncov_iso.is_empty):
        uncov_iso_ll = gpd.GeoSeries([uncov_iso.simplify(5)], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]

    top_buf_ll = None
    top_iso_ll = None
    if top_buf is not None:
        top_buf_ll = gpd.GeoDataFrame([top_buf], geometry="geometry", crs=TARGET_CRS).to_crs(MAP_CRS)
    if top_iso is not None:
        top_iso_ll = gpd.GeoDataFrame([top_iso], geometry="geometry", crs=TARGET_CRS).to_crs(MAP_CRS)

    kpi = dict(
        region_nm=region_nm,
        buf_uncov_km2=buf_area / 1e6,
        iso_uncov_km2=iso_area / 1e6,
        buf_uncov_pop=buf_pop,
        iso_uncov_pop=iso_pop,
        buf_ratio=(buf_area / admin_area) if admin_area > 0 else 0,
        iso_ratio=(iso_area / admin_area) if admin_area > 0 else 0,
        additional_pop=additional_pop,
        total_pop=total_pop,
    )

# =========================================================
# 9) KPI 출력
# =========================================================
st.markdown("---")
st.subheader(f"KPI 비교 ({kpi['region_nm']})")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        label="비커버 면적(네트워크)",
        value=f"{kpi['iso_uncov_km2']:.3f} km\u00b2",
        delta=f"{kpi['iso_uncov_km2'] - kpi['buf_uncov_km2']:+.3f} km\u00b2 (네트워크 − 버퍼)",
        delta_color="inverse",
    )
with c2:
    st.metric(
        label="비커버 인구(네트워크)",
        value=f"{kpi['iso_uncov_pop']:,.0f} 명",
        delta=f"{kpi['iso_uncov_pop'] - kpi['buf_uncov_pop']:+,.0f} 명",
        delta_color="inverse",
    )
with c3:
    st.metric(
        label="비커버 비율(네트워크)",
        value=f"{kpi['iso_ratio']:.1%}",
        delta=f"{(kpi['iso_ratio'] - kpi['buf_ratio'])*100:+.1f} %p",
        delta_color="inverse",
    )
with c4:
    st.metric(
        label="추가 발견 비커버 인구",
        value=f"{kpi['additional_pop']:,.0f} 명",
        help="직선 버퍼로는 커버된 것처럼 보이지만, 네트워크로는 도달 불가능한(비커버) 인구",
    )

# =========================================================
# 10) 지도 그리기(좌:버퍼 / 우:네트워크)
#     - 버스/지하철 아이콘 개선(AwesomeMarkers)
# =========================================================

# (공통) 숫자 뱃지 아이콘(최소 HTML)
def number_badge_html(n, bg):
    return f"""
    <div style="
      width:28px;height:28px;border-radius:999px;
      background:{bg};color:#fff;font-weight:800;font-size:14px;
      display:flex;align-items:center;justify-content:center;
      border:2px solid #fff; box-shadow:0 2px 8px rgba(0,0,0,0.35);
    ">{n}</div>
    """

# 버스/지하철 아이콘(AwesomeMarkers)
bus_icon = folium.Icon(color="blue", icon="bus", prefix="fa")
sub_icon = folium.Icon(color="orange", icon="subway", prefix="fa")

def add_base_layers(m):
    folium.GeoJson(
        sel_ll,
        name="행정동 경계",
        style_function=lambda x: {"fillOpacity": 0.03, "color": "#444", "weight": 3},
        tooltip=folium.GeoJsonTooltip(fields=["region_nm"], aliases=["행정동"]),
    ).add_to(m)

    # 정류장 클러스터(마커 많을 수 있으니)
    mc_bus = MarkerCluster(name="버스정류장").add_to(m)
    for _, r in bus_ll.iterrows():
        folium.Marker(
            location=[r.geometry.y, r.geometry.x],
            tooltip=f"버스정류장 | {r.get('정류소명','')}",
            icon=bus_icon,
        ).add_to(mc_bus)

    mc_sub = MarkerCluster(name="지하철역").add_to(m)
    for _, r in sub_ll.iterrows():
        folium.Marker(
            location=[r.geometry.y, r.geometry.x],
            tooltip="지하철역",
            icon=sub_icon,
        ).add_to(mc_sub)

def add_top_grid_and_routes(m, top_ll, routes, poly_color, top_name):
    if top_ll is not None and len(top_ll) > 0:
        r = top_ll.iloc[0]
        pop = float(r.get("pop", 0))
        gid = r.get("gid", "")
        tip = f"{top_name} | gid={gid} | pop={pop:,.0f}"

        folium.GeoJson(
            {"type": "Feature", "properties": {}, "geometry": mapping(r.geometry)},
            name=f"{top_name} TOP 격자",
            style_function=lambda x: {"fillOpacity": 0.50, "fillColor": poly_color, "color": poly_color, "weight": 3},
            tooltip=tip,
        ).add_to(m)

        c = r.geometry.centroid
        folium.Marker(
            location=[c.y, c.x],
            tooltip=tip,
            icon=folium.DivIcon(html=number_badge_html(1, poly_color)),
        ).add_to(m)

    if routes is not None and len(routes) > 0:
        fg = folium.FeatureGroup(name="TOP 격자→버스 5분 최단경로", show=True)
        for ls in routes:
            if ls is None or ls.is_empty:
                continue
            folium.PolyLine(
                [(y, x) for x, y in ls.coords],
                weight=4, opacity=0.85, color="#111111",
            ).add_to(fg)
        fg.add_to(m)

# ---- (좌) 버퍼 지도 ----
m_buf = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")
add_base_layers(m_buf)

if cover_buf_ll is not None and (not cover_buf_ll.is_empty):
    folium.GeoJson(
        mapping(cover_buf_ll),
        name="커버(버퍼)",
        style_function=lambda x: {"fillOpacity": 0.22, "fillColor": "#28a745", "color": "#28a745", "weight": 1},
    ).add_to(m_buf)

if uncov_buf_ll is not None and (not uncov_buf_ll.is_empty):
    folium.GeoJson(
        mapping(uncov_buf_ll),
        name="비커버(버퍼)",
        style_function=lambda x: {"fillOpacity": 0.32, "fillColor": "#cc0000", "color": "#cc0000", "weight": 2},
    ).add_to(m_buf)

add_top_grid_and_routes(m_buf, top_buf_ll, routes_top_buf, poly_color="#ff6600", top_name="버퍼 비커버 최대인구")
folium.LayerControl(collapsed=False).add_to(m_buf)
m_buf.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

# ---- (우) 네트워크 지도 ----
m_iso = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")
add_base_layers(m_iso)

if cover_iso_ll is not None and (not cover_iso_ll.is_empty):
    folium.GeoJson(
        mapping(cover_iso_ll),
        name="커버(Isochrone)",
        style_function=lambda x: {"fillOpacity": 0.18, "fillColor": "#0066ff", "color": "#0066ff", "weight": 1},
    ).add_to(m_iso)

if uncov_iso_ll is not None and (not uncov_iso_ll.is_empty):
    folium.GeoJson(
        mapping(uncov_iso_ll),
        name="비커버(Isochrone)",
        style_function=lambda x: {"fillOpacity": 0.28, "fillColor": "#7a00cc", "color": "#7a00cc", "weight": 2},
    ).add_to(m_iso)

add_top_grid_and_routes(m_iso, top_iso_ll, routes_top_iso, poly_color="#e91e63", top_name="네트워크 비커버 최대인구")
folium.LayerControl(collapsed=False).add_to(m_iso)
m_iso.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

# =========================================================
# 11) 화면 배치
# =========================================================
st.markdown("---")
col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.subheader("직선 버퍼 기반 분석")
    st_folium(m_buf, width=None, height=MAP_HEIGHT_PX, key="map_buf", returned_objects=[])

with col_r:
    st.subheader("네트워크(Isochrone) 기반 분석")
    st_folium(m_iso, width=None, height=MAP_HEIGHT_PX, key="map_iso", returned_objects=[])

with st.expander("분석 방법론 비교"):
    st.markdown(
        """
| 항목 | 직선 버퍼 | 네트워크 기반 (Isochrone) |
|------|-----------|--------------------------|
| 방식 | 정류장 중심 원형 버퍼 (300 m / 500 m) | OSMnx 도보 네트워크 + ego_graph(거리=length) + 도로폭(25m) 버퍼 |
| 장점 | 계산 빠름, 직관적 | 실제 보행 경로 반영 + 경로 복원 가능 |
| 단점 | 장애물/단절 미반영 | OSM 다운로드/계산 필요 |
| 비커버 판단 | 원 바깥 = 비커버 | 도보 네트워크로 도달 불가 = 비커버 |
| TOP 격자 경로 | TOP 격자 중심→버스정류장 5분 최단경로 표시 | TOP 격자 중심→버스정류장 5분 최단경로 표시 |
        """
    )
