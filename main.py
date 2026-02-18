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
from shapely.geometry import mapping, LineString

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
WALK_5MIN_M    = 5 * 60 * 1.4   # 420m (보행 1.4m/s × 5분)

MAP_HEIGHT_PX = 650

# =========================================================
# 1) 페이지 설정
# =========================================================

st.set_page_config(
    page_title="대중교통 커버리지 비교: 직선 버퍼 vs 네트워크",
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

st.title("대중교통 커버리지 분석: 직선 버퍼 vs 네트워크 기반")
st.caption("버스 300 m / 지하철 500 m 기준 · 5분 도보 경로 표시 — 남현동 · 청림동")

# =========================================================
# 2) 데이터 로드 + 분석 (캐싱)
# =========================================================

def _to_ll(geom):
    """EPSG:5179 Shapely geometry → EPSG:4326"""
    return gpd.GeoSeries([geom], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]


@st.cache_resource(show_spinner=False)
def compute_all():
    # ── 행정동 ──
    gdf_admin = gpd.read_file(ADMIN_SHP)
    gdf_admin["region_id"] = gdf_admin["ADM_CD"].astype(str).str.strip()
    gdf_admin["region_nm"] = gdf_admin["ADM_NM"].astype(str).str.strip()
    gdf_admin = gdf_admin.to_crs(TARGET_CRS)

    gdf_sel = gdf_admin[gdf_admin["region_id"].isin(TARGET_IDS)].copy()
    if len(gdf_sel) == 0:
        raise ValueError(f"행정동 {list(TARGET_IDS)} 을 찾을 수 없습니다.")
    sel_union = unary_union(gdf_sel.geometry)

    # ── 버스정류장 ──
    bus_raw = pd.read_excel(BUS_XLSX)
    bus_raw["X좌표"] = pd.to_numeric(bus_raw["X좌표"], errors="coerce")
    bus_raw["Y좌표"] = pd.to_numeric(bus_raw["Y좌표"], errors="coerce")
    bus_raw = bus_raw.dropna(subset=["X좌표", "Y좌표"])
    gdf_bus = gpd.GeoDataFrame(
        bus_raw, geometry=gpd.points_from_xy(bus_raw["X좌표"], bus_raw["Y좌표"]), crs=MAP_CRS
    ).to_crs(TARGET_CRS)
    gdf_bus_sel = gdf_bus[gdf_bus.geometry.within(sel_union)].copy()
    gdf_bus_sel["stop_type"] = "bus"

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
    gdf_sub_sel = gdf_sub[gdf_sub.geometry.within(sel_union)].copy()
    gdf_sub_sel["stop_type"] = "subway"

    # ── 인구 격자 ──
    gdf_grid = gpd.read_file(GRID_SHP).to_crs(TARGET_CRS)
    gdf_grid["gid"] = gdf_grid["gid"].astype(str)
    gdf_grid["pop"] = pd.to_numeric(gdf_grid.get("val", 0), errors="coerce").fillna(0.0)
    gdf_grid_sel = gpd.clip(
        gdf_grid[gdf_grid.geometry.intersects(sel_union)], gdf_sel
    )[["gid", "pop", "geometry"]].copy()

    # 격자에 행정동 region_id 부여 (centroid 기준 sjoin)
    grid_with_dong = gpd.sjoin(
        gdf_grid_sel, gdf_sel[["region_id", "region_nm", "geometry"]],
        how="left", predicate="intersects",
    ).drop(columns="index_right")
    # 중복 제거 (격자가 경계에 걸칠 수 있음)
    grid_with_dong = grid_with_dong.drop_duplicates(subset="gid", keep="first")

    # ── (A) 직선 버퍼 커버리지 ──
    bufs = []
    if len(gdf_bus_sel):
        bufs.append(unary_union(gdf_bus_sel.geometry.buffer(BUS_BUFFER_M)))
    if len(gdf_sub_sel):
        bufs.append(unary_union(gdf_sub_sel.geometry.buffer(SUB_BUFFER_M)))
    cover_buf = unary_union(bufs) if bufs else None
    uncov_buf = sel_union.difference(cover_buf) if cover_buf else sel_union

    # ── (B) Isochrone(네트워크) 커버리지 + 5분 도보 경로 ──
    poly_graph_ll = (
        gpd.GeoSeries([sel_union.buffer(GRAPH_BUFFER_M)], crs=TARGET_CRS)
        .to_crs(MAP_CRS).iloc[0]
    )
    ox.settings.log_console = False
    G = ox.graph_from_polygon(poly_graph_ll, network_type="walk", simplify=True)

    gdf_stops = gpd.GeoDataFrame(
        pd.concat([gdf_bus_sel, gdf_sub_sel], ignore_index=True),
        geometry="geometry", crs=TARGET_CRS,
    )
    gdf_stops_ll = gdf_stops.to_crs(MAP_CRS).copy()
    gdf_stops_ll["v_node"] = ox.distance.nearest_nodes(
        G, X=gdf_stops_ll.geometry.x.values, Y=gdf_stops_ll.geometry.y.values
    )

    iso_polys = []
    reachable_edge_seen = set()
    reachable_edge_rows = []

    for _, r in gdf_stops_ll.iterrows():
        v = int(r["v_node"])
        iso_cutoff = BUS_BUFFER_M if r["stop_type"] == "bus" else SUB_BUFFER_M
        route_cutoff = WALK_5MIN_M   # 5분 경로용

        # isochrone 커버리지 (300m/500m)
        try:
            Gsub_iso = nx.ego_graph(G, v, radius=iso_cutoff, distance="length", undirected=True)
        except Exception:
            Gsub_iso = None

        if Gsub_iso and Gsub_iso.number_of_edges() > 0:
            _, gdf_edges = ox.graph_to_gdfs(Gsub_iso, nodes=True, edges=True, fill_edge_geometry=True)
            poly_m = unary_union(gdf_edges.to_crs(TARGET_CRS).geometry.buffer(EDGE_BUFFER_M))
            if poly_m and not poly_m.is_empty:
                iso_polys.append(poly_m)

        # 5분 도보 경로 (420m) — 지도 표시용
        try:
            Gsub_5min = nx.ego_graph(G, v, radius=route_cutoff, distance="length", undirected=True)
        except Exception:
            continue

        if Gsub_5min.number_of_edges() == 0:
            continue

        _, gdf_e5 = ox.graph_to_gdfs(Gsub_5min, nodes=True, edges=True, fill_edge_geometry=True)
        for idx, ed in gdf_e5.iterrows():
            u, w, k = idx
            edge_key = (min(u, w), max(u, w), k)
            if edge_key in reachable_edge_seen:
                continue
            reachable_edge_seen.add(edge_key)
            geom = ed.geometry
            if geom and not geom.is_empty and float(ed.get("length", 0)) > 0:
                reachable_edge_rows.append({"geometry": geom})

    cover_iso = unary_union(iso_polys) if iso_polys else None
    uncov_iso = sel_union.difference(cover_iso) if cover_iso else sel_union

    gdf_edges_reachable = gpd.GeoDataFrame(
        reachable_edge_rows, geometry="geometry", crs=MAP_CRS
    ) if reachable_edge_rows else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=MAP_CRS)

    # ── KPI 계산 ──
    admin_area = sel_union.area
    buf_area = uncov_buf.area if uncov_buf and not uncov_buf.is_empty else 0
    iso_area = uncov_iso.area if uncov_iso and not uncov_iso.is_empty else 0

    centroids = grid_with_dong.geometry.centroid
    buf_mask = centroids.within(uncov_buf) if (uncov_buf and not uncov_buf.is_empty) else pd.Series(False, index=grid_with_dong.index)
    iso_mask = centroids.within(uncov_iso) if (uncov_iso and not uncov_iso.is_empty) else pd.Series(False, index=grid_with_dong.index)

    buf_pop = float(grid_with_dong.loc[buf_mask, "pop"].sum())
    iso_pop = float(grid_with_dong.loc[iso_mask, "pop"].sum())
    total_pop = float(grid_with_dong["pop"].sum())

    false_covered = (~buf_mask) & iso_mask
    additional_pop = float(grid_with_dong.loc[false_covered, "pop"].sum())

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

    # ── 행정동별 비커버 최대인구 격자 1개씩 ──
    top_grid_buf_rows = []
    top_grid_iso_rows = []

    for rid, rnm in TARGET_IDS.items():
        dong_geom = gdf_sel.loc[gdf_sel["region_id"] == rid, "geometry"]
        if len(dong_geom) == 0:
            continue
        dong_poly = unary_union(dong_geom)

        dong_grids = grid_with_dong[grid_with_dong["region_id"] == rid].copy()
        if len(dong_grids) == 0:
            continue

        dong_cents = dong_grids.geometry.centroid

        # 버퍼 비커버 내 최대 인구 격자
        uncov_buf_dong = dong_poly.difference(cover_buf) if cover_buf else dong_poly
        if uncov_buf_dong and not uncov_buf_dong.is_empty:
            mask_b = dong_cents.within(uncov_buf_dong)
            cands = dong_grids[mask_b]
            if len(cands) > 0:
                top = cands.loc[cands["pop"].idxmax()].copy()
                top["region_nm"] = rnm
                top_grid_buf_rows.append(top)

        # isochrone 비커버 내 최대 인구 격자
        uncov_iso_dong = dong_poly.difference(cover_iso) if cover_iso else dong_poly
        if uncov_iso_dong and not uncov_iso_dong.is_empty:
            mask_i = dong_cents.within(uncov_iso_dong)
            cands = dong_grids[mask_i]
            if len(cands) > 0:
                top = cands.loc[cands["pop"].idxmax()].copy()
                top["region_nm"] = rnm
                top_grid_iso_rows.append(top)

    gdf_top_buf = gpd.GeoDataFrame(top_grid_buf_rows, crs=TARGET_CRS) if top_grid_buf_rows else None
    gdf_top_iso = gpd.GeoDataFrame(top_grid_iso_rows, crs=TARGET_CRS) if top_grid_iso_rows else None

    # ── 표시용 4326 변환 ──
    sel_ll = gdf_sel.to_crs(MAP_CRS)
    bounds = sel_ll.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    cover_buf_ll = _to_ll(cover_buf.intersection(sel_union)) if cover_buf else None
    uncov_buf_ll = _to_ll(uncov_buf) if (uncov_buf and not uncov_buf.is_empty) else None
    cover_iso_cl = cover_iso.intersection(sel_union) if cover_iso else None
    cover_iso_ll = _to_ll(cover_iso_cl.simplify(5)) if cover_iso_cl else None
    uncov_iso_ll = _to_ll(uncov_iso.simplify(5)) if (uncov_iso and not uncov_iso.is_empty) else None

    bus_ll = gdf_bus_sel.to_crs(MAP_CRS)
    sub_ll = gdf_sub_sel.to_crs(MAP_CRS)
    top_buf_ll = gdf_top_buf.to_crs(MAP_CRS) if gdf_top_buf is not None else None
    top_iso_ll = gdf_top_iso.to_crs(MAP_CRS) if gdf_top_iso is not None else None

    return {
        "kpi": kpi,
        "sel_ll": sel_ll,
        "bounds": bounds,
        "center": center,
        "cover_buf_ll": cover_buf_ll,
        "uncov_buf_ll": uncov_buf_ll,
        "cover_iso_ll": cover_iso_ll,
        "uncov_iso_ll": uncov_iso_ll,
        "bus_ll": bus_ll,
        "sub_ll": sub_ll,
        "edges_ll": gdf_edges_reachable,
        "top_buf_ll": top_buf_ll,
        "top_iso_ll": top_iso_ll,
    }

# =========================================================
# 3) 지도 생성 함수
# =========================================================

def _add_common(m, d):
    """행정동 경계 + 정류장 마커 + 5분 도보 경로"""
    # 행정동 경계
    folium.GeoJson(
        d["sel_ll"], name="행정동 경계",
        style_function=lambda x: {"fillOpacity": 0.03, "color": "#444", "weight": 3},
        tooltip=folium.GeoJsonTooltip(fields=["region_nm"], aliases=["행정동"]),
    ).add_to(m)

    # 5분 도보 경로
    fg = folium.FeatureGroup(name="5분 도보 도달 경로", show=True)
    for _, ed in d["edges_ll"].iterrows():
        geom = ed.geometry
        if geom is None or geom.is_empty:
            continue
        lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        for ls in lines:
            folium.PolyLine(
                [(y, x) for x, y in ls.coords],
                weight=2, opacity=0.35, color="#00bcd4",
            ).add_to(fg)
    fg.add_to(m)

    # 정류장 마커
    for _, r in d["bus_ll"].iterrows():
        folium.CircleMarker(
            location=[r.geometry.y, r.geometry.x],
            radius=4, color="#0066ff", fill=True, fill_opacity=0.8,
            tooltip=f"버스정류장 | {r.get('정류소명', '')}",
        ).add_to(m)
    for _, r in d["sub_ll"].iterrows():
        folium.CircleMarker(
            location=[r.geometry.y, r.geometry.x],
            radius=6, color="#ff6600", fill=True, fill_opacity=0.9,
            tooltip="지하철역",
        ).add_to(m)


def _add_top_grid(m, gdf_top_ll, label_prefix, color):
    """행정동별 비커버 최대인구 격자 표시"""
    if gdf_top_ll is None or len(gdf_top_ll) == 0:
        return
    for _, r in gdf_top_ll.iterrows():
        nm = r.get("region_nm", "")
        pop = float(r.get("pop", 0))
        gid = r.get("gid", "")
        tip = f"{label_prefix} | {nm} | gid={gid} | pop={pop:,.0f}"

        # 격자 폴리곤 강조
        folium.GeoJson(
            {"type": "Feature", "properties": {}, "geometry": mapping(r.geometry)},
            style_function=lambda x, c=color: {
                "fillOpacity": 0.55, "fillColor": c, "color": c, "weight": 3,
            },
            tooltip=tip,
        ).add_to(m)

        # 중심에 라벨 마커
        c = r.geometry.centroid
        folium.Marker(
            location=[c.y, c.x],
            tooltip=tip,
            icon=folium.DivIcon(html=f"""
                <div style="
                    display:flex; align-items:center; justify-content:center;
                    padding:2px 6px; font-size:11px; font-weight:800; color:#fff;
                    background:{c if isinstance(c, str) else color}; border:2px solid #fff;
                    border-radius:4px; box-shadow:0 1px 4px rgba(0,0,0,0.4);
                    white-space:nowrap;
                ">{nm} TOP1<br>pop {pop:,.0f}</div>
            """),
        ).add_to(m)


def create_buffer_map(d):
    m = folium.Map(location=d["center"], zoom_start=14, tiles="cartodbpositron")
    _add_common(m, d)

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

    _add_top_grid(m, d["top_buf_ll"], "버퍼 비커버 최대인구", "#ff6600")

    folium.LayerControl(collapsed=False).add_to(m)
    b = d["bounds"]
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    return m


def create_isochrone_map(d):
    m = folium.Map(location=d["center"], zoom_start=14, tiles="cartodbpositron")
    _add_common(m, d)

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

    _add_top_grid(m, d["top_iso_ll"], "네트워크 비커버 최대인구", "#e91e63")

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
st.subheader("KPI 비교 (남현동 + 청림동)")

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
| **단점** | 건물·하천·도로 등 장애물 미반영 | OSM 네트워크 다운로드 필요, 계산 시간 |
| **비커버 판단** | 원 바깥 = 비커버 | 도보로 도달 불가 = 비커버 |
| **5분 경로** | 동일한 도보 네트워크 경로 표시 (420 m) | 동일한 도보 네트워크 경로 표시 (420 m) |
        """
    )
