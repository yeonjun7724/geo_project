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
# 0) PATHS / CONST
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

GRID_SHP = os.path.join(DATA_DIR, "nlsp_021001001.shp")          # 전수 격자 SHP
UNCOVERED_GPKG = os.path.join(DATA_DIR, "demo_uncovered.gpkg")  # 비커버 폴리곤(선택)

GRID_ID_COL = "gid"
GRID_POP_COL = "val"

TARGET_CRS = 5179
MAP_CRS = 4326


# =========================================================
# 1) Streamlit Page
# =========================================================
st.set_page_config(page_title="5강 | Streamlit + Pydeck + OSMnx", layout="wide")

st.title("🚲 5강 | Streamlit 대시보드: 격자 선택 → KPI 즉석 계산 → 좌(Pydeck) / 우(5분 네트워크)")
st.caption("우측은 선택 격자 중심점에서 시작해 OSMnx+NetworkX로 5분 내 도달 가능한 네트워크 라인을 즉석 계산해 표시한다.")


# =========================================================
# 2) Load Grid (전수 격자)
# =========================================================
if not os.path.exists(GRID_SHP):
    st.error(f"GRID_SHP not found: {GRID_SHP}\n(data 폴더에 shp/shx/dbf/prj 세트를 모두 넣어주세요.)")
    st.stop()

with st.spinner("전수 격자 로딩 중..."):
    gdf_grid = gpd.read_file(GRID_SHP)

if gdf_grid.crs is None:
    st.error("GRID_SHP CRS is None (.prj 확인)")
    st.stop()

gdf_grid = gdf_grid.to_crs(TARGET_CRS)

if GRID_ID_COL not in gdf_grid.columns:
    st.error(f"'{GRID_ID_COL}' 컬럼이 없습니다. 현재 컬럼: {list(gdf_grid.columns)}")
    st.stop()

gdf_grid[GRID_ID_COL] = gdf_grid[GRID_ID_COL].astype(str)

if GRID_POP_COL in gdf_grid.columns:
    gdf_grid["pop"] = pd.to_numeric(gdf_grid[GRID_POP_COL], errors="coerce").fillna(0).astype(float)
elif "pop" in gdf_grid.columns:
    gdf_grid["pop"] = pd.to_numeric(gdf_grid["pop"], errors="coerce").fillna(0).astype(float)
else:
    gdf_grid["pop"] = 0.0

gdf_grid["geometry"] = gdf_grid.geometry.buffer(0)
gdf_grid = gdf_grid[[GRID_ID_COL, "pop", "geometry"]].copy()


# =========================================================
# 3) Load Uncovered (선택) + attach is_uncovered
# =========================================================
with st.spinner("비커버 폴리곤 결합 중..."):
    if os.path.exists(UNCOVERED_GPKG):
        gdf_unc = gpd.read_file(UNCOVERED_GPKG)
        if gdf_unc.crs is None:
            st.error("UNCOVERED_GPKG CRS is None")
            st.stop()
        gdf_unc = gdf_unc.to_crs(TARGET_CRS)
        gdf_unc["geometry"] = gdf_unc.geometry.buffer(0)

        if len(gdf_unc) == 0:
            gdf_grid["is_uncovered"] = False
        else:
            unc_union = gdf_unc.geometry.union_all()
            gdf_grid["is_uncovered"] = gdf_grid.geometry.intersects(unc_union)
    else:
        gdf_grid["is_uncovered"] = False


# =========================================================
# 4) Sidebar (gid / KPI 반경 / 네트워크 파라미터)
# =========================================================
all_gids = gdf_grid[GRID_ID_COL].tolist()
if len(all_gids) == 0:
    st.error("전수 격자가 비어 있습니다. shp 내용을 확인하세요.")
    st.stop()

st.sidebar.header("설정")

sel_gid = st.sidebar.selectbox("전수 격자 gid 선택", options=all_gids, index=0)

RADIUS_M = st.sidebar.slider("KPI 반경(m)", 300, 3000, 1250, 50)

speed_mps = st.sidebar.slider("보행 속도(m/s)", 0.8, 2.0, 1.4, 0.1)
cutoff_min = st.sidebar.slider("네트워크 컷오프(분)", 1, 15, 5, 1)
cutoff_sec = int(cutoff_min * 60)

graph_dist_m = st.sidebar.slider("그래프 다운로드 반경(m)", 1000, 8000, 3500, 250)

st.sidebar.caption("우측 네트워크는 travel_time = length(m) / speed(m/s) 로 계산한다.")


# =========================================================
# 5) KPI 즉석 계산 (선택 gid 중심점 반경)
# =========================================================
row = gdf_grid.loc[gdf_grid[GRID_ID_COL] == str(sel_gid)]
if len(row) == 0:
    st.error("선택 gid를 grid에서 찾지 못했습니다. gid 컬럼/형식을 확인하세요.")
    st.stop()

sel_poly = row.geometry.iloc[0]
sel_center_5179 = sel_poly.centroid
circle_5179 = sel_center_5179.buffer(float(RADIUS_M))

mask_in = gdf_grid.geometry.intersects(circle_5179)
gdf_in = gdf_grid.loc[mask_in, [GRID_ID_COL, "pop", "is_uncovered", "geometry"]].copy()

total_pop = float(gdf_in["pop"].sum())
uncovered_pop = float(gdf_in.loc[gdf_in["is_uncovered"] == True, "pop"].sum())
covered_pop = total_pop - uncovered_pop
uncovered_rate = (uncovered_pop / total_pop) if total_pop > 0 else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("선택 gid", str(sel_gid))
c2.metric("반경 내 격자 수", f"{len(gdf_in):,}")
c3.metric("총 인구", f"{total_pop:,.0f}")
c4.metric("비커버 인구", f"{uncovered_pop:,.0f}")
c5.metric("비커버 비율", f"{uncovered_rate*100:.2f}%")


# =========================================================
# 6) 좌/우 레이아웃
# =========================================================
left, right = st.columns([1, 1])


# =========================================================
# 6-1) LEFT: Pydeck 3D
# =========================================================
with left:
    st.subheader("좌측: Pydeck 3D 격자 + KPI 반경")

    gdf_ll = gdf_in.to_crs(MAP_CRS).copy()

    pop = gdf_ll["pop"].clip(lower=0).astype(float)
    cap_val = float(pop.quantile(0.995)) if len(pop) > 0 else 0.0
    pop_capped = np.minimum(pop, cap_val) if cap_val > 0 else pop

    gdf_ll["elev"] = (np.power(pop_capped, 1.80) * 0.02).astype(float)

    # Pydeck PolygonLayer 입력 records 만들기 (스크립트 방식: 루프 그대로)
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

    circle_ll = gpd.GeoSeries([circle_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
    circle_coords = list(circle_ll.exterior.coords)

    sel_center_ll = gpd.GeoSeries([sel_center_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]

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

    st.pydeck_chart(deck, width="stretch")


# =========================================================
# 6-2) RIGHT: OSMnx + NetworkX (즉석 5분 네트워크)
# =========================================================
with right:
    st.subheader("우측: OSMnx+NetworkX 즉석 계산 5분 네트워크")

    # 선택 중심점(4326)
    sel_center_ll = gpd.GeoSeries([sel_center_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
    lon, lat = float(sel_center_ll.x), float(sel_center_ll.y)

    # 그래프 다운로드
    with st.spinner(f"OSM 그래프 다운로드... (dist={graph_dist_m}m)"):
        ox.settings.log_console = False
        G = ox.graph_from_point(
            (lat, lon),
            dist=int(graph_dist_m),
            network_type="walk",
            simplify=True,
        )

        # OSMnx 버전 호환: add_edge_lengths 위치가 다를 수 있음
        try:
            G = ox.distance.add_edge_lengths(G)
        except Exception:
            try:
                G = ox.add_edge_lengths(G)
            except Exception:
                pass

        # ✅ sklearn 옵션 의존성(nearest_nodes) 회피: 그래프를 투영해서 처리
        G = ox.project_graph(G)

    # travel_time(초) 부여
    sp = float(speed_mps)
    for u, v, k, data in G.edges(keys=True, data=True):
        length_m = float(data.get("length", 0.0))
        data["travel_time"] = (length_m / sp) if sp > 0 else np.inf

    # 중심점을 그래프 CRS로 변환 후 nearest_nodes
    graph_crs = G.graph.get("crs", None)
    if graph_crs is None:
        st.error("그래프 CRS를 확인하지 못했습니다. (project_graph 실패 가능) dist/환경을 확인하세요.")
        st.stop()

    pt_graph = gpd.GeoSeries([sel_center_5179], crs=TARGET_CRS).to_crs(graph_crs).iloc[0]
    Xp, Yp = float(pt_graph.x), float(pt_graph.y)

    try:
        source_node = ox.distance.nearest_nodes(G, X=Xp, Y=Yp)
    except Exception as e:
        st.error(f"nearest_nodes 실패: {e}")
        st.stop()

    # reachable nodes (Dijkstra cutoff)
    with st.spinner(f"{cutoff_min}분 네트워크 계산 중... (cutoff={cutoff_sec}s)"):
        lengths = nx.single_source_dijkstra_path_length(
            G, int(source_node), cutoff=float(cutoff_sec), weight="travel_time"
        )
        reachable_nodes = set(lengths.keys())
        SG = G.subgraph(reachable_nodes).copy()

        gdf_edges = ox.graph_to_gdfs(SG, nodes=False, edges=True, fill_edge_geometry=True)
        if gdf_edges.crs is None:
            gdf_edges = gdf_edges.set_crs(graph_crs)

        # Folium은 4326이므로 최종 변환
        gdf_edges = gdf_edges.to_crs(MAP_CRS)

        if "length" in gdf_edges.columns:
            gdf_edges["length_m"] = gdf_edges["length"].astype(float)
        if "travel_time" in gdf_edges.columns:
            gdf_edges["time_s"] = gdf_edges["travel_time"].astype(float)

        gdf_edges = gdf_edges.reset_index(drop=True)

    # 네트워크 KPI
    n_edges = int(len(gdf_edges))
    total_len_km = float(gdf_edges["length_m"].sum() / 1000.0) if "length_m" in gdf_edges.columns else np.nan

    c6, c7 = st.columns(2)
    c6.metric("네트워크 edge 수", f"{n_edges:,}")
    c7.metric("네트워크 총 길이(km)", f"{total_len_km:,.2f}" if not np.isnan(total_len_km) else "-")

    # Folium 지도
    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="cartodbpositron")

    folium.Marker(
        location=[lat, lon],
        tooltip=f"gid={sel_gid} (nearest node: {source_node})",
        icon=folium.Icon(color="red", icon="play", prefix="fa"),
    ).add_to(m)

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
        st.info("도달 가능한 네트워크가 비어 있습니다. dist/속도/cutoff를 조절하세요.")

    circle_ll = gpd.GeoSeries([circle_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
    folium.GeoJson(
        {"type": "Feature", "properties": {}, "geometry": circle_ll.__geo_interface__},
        name="kpi_radius",
        style_function=lambda x: {"color": "#111111", "weight": 2, "opacity": 0.8},
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, width=None, height=650)


# =========================================================
# 7) Debug Panel
# =========================================================
with st.expander("데이터/그래프 진단"):
    st.write("GRID_SHP:", GRID_SHP, "(exists:", os.path.exists(GRID_SHP), ")")
    st.write("UNCOVERED_GPKG:", UNCOVERED_GPKG, "(exists:", os.path.exists(UNCOVERED_GPKG), ")")
    st.write("grid CRS:", str(gdf_grid.crs))
    st.write("grid columns:", list(gdf_grid.columns))
    st.write("selected gid:", sel_gid)
    st.write("RADIUS_M:", RADIUS_M)
    st.write("graph_dist_m:", graph_dist_m, "| speed_mps:", speed_mps, "| cutoff_min:", cutoff_min)
    try:
        st.write("OSM graph nodes:", len(G.nodes), "edges:", len(G.edges))
    except Exception:
        st.write("OSM graph: (우측 실행 전)")
