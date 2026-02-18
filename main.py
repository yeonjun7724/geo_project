# =========================================================
# 0) 라이브러리
# =========================================================
import os, warnings  # os: 경로 처리 / warnings: 경고 처리
warnings.filterwarnings("ignore")  # 경고 메시지 숨김

import numpy as np  # 수치 계산
import pandas as pd  # 테이블 데이터 처리
import geopandas as gpd  # 공간 데이터 처리
import folium  # 지도 시각화
import streamlit as st  # Streamlit UI
from streamlit_folium import st_folium  # Streamlit에서 Folium 렌더링
import osmnx as ox  # OSM 네트워크 다운로드/처리
import networkx as nx  # 최단경로/그래프 알고리즘
from shapely.ops import unary_union  # 지오메트리 합치기
from shapely.geometry import mapping  # GeoJSON 변환

# =========================================================
# 1) 상수 / 경로
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 절대 경로 기준 폴더
DATA_DIR = os.path.join(BASE_DIR, "data")  # 데이터 폴더

ADMIN_SHP  = os.path.join(DATA_DIR, "BND_ADM_DONG_PG.gpkg")  # 행정동 경계
BUS_XLSX   = os.path.join(DATA_DIR, "서울시버스정류소위치정보(20260108).xlsx")  # 버스정류장
SUBWAY_CSV = os.path.join(DATA_DIR, "서울교통공사_1_8호선 역사 좌표(위경도) 정보_20250814.csv")  # 지하철역
GRID_SHP   = os.path.join(DATA_DIR, "nlsp_021001001.shp")  # 인구격자

TARGET_IDS = {"11210630": "남현동", "11210540": "청림동"}  # 분석 대상 행정동

TARGET_CRS = 5179  # 분석/거리 계산 좌표계(미터)
MAP_CRS    = 4326  # 지도 표시 좌표계(위경도)

BUS_BUFFER_M   = 300.0  # 버스 커버 기준(m)
SUB_BUFFER_M   = 500.0  # 지하철 커버 기준(m)
GRAPH_BUFFER_M = 1500.0  # 네트워크 다운로드 영역 여유 버퍼(m)
EDGE_BUFFER_M  = 25.0  # isochrone 도로폭 보정 버퍼(m)

MAP_HEIGHT_PX = 650  # 지도 높이(px)

# ---- 추가: “최종 격자 → 모든 정류장/역” 경로 그리기 제어 ----
DRAW_ALL_ROUTES = True  # True면 “최종 격자에서 각 정류장/역까지” 경로를 지도에 그림
MAX_DRAW_ROUTES = 300  # 너무 많아지면 느려지므로, 그릴 최대 경로 개수(안전장치)
ROUTE_WEIGHT = 4  # 경로 라인 두께
ROUTE_OPACITY = 0.85  # 경로 라인 투명도
ROUTE_COLOR = "#111111"  # 경로 라인 색(검정 계열)

# =========================================================
# 2) 페이지 설정
# =========================================================
st.set_page_config(page_title="대중교통 커버리지 비교", layout="wide")  # 페이지 설정

st.markdown(  # 스타일 조정(CSS)
    """
    <style>
      .block-container { padding: 1.2rem 1.0rem 1.6rem 1.0rem; max-width: none; }
      h1, h2, h3 { letter-spacing: -0.02em; }
      .stApp h1, div[data-testid="stMarkdownContainer"] h1 { text-align: center; width: 100%; }
      div[data-testid="stMarkdownContainer"] h1 { margin-top: 0.2rem; margin-bottom: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,  # HTML/CSS 허용
)

st.title("대중교통 커버리지 분석: 직선 버퍼 vs 네트워크 기반")  # 제목
st.caption("버스 300 m / 지하철 500 m 기준 · ‘최종 TOP 격자’에서 각 버스정류장/지하철역까지 최단경로(라인) 표시")  # 설명

# =========================================================
# 3) 드롭다운(행정동 선택)
# =========================================================
st.markdown("---")  # 구분선
rid = st.selectbox(  # 선택 박스
    "행정동 선택",  # 라벨
    options=list(TARGET_IDS.keys()),  # 옵션(행정동 코드)
    format_func=lambda x: f"{TARGET_IDS.get(x, x)} ({x})",  # 표시 문자열
    index=0,  # 기본 선택
)
st.caption(f"선택 행정동: {TARGET_IDS.get(rid)}")  # 선택 결과 표시

# =========================================================
# 4) 데이터 로드 + 분석 (스크립트)
# =========================================================
with st.spinner("데이터 로드/분석 중... (OSM 네트워크 다운로드 포함)"):  # 스피너 표시
    # ─────────────────────────────────────────────────────
    # (1) 행정동 로드/선택
    # ─────────────────────────────────────────────────────
    gdf_admin = gpd.read_file(ADMIN_SHP)  # 행정동 레이어 로드
    gdf_admin["region_id"] = gdf_admin["ADM_CD"].astype(str).str.strip()  # 코드 표준화
    gdf_admin["region_nm"] = gdf_admin["ADM_NM"].astype(str).str.strip()  # 이름 표준화
    gdf_admin = gdf_admin.to_crs(TARGET_CRS)  # 분석 좌표계로 변환

    gdf_sel = gdf_admin[gdf_admin["region_id"] == rid].copy()  # 선택 행정동만 필터
    if len(gdf_sel) == 0:  # 선택 실패 시
        st.error("선택한 행정동을 찾을 수 없습니다.")  # 오류 출력
        st.stop()  # 중단

    region_nm = gdf_sel["region_nm"].iloc[0]  # 행정동명
    sel_union = unary_union(gdf_sel.geometry)  # 행정동 폴리곤 유니온(멀티면 합침)

    sel_ll = gdf_sel.to_crs(MAP_CRS)  # 표시용(4326) 변환
    bounds = sel_ll.total_bounds  # 경계좌표 [minx, miny, maxx, maxy]
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]  # 중심점 [lat, lon]

    # ─────────────────────────────────────────────────────
    # (2) 버스정류장 로드/행정동 내부 선택
    # ─────────────────────────────────────────────────────
    bus_raw = pd.read_excel(BUS_XLSX)  # 엑셀 로드
    bus_raw["X좌표"] = pd.to_numeric(bus_raw["X좌표"], errors="coerce")  # X 숫자화
    bus_raw["Y좌표"] = pd.to_numeric(bus_raw["Y좌표"], errors="coerce")  # Y 숫자화
    bus_raw = bus_raw.dropna(subset=["X좌표", "Y좌표"])  # 결측 제거

    gdf_bus = gpd.GeoDataFrame(  # 버스 GeoDataFrame
        bus_raw,  # 원본
        geometry=gpd.points_from_xy(bus_raw["X좌표"], bus_raw["Y좌표"]),  # Point 생성
        crs=MAP_CRS,  # 입력 좌표계(위경도)
    ).to_crs(TARGET_CRS)  # 분석 좌표계로 변환

    gdf_bus_sel = gdf_bus[gdf_bus.geometry.within(sel_union)].copy()  # 행정동 내부 버스만

    # ─────────────────────────────────────────────────────
    # (3) 지하철역 로드/행정동 내부 선택
    # ─────────────────────────────────────────────────────
    try:  # UTF-8 우선
        sub_raw = pd.read_csv(SUBWAY_CSV, encoding="utf-8")  # CSV 로드
    except UnicodeDecodeError:  # UTF-8 실패 시
        sub_raw = pd.read_csv(SUBWAY_CSV, encoding="cp949")  # CP949 재시도

    sub_raw["경도"] = pd.to_numeric(sub_raw["경도"], errors="coerce")  # 경도 숫자화
    sub_raw["위도"] = pd.to_numeric(sub_raw["위도"], errors="coerce")  # 위도 숫자화
    sub_raw = sub_raw.dropna(subset=["경도", "위도"])  # 결측 제거

    gdf_sub = gpd.GeoDataFrame(  # 지하철 GeoDataFrame
        sub_raw,  # 원본
        geometry=gpd.points_from_xy(sub_raw["경도"], sub_raw["위도"]),  # Point 생성
        crs=MAP_CRS,  # 입력 좌표계(위경도)
    ).to_crs(TARGET_CRS)  # 분석 좌표계로 변환

    gdf_sub_sel = gdf_sub[gdf_sub.geometry.within(sel_union)].copy()  # 행정동 내부 역만

    # ─────────────────────────────────────────────────────
    # (4) 인구격자 로드/행정동 클립
    # ─────────────────────────────────────────────────────
    gdf_grid = gpd.read_file(GRID_SHP).to_crs(TARGET_CRS)  # 격자 로드 + CRS 변환
    gdf_grid["gid"] = gdf_grid["gid"].astype(str)  # gid 문자열화
    gdf_grid["pop"] = pd.to_numeric(gdf_grid.get("val", 0), errors="coerce").fillna(0.0)  # pop 생성

    gdf_grid_sel = gpd.clip(  # 행정동으로 클립
        gdf_grid[gdf_grid.geometry.intersects(sel_union)],  # 교차 후보만 먼저 추려 속도 개선
        gdf_sel,  # 클립 경계
    )[["gid", "pop", "geometry"]].copy()  # 필요한 컬럼만 유지

    gdf_grid_sel["centroid_m"] = gdf_grid_sel.geometry.centroid  # 격자 중심점(5179)

    # =========================================================
    # 5) (A) 직선 버퍼 커버/비커버
    # =========================================================
    bufs = []  # 버퍼 리스트

    if len(gdf_bus_sel) > 0:  # 버스가 있으면
        bufs.append(unary_union(gdf_bus_sel.geometry.buffer(BUS_BUFFER_M)))  # 버스 300m 버퍼

    if len(gdf_sub_sel) > 0:  # 지하철이 있으면
        bufs.append(unary_union(gdf_sub_sel.geometry.buffer(SUB_BUFFER_M)))  # 지하철 500m 버퍼

    cover_buf = unary_union(bufs) if len(bufs) > 0 else None  # 커버(버퍼) 유니온
    uncov_buf = sel_union.difference(cover_buf) if cover_buf else sel_union  # 비커버(버퍼)

    # =========================================================
    # 6) (B) 네트워크(Isochrone) 커버/비커버
    # =========================================================
    poly_graph_ll = (  # 네트워크 다운로드 범위(행정동 + buffer) -> 4326
        gpd.GeoSeries([sel_union.buffer(GRAPH_BUFFER_M)], crs=TARGET_CRS)  # 5179 buffer
        .to_crs(MAP_CRS)  # 4326 변환
        .iloc[0]  # 단일 geometry
    )

    ox.settings.log_console = False  # OSMnx 콘솔 로그 off
    G = ox.graph_from_polygon(poly_graph_ll, network_type="walk", simplify=True)  # 도보 네트워크 생성

    bus_ll = gdf_bus_sel.to_crs(MAP_CRS).copy()  # 버스(4326)
    sub_ll = gdf_sub_sel.to_crs(MAP_CRS).copy()  # 지하철(4326)

    bus_nodes = []  # 버스 노드 리스트
    if len(bus_ll) > 0:  # 버스가 있으면
        bus_nodes = list(ox.distance.nearest_nodes(G, X=bus_ll.geometry.x.values, Y=bus_ll.geometry.y.values))  # 노드 매핑

    subway_nodes = []  # 지하철 노드 리스트
    if len(sub_ll) > 0:  # 지하철이 있으면
        subway_nodes = list(ox.distance.nearest_nodes(G, X=sub_ll.geometry.x.values, Y=sub_ll.geometry.y.values))  # 노드 매핑

    # ---- isochrone 커버 폴리곤 만들기(정류장/역별 ego_graph) ----
    gdf_bus_sel2 = gdf_bus_sel.copy()  # 버스 복사
    gdf_bus_sel2["stop_type"] = "bus"  # 타입 부여

    gdf_sub_sel2 = gdf_sub_sel.copy()  # 지하철 복사
    gdf_sub_sel2["stop_type"] = "subway"  # 타입 부여

    gdf_stops = gpd.GeoDataFrame(  # 버스+지하철 합치기
        pd.concat([gdf_bus_sel2, gdf_sub_sel2], ignore_index=True),  # 결합
        geometry="geometry",  # geometry 지정
        crs=TARGET_CRS,  # CRS 지정
    )

    gdf_stops_ll = gdf_stops.to_crs(MAP_CRS).copy()  # 4326 변환

    if len(gdf_stops_ll) > 0:  # 정류장이 있으면
        gdf_stops_ll["v_node"] = ox.distance.nearest_nodes(  # 정류장별 최근접 노드
            G, X=gdf_stops_ll.geometry.x.values, Y=gdf_stops_ll.geometry.y.values
        )

    iso_polys = []  # isochrone 폴리곤 리스트

    for _, r in gdf_stops_ll.iterrows():  # 정류장 순회
        v = int(r["v_node"])  # 출발 노드
        iso_cutoff = BUS_BUFFER_M if r["stop_type"] == "bus" else SUB_BUFFER_M  # 타입별 cutoff

        try:  # ego_graph 생성 시도
            Gsub_iso = nx.ego_graph(G, v, radius=float(iso_cutoff), distance="length", undirected=True)  # 거리 기반 서브그래프
        except Exception:  # 실패하면
            continue  # 다음으로

        if Gsub_iso.number_of_edges() == 0:  # 엣지 없으면
            continue  # 다음으로

        _, gdf_edges = ox.graph_to_gdfs(Gsub_iso, nodes=True, edges=True, fill_edge_geometry=True)  # 엣지를 GeoDF로

        poly_m = unary_union(gdf_edges.to_crs(TARGET_CRS).geometry.buffer(EDGE_BUFFER_M))  # 도로폭 보정 버퍼 후 유니온

        if poly_m is not None and (not poly_m.is_empty):  # 유효하면
            iso_polys.append(poly_m)  # 추가

    cover_iso = unary_union(iso_polys) if len(iso_polys) > 0 else None  # isochrone 커버 유니온
    uncov_iso = sel_union.difference(cover_iso) if cover_iso else sel_union  # isochrone 비커버

    # =========================================================
    # 7) KPI + TOP 격자(버퍼/네트워크 각각)
    # =========================================================
    admin_area = sel_union.area  # 행정동 면적

    buf_mask = gdf_grid_sel["centroid_m"].within(uncov_buf) if (uncov_buf and not uncov_buf.is_empty) else pd.Series(False, index=gdf_grid_sel.index)  # 버퍼 비커버 마스크
    iso_mask = gdf_grid_sel["centroid_m"].within(uncov_iso) if (uncov_iso and not uncov_iso.is_empty) else pd.Series(False, index=gdf_grid_sel.index)  # isochrone 비커버 마스크

    buf_pop = float(gdf_grid_sel.loc[buf_mask, "pop"].sum())  # 버퍼 비커버 인구
    iso_pop = float(gdf_grid_sel.loc[iso_mask, "pop"].sum())  # isochrone 비커버 인구
    total_pop = float(gdf_grid_sel["pop"].sum())  # 총 인구

    buf_area = float(uncov_buf.area) if (uncov_buf and not uncov_buf.is_empty) else 0.0  # 버퍼 비커버 면적
    iso_area = float(uncov_iso.area) if (uncov_iso and not uncov_iso.is_empty) else 0.0  # isochrone 비커버 면적

    false_covered = (~buf_mask) & iso_mask  # 버퍼는 커버인데 네트워크는 비커버
    additional_pop = float(gdf_grid_sel.loc[false_covered, "pop"].sum())  # 추가 발견 인구

    top_buf = None  # 버퍼 TOP 격자
    top_iso = None  # 네트워크 TOP 격자

    if (uncov_buf is not None) and (not uncov_buf.is_empty):  # 버퍼 비커버가 유효하면
        cands = gdf_grid_sel[gdf_grid_sel["centroid_m"].within(uncov_buf)]  # 후보
        if len(cands) > 0:  # 후보가 있으면
            top_buf = cands.loc[cands["pop"].idxmax()].copy()  # pop 최대

    if (uncov_iso is not None) and (not uncov_iso.is_empty):  # 네트워크 비커버가 유효하면
        cands = gdf_grid_sel[gdf_grid_sel["centroid_m"].within(uncov_iso)]  # 후보
        if len(cands) > 0:  # 후보가 있으면
            top_iso = cands.loc[cands["pop"].idxmax()].copy()  # pop 최대

    # =========================================================
    # 8) “최종 격자” 정의 + 그 격자에서 모든 정류장/역까지 최단경로(라인) 복원
    #    - 최종 격자 우선순위: 네트워크 TOP(top_iso) 있으면 그걸 “최종”로 사용
    #    - 없으면 버퍼 TOP(top_buf)을 최종으로 사용
    # =========================================================
    final_top = top_iso if top_iso is not None else top_buf  # 최종 TOP 격자 선택
    final_top_mode = "네트워크TOP" if top_iso is not None else "버퍼TOP"  # 최종 선택 모드 문자열

    final_routes = []  # 최종 격자에서 그릴 경로(LineString) 리스트
    final_src_node = None  # 최종 격자 중심점에 매핑된 src 노드

    if DRAW_ALL_ROUTES and final_top is not None:  # 경로 그리기 ON + 최종 격자가 있으면
        final_cent_ll = gpd.GeoSeries([final_top["centroid_m"]], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 최종 격자 중심점(4326)
        final_src_node = ox.distance.nearest_nodes(G, X=float(final_cent_ll.x), Y=float(final_cent_ll.y))  # src 노드 매핑

        target_nodes = []  # 도착 노드 모음
        target_nodes.extend(list(bus_nodes))  # 버스 노드 추가
        target_nodes.extend(list(subway_nodes))  # 지하철 노드 추가
        target_nodes = list(dict.fromkeys(target_nodes))  # 중복 제거(순서 유지)

        if len(target_nodes) > MAX_DRAW_ROUTES:  # 경로가 너무 많으면
            target_nodes = target_nodes[:MAX_DRAW_ROUTES]  # 안전장치로 앞부분만 사용

        for tn in target_nodes:  # 각 도착 노드에 대해
            try:  # 경로 복원 시도
                path_nodes = nx.shortest_path(G, source=final_src_node, target=tn, weight="length")  # 최단경로 노드 리스트
            except Exception:  # 경로가 없거나 실패하면
                continue  # 다음으로

            try:  # 경로를 LineString으로 변환 시도
                gdf_route = ox.utils_graph.route_to_gdf(G, path_nodes, weight="length")  # 경로 엣지 GeoDF
                line = gdf_route["geometry"].unary_union  # 엣지를 합쳐 하나의 라인/멀티라인으로
            except Exception:  # 변환 실패하면
                continue  # 다음으로

            if line is None or line.is_empty:  # 비어있으면
                continue  # 다음으로

            if line.geom_type == "LineString":  # 단일 라인이면
                final_routes.append(line)  # 그대로 저장
            else:  # MultiLineString 등 복합이면
                parts = list(line.geoms)  # 파트 분해
                parts = [p for p in parts if (p is not None and (not p.is_empty))]  # 유효 파트만
                if len(parts) == 0:  # 파트가 없으면
                    continue  # 다음으로
                parts.sort(key=lambda g: g.length, reverse=True)  # 가장 긴 파트를 우선
                final_routes.append(parts[0])  # 가장 긴 라인을 대표로 저장

    # =========================================================
    # 9) 표시용(4326) geometry 준비
    # =========================================================
    cover_buf_ll = None  # 버퍼 커버(4326)
    uncov_buf_ll = None  # 버퍼 비커버(4326)
    cover_iso_ll = None  # isochrone 커버(4326)
    uncov_iso_ll = None  # isochrone 비커버(4326)

    if cover_buf is not None:  # 버퍼 커버가 있으면
        cover_buf_ll = gpd.GeoSeries([cover_buf.intersection(sel_union)], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 행정동 내로 clip + 4326
    if uncov_buf is not None and (not uncov_buf.is_empty):  # 버퍼 비커버가 있으면
        uncov_buf_ll = gpd.GeoSeries([uncov_buf], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 4326 변환

    if cover_iso is not None:  # isochrone 커버가 있으면
        cover_iso_ll = gpd.GeoSeries([cover_iso.intersection(sel_union).simplify(5)], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 단순화 + clip + 4326
    if uncov_iso is not None and (not uncov_iso.is_empty):  # isochrone 비커버가 있으면
        uncov_iso_ll = gpd.GeoSeries([uncov_iso.simplify(5)], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 단순화 + 4326

    top_buf_ll = None  # 버퍼 TOP(4326)
    top_iso_ll = None  # 네트워크 TOP(4326)
    final_top_ll = None  # 최종 TOP(4326)

    if top_buf is not None:  # 버퍼 TOP이 있으면
        top_buf_ll = gpd.GeoDataFrame([top_buf], geometry="geometry", crs=TARGET_CRS).to_crs(MAP_CRS)  # 4326 변환
    if top_iso is not None:  # 네트워크 TOP이 있으면
        top_iso_ll = gpd.GeoDataFrame([top_iso], geometry="geometry", crs=TARGET_CRS).to_crs(MAP_CRS)  # 4326 변환
    if final_top is not None:  # 최종 TOP이 있으면
        final_top_ll = gpd.GeoDataFrame([final_top], geometry="geometry", crs=TARGET_CRS).to_crs(MAP_CRS)  # 4326 변환

    kpi = dict(  # KPI 딕셔너리
        region_nm=region_nm,  # 이름
        buf_uncov_km2=buf_area / 1e6,  # km2 변환
        iso_uncov_km2=iso_area / 1e6,  # km2 변환
        buf_uncov_pop=buf_pop,  # 인구
        iso_uncov_pop=iso_pop,  # 인구
        buf_ratio=(buf_area / admin_area) if admin_area > 0 else 0,  # 비율
        iso_ratio=(iso_area / admin_area) if admin_area > 0 else 0,  # 비율
        additional_pop=additional_pop,  # 추가 인구
        total_pop=total_pop,  # 총 인구
        final_top_mode=final_top_mode,  # 최종 TOP 기준
        n_routes=len(final_routes),  # 그려지는 경로 개수
    )

# =========================================================
# 5) KPI 출력
# =========================================================
st.markdown("---")  # 구분선
st.subheader(f"KPI 비교 ({kpi['region_nm']})")  # KPI 제목
c1, c2, c3, c4 = st.columns(4)  # 4열

with c1:  # KPI1
    st.metric(  # 면적 KPI
        label="비커버 면적(네트워크)",  # 라벨
        value=f"{kpi['iso_uncov_km2']:.3f} km\u00b2",  # 값
        delta=f"{kpi['iso_uncov_km2'] - kpi['buf_uncov_km2']:+.3f} km\u00b2 (네트워크 − 버퍼)",  # 델타
        delta_color="inverse",  # 색상
    )

with c2:  # KPI2
    st.metric(  # 인구 KPI
        label="비커버 인구(네트워크)",  # 라벨
        value=f"{kpi['iso_uncov_pop']:,.0f} 명",  # 값
        delta=f"{kpi['iso_uncov_pop'] - kpi['buf_uncov_pop']:+,.0f} 명",  # 델타
        delta_color="inverse",  # 색상
    )

with c3:  # KPI3
    st.metric(  # 비율 KPI
        label="비커버 비율(네트워크)",  # 라벨
        value=f"{kpi['iso_ratio']:.1%}",  # 값
        delta=f"{(kpi['iso_ratio'] - kpi['buf_ratio'])*100:+.1f} %p",  # 델타(%p)
        delta_color="inverse",  # 색상
    )

with c4:  # KPI4
    st.metric(  # 추가 발견 KPI
        label="추가 발견 비커버 인구",  # 라벨
        value=f"{kpi['additional_pop']:,.0f} 명",  # 값
        help="직선 버퍼로는 커버된 것처럼 보이지만, 네트워크로는 도달 불가능한(비커버) 인구",  # 도움말
    )

st.caption(f"최종 격자 기준: {kpi['final_top_mode']} | 지도에 그려진 경로 수: {kpi['n_routes']}개")  # 최종 기준/경로 수 표시

# =========================================================
# 6) 지도 생성(스크립트)
# =========================================================
def number_badge_html(n, bg):  # 숫자 배지 HTML 생성
    return f"""
    <div style="
      width:28px;height:28px;border-radius:999px;
      background:{bg};color:#fff;font-weight:800;font-size:14px;
      display:flex;align-items:center;justify-content:center;
      border:2px solid #fff; box-shadow:0 2px 8px rgba(0,0,0,0.35);
    ">{n}</div>
    """  # HTML 반환

bus_icon = folium.Icon(color="blue", icon="bus", prefix="fa")  # 버스 아이콘
sub_icon = folium.Icon(color="orange", icon="subway", prefix="fa")  # 지하철 아이콘

def add_base_layers(m):  # 공통 레이어 추가
    folium.GeoJson(  # 행정동 경계
        sel_ll,  # 4326 경계
        name="행정동 경계",  # 레이어명
        style_function=lambda x: {"fillOpacity": 0.03, "color": "#444", "weight": 3},  # 스타일
        tooltip=folium.GeoJsonTooltip(fields=["region_nm"], aliases=["행정동"]),  # 툴팁
    ).add_to(m)  # 지도 추가

    for _, r in bus_ll.iterrows():  # 버스 마커
        folium.Marker(  # 마커 생성
            location=[r.geometry.y, r.geometry.x],  # 좌표
            tooltip=f"버스정류장 | {r.get('정류소명','')}",  # 툴팁
            icon=bus_icon,  # 아이콘
        ).add_to(m)  # 지도 추가

    for _, r in sub_ll.iterrows():  # 지하철 마커
        folium.Marker(  # 마커 생성
            location=[r.geometry.y, r.geometry.x],  # 좌표
            tooltip="지하철역",  # 툴팁
            icon=sub_icon,  # 아이콘
        ).add_to(m)  # 지도 추가

def add_top_grid(m, top_ll, poly_color, label):  # TOP 격자 표시
    if top_ll is None or len(top_ll) == 0:  # 없으면
        return  # 종료

    r = top_ll.iloc[0]  # 단일 행
    pop = float(r.get("pop", 0))  # 인구
    gid = r.get("gid", "")  # gid
    tip = f"{label} | gid={gid} | pop={pop:,.0f}"  # 툴팁

    folium.GeoJson(  # TOP 격자 폴리곤
        {"type": "Feature", "properties": {}, "geometry": mapping(r.geometry)},  # GeoJSON
        name=f"{label} TOP 격자",  # 레이어명
        style_function=lambda x: {"fillOpacity": 0.50, "fillColor": poly_color, "color": poly_color, "weight": 3},  # 스타일
        tooltip=tip,  # 툴팁
    ).add_to(m)  # 지도 추가

    c = r.geometry.centroid  # 중심점
    folium.Marker(  # 숫자 배지
        location=[c.y, c.x],  # 좌표
        tooltip=tip,  # 툴팁
        icon=folium.DivIcon(html=number_badge_html(1, poly_color)),  # 배지 아이콘
    ).add_to(m)  # 지도 추가

def add_routes(m, routes, name="최종 격자→정류장/역 최단경로"):  # 경로 라인 추가
    if routes is None or len(routes) == 0:  # 경로 없으면
        return  # 종료

    fg = folium.FeatureGroup(name=name, show=True)  # 경로 레이어 그룹

    for ls in routes:  # 각 라인에 대해
        if ls is None or ls.is_empty:  # 비어있으면
            continue  # 스킵

        folium.PolyLine(  # 라인 추가
            [(y, x) for x, y in ls.coords],  # (lon,lat) -> (lat,lon)로 변환
            weight=ROUTE_WEIGHT,  # 두께
            opacity=ROUTE_OPACITY,  # 투명도
            color=ROUTE_COLOR,  # 색
        ).add_to(fg)  # 레이어 그룹에 추가

    fg.add_to(m)  # 지도에 레이어 그룹 추가

# ---- (좌) 버퍼 지도 ----
m_buf = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")  # 지도 생성
add_base_layers(m_buf)  # 공통 레이어 추가

if cover_buf_ll is not None and (not cover_buf_ll.is_empty):  # 커버(버퍼) 있으면
    folium.GeoJson(  # 커버 폴리곤
        mapping(cover_buf_ll),  # GeoJSON
        name="커버(버퍼)",  # 레이어명
        style_function=lambda x: {"fillOpacity": 0.22, "fillColor": "#28a745", "color": "#28a745", "weight": 1},  # 스타일
    ).add_to(m_buf)  # 지도 추가

if uncov_buf_ll is not None and (not uncov_buf_ll.is_empty):  # 비커버(버퍼) 있으면
    folium.GeoJson(  # 비커버 폴리곤
        mapping(uncov_buf_ll),  # GeoJSON
        name="비커버(버퍼)",  # 레이어명
        style_function=lambda x: {"fillOpacity": 0.32, "fillColor": "#cc0000", "color": "#cc0000", "weight": 2},  # 스타일
    ).add_to(m_buf)  # 지도 추가

add_top_grid(m_buf, top_buf_ll, poly_color="#ff6600", label="버퍼 비커버 최대인구")  # 버퍼 TOP 표시

# ---- (우) 네트워크 지도 ----
m_iso = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")  # 지도 생성
add_base_layers(m_iso)  # 공통 레이어 추가

if cover_iso_ll is not None and (not cover_iso_ll.is_empty):  # 커버(Isochrone) 있으면
    folium.GeoJson(  # 커버 폴리곤
        mapping(cover_iso_ll),  # GeoJSON
        name="커버(Isochrone)",  # 레이어명
        style_function=lambda x: {"fillOpacity": 0.18, "fillColor": "#0066ff", "color": "#0066ff", "weight": 1},  # 스타일
    ).add_to(m_iso)  # 지도 추가

if uncov_iso_ll is not None and (not uncov_iso_ll.is_empty):  # 비커버(Isochrone) 있으면
    folium.GeoJson(  # 비커버 폴리곤
        mapping(uncov_iso_ll),  # GeoJSON
        name="비커버(Isochrone)",  # 레이어명
        style_function=lambda x: {"fillOpacity": 0.28, "fillColor": "#7a00cc", "color": "#7a00cc", "weight": 2},  # 스타일
    ).add_to(m_iso)  # 지도 추가

add_top_grid(m_iso, top_iso_ll, poly_color="#e91e63", label="네트워크 비커버 최대인구")  # 네트워크 TOP 표시

# ---- 최종 TOP 격자 및 경로는 “두 지도 모두”에 표시(원하면 한쪽만으로 바꿔도 됨) ----
if final_top_ll is not None:  # 최종 TOP이 있으면
    add_top_grid(m_buf, final_top_ll, poly_color="#111111", label=f"최종 TOP({final_top_mode})")  # 버퍼 지도에도 최종 TOP 표시
    add_top_grid(m_iso, final_top_ll, poly_color="#111111", label=f"최종 TOP({final_top_mode})")  # 네트워크 지도에도 최종 TOP 표시

if DRAW_ALL_ROUTES:  # 경로 그리기 ON이면
    add_routes(m_buf, final_routes, name=f"최종 TOP({final_top_mode})→정류장/역 최단경로")  # 버퍼 지도에 경로 추가
    add_routes(m_iso, final_routes, name=f"최종 TOP({final_top_mode})→정류장/역 최단경로")  # 네트워크 지도에 경로 추가

folium.LayerControl(collapsed=False).add_to(m_buf)  # 레이어 컨트롤
folium.LayerControl(collapsed=False).add_to(m_iso)  # 레이어 컨트롤

m_buf.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])  # 버퍼 지도 범위 맞춤
m_iso.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])  # 네트워크 지도 범위 맞춤

# =========================================================
# 7) 화면 배치
# =========================================================
st.markdown("---")  # 구분선
col_l, col_r = st.columns(2, gap="large")  # 2열 레이아웃

with col_l:  # 왼쪽
    st.subheader("직선 버퍼 기반 분석")  # 제목
    st_folium(m_buf, width=None, height=MAP_HEIGHT_PX, key="map_buf", returned_objects=[])  # 지도 렌더

with col_r:  # 오른쪽
    st.subheader("네트워크(Isochrone) 기반 분석")  # 제목
    st_folium(m_iso, width=None, height=MAP_HEIGHT_PX, key="map_iso", returned_objects=[])  # 지도 렌더

with st.expander("분석 방법론 비교"):  # 방법론 비교 섹션
    st.markdown(  # 표 출력
        """
| 항목 | 직선 버퍼 | 네트워크 기반 (Isochrone) |
|------|-----------|--------------------------|
| 방식 | 정류장 중심 원형 버퍼 (300 m / 500 m) | OSMnx 도보 네트워크 + ego_graph(거리=length) + 도로폭(25m) 버퍼 |
| 장점 | 계산 빠름, 직관적 | 실제 보행 경로 반영 + 경로 복원 가능 |
| 단점 | 장애물/단절 미반영 | OSM 다운로드/계산 필요 |
| 비커버 판단 | 원 바깥 = 비커버 | 도보 네트워크로 도달 불가 = 비커버 |
| 최종 격자 경로 | (공통) 최종 TOP 격자에서 각 정류장/역까지 최단경로 라인 표시 | (공통) 최종 TOP 격자에서 각 정류장/역까지 최단경로 라인 표시 |
        """
    )
