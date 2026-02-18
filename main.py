# =========================================================
# 0) 라이브러리
# =========================================================
import os, warnings  # os: 경로 처리 / warnings: 경고 처리
warnings.filterwarnings("ignore")  # 불필요한 경고 출력 숨김

import numpy as np  # 수치 계산
import pandas as pd  # 테이블 데이터 처리
import geopandas as gpd  # 공간 데이터 처리
import folium  # 지도 시각화(Folium)
import streamlit as st  # 스트림릿 UI
from streamlit_folium import st_folium  # 스트림릿에서 Folium 지도 렌더링
import osmnx as ox  # OSM 네트워크 다운로드/분석
import networkx as nx  # 최단거리/그래프 알고리즘
from shapely.ops import unary_union  # 여러 지오메트리 합치기
from shapely.geometry import mapping  # GeoJSON 변환용

# =========================================================
# 1) 상수 / 경로
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트 파일의 절대 경로 기준 디렉토리
DATA_DIR = os.path.join(BASE_DIR, "data")  # 데이터 폴더 경로

ADMIN_SHP  = os.path.join(DATA_DIR, "BND_ADM_DONG_PG.gpkg")  # 행정동 경계(GeoPackage)
BUS_XLSX   = os.path.join(DATA_DIR, "서울시버스정류소위치정보(20260108).xlsx")  # 버스정류장 엑셀
SUBWAY_CSV = os.path.join(DATA_DIR, "서울교통공사_1_8호선 역사 좌표(위경도) 정보_20250814.csv")  # 지하철역 CSV
GRID_SHP   = os.path.join(DATA_DIR, "nlsp_021001001.shp")  # 인구격자 shp

TARGET_IDS = {"11210630": "남현동", "11210540": "청림동"}  # 분석할 행정동(코드:이름)

TARGET_CRS = 5179  # 분석/거리 계산용 좌표계(EPSG:5179, meter)
MAP_CRS    = 4326  # 지도 표시용 좌표계(EPSG:4326, lat/lon)

BUS_BUFFER_M   = 300.0  # 버스 커버 기준(m)
SUB_BUFFER_M   = 500.0  # 지하철 커버 기준(m)
GRAPH_BUFFER_M = 1500.0  # 네트워크 다운로드 영역 여유 버퍼(m)
EDGE_BUFFER_M  = 25.0  # 이소크론 도로폭 보정(엣지 버퍼 m)

MAP_HEIGHT_PX = 650  # 지도 높이

# =========================================================
# 2) 페이지 설정
# =========================================================
st.set_page_config(page_title="대중교통 커버리지 비교", layout="wide")  # 스트림릿 페이지 설정

st.markdown(  # 간단한 CSS로 여백/타이포 조정
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

st.title("대중교통 커버리지 분석: 직선 버퍼 vs 네트워크 기반")  # 타이틀
st.caption("버스 300 m / 지하철 500 m 기준 · TOP 격자에서 각 정류장/역까지 네트워크 최단거리(m) 표시")  # 캡션

# =========================================================
# 3) 드롭다운(행정동 선택)
# =========================================================
st.markdown("---")  # 구분선
rid = st.selectbox(  # 행정동 선택 드롭다운
    "행정동 선택",
    options=list(TARGET_IDS.keys()),  # 선택 가능한 region_id 목록
    format_func=lambda x: f"{TARGET_IDS.get(x, x)} ({x})",  # 표시 포맷(이름 + 코드)
    index=0,  # 기본 선택 인덱스
)
st.caption(f"선택 행정동: {TARGET_IDS.get(rid)}")  # 선택 결과 표시

# =========================================================
# 4) 데이터 로드 + 분석 (스크립트)
# =========================================================
with st.spinner("데이터 로드/분석 중... (OSM 네트워크 다운로드 포함)"):  # 처리 중 스피너 표시
    # ─────────────────────────────────────────────────────
    # (1) 행정동 로드/선택
    # ─────────────────────────────────────────────────────
    gdf_admin = gpd.read_file(ADMIN_SHP)  # 행정동 경계 로드
    gdf_admin["region_id"] = gdf_admin["ADM_CD"].astype(str).str.strip()  # 행정동 코드 표준화
    gdf_admin["region_nm"] = gdf_admin["ADM_NM"].astype(str).str.strip()  # 행정동명 표준화
    gdf_admin = gdf_admin.to_crs(TARGET_CRS)  # 분석용 좌표계로 변환(거리 계산용)

    gdf_sel = gdf_admin[gdf_admin["region_id"] == rid].copy()  # 선택된 행정동만 필터링
    if len(gdf_sel) == 0:  # 선택이 실패한 경우
        st.error("선택한 행정동을 찾을 수 없습니다.")  # 에러 메시지
        st.stop()  # 앱 중단

    region_nm = gdf_sel["region_nm"].iloc[0]  # 행정동명
    sel_union = unary_union(gdf_sel.geometry)  # 멀티폴리곤이면 하나로 합치기

    sel_ll = gdf_sel.to_crs(MAP_CRS)  # 표시용(위경도)으로 변환
    bounds = sel_ll.total_bounds  # [minx, miny, maxx, maxy]
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]  # 지도 중심(lat, lon)

    # ─────────────────────────────────────────────────────
    # (2) 버스정류장 로드/행정동 내부 필터
    # ─────────────────────────────────────────────────────
    bus_raw = pd.read_excel(BUS_XLSX)  # 엑셀 로드
    bus_raw["X좌표"] = pd.to_numeric(bus_raw["X좌표"], errors="coerce")  # 경도 컬럼 숫자화
    bus_raw["Y좌표"] = pd.to_numeric(bus_raw["Y좌표"], errors="coerce")  # 위도 컬럼 숫자화
    bus_raw = bus_raw.dropna(subset=["X좌표", "Y좌표"])  # 좌표 결측 제거

    gdf_bus = gpd.GeoDataFrame(  # 버스 GeoDataFrame 생성
        bus_raw,  # 원본 테이블
        geometry=gpd.points_from_xy(bus_raw["X좌표"], bus_raw["Y좌표"]),  # 점 지오메트리 생성
        crs=MAP_CRS,  # 입력 좌표계(위경도 가정)
    ).to_crs(TARGET_CRS)  # 분석용 좌표계로 변환

    gdf_bus_sel = gdf_bus[gdf_bus.geometry.within(sel_union)].copy()  # 행정동 내부 점만 선택

    # ─────────────────────────────────────────────────────
    # (3) 지하철역 로드/행정동 내부 필터
    # ─────────────────────────────────────────────────────
    try:  # UTF-8로 먼저 시도
        sub_raw = pd.read_csv(SUBWAY_CSV, encoding="utf-8")  # 지하철 CSV 로드
    except UnicodeDecodeError:  # UTF-8 실패 시
        sub_raw = pd.read_csv(SUBWAY_CSV, encoding="cp949")  # CP949로 재시도

    sub_raw["경도"] = pd.to_numeric(sub_raw["경도"], errors="coerce")  # 경도 숫자화
    sub_raw["위도"] = pd.to_numeric(sub_raw["위도"], errors="coerce")  # 위도 숫자화
    sub_raw = sub_raw.dropna(subset=["경도", "위도"])  # 좌표 결측 제거

    gdf_sub = gpd.GeoDataFrame(  # 지하철 GeoDataFrame 생성
        sub_raw,  # 원본 테이블
        geometry=gpd.points_from_xy(sub_raw["경도"], sub_raw["위도"]),  # 점 지오메트리 생성
        crs=MAP_CRS,  # 입력 좌표계(위경도)
    ).to_crs(TARGET_CRS)  # 분석용 좌표계로 변환

    gdf_sub_sel = gdf_sub[gdf_sub.geometry.within(sel_union)].copy()  # 행정동 내부 역만 선택

    # ─────────────────────────────────────────────────────
    # (4) 인구격자 로드/행정동 클립
    # ─────────────────────────────────────────────────────
    gdf_grid = gpd.read_file(GRID_SHP).to_crs(TARGET_CRS)  # 인구격자 로드 + CRS 변환
    gdf_grid["gid"] = gdf_grid["gid"].astype(str)  # gid 문자열화(정렬/표시에 안전)
    gdf_grid["pop"] = pd.to_numeric(gdf_grid.get("val", 0), errors="coerce").fillna(0.0)  # 인구값(pop) 생성

    gdf_grid_sel = gpd.clip(  # 행정동 경계로 격자 클립
        gdf_grid[gdf_grid.geometry.intersects(sel_union)],  # 먼저 교차 격자만 줄여서 속도 개선
        gdf_sel,  # 클립 경계
    )[["gid", "pop", "geometry"]].copy()  # 필요한 컬럼만 유지

    gdf_grid_sel["centroid_m"] = gdf_grid_sel.geometry.centroid  # 격자 중심점(5179) 생성

    # =========================================================
    # 5) (A) 직선 버퍼 커버/비커버
    # =========================================================
    bufs = []  # 버스/지하철 버퍼를 담을 리스트

    if len(gdf_bus_sel) > 0:  # 버스정류장이 존재하면
        bufs.append(unary_union(gdf_bus_sel.geometry.buffer(BUS_BUFFER_M)))  # 버스 300m 버퍼 유니온

    if len(gdf_sub_sel) > 0:  # 지하철역이 존재하면
        bufs.append(unary_union(gdf_sub_sel.geometry.buffer(SUB_BUFFER_M)))  # 지하철 500m 버퍼 유니온

    cover_buf = unary_union(bufs) if len(bufs) > 0 else None  # 커버 영역(버퍼)
    uncov_buf = sel_union.difference(cover_buf) if cover_buf else sel_union  # 비커버(행정동-커버), 없으면 전체

    # =========================================================
    # 6) (B) 네트워크(Isochrone) 커버/비커버 + TOP 격자 최단거리(m)
    # =========================================================
    poly_graph_ll = (  # 네트워크 다운로드 범위(행정동 + 여유버퍼) -> 위경도로 변환
        gpd.GeoSeries([sel_union.buffer(GRAPH_BUFFER_M)], crs=TARGET_CRS)  # 5179에서 buffer 적용
        .to_crs(MAP_CRS)  # 4326으로 변환
        .iloc[0]  # 단일 지오메트리 추출
    )

    ox.settings.log_console = False  # OSMnx 로그 콘솔 출력 비활성화
    G = ox.graph_from_polygon(poly_graph_ll, network_type="walk", simplify=True)  # 도보 네트워크 다운로드

    bus_ll = gdf_bus_sel.to_crs(MAP_CRS).copy()  # 버스정류장(4326)
    sub_ll = gdf_sub_sel.to_crs(MAP_CRS).copy()  # 지하철역(4326)

    bus_nodes = []  # 버스 정류장에 매핑된 그래프 노드 id
    if len(bus_ll) > 0:  # 버스정류장이 있으면
        bus_nodes = list(ox.distance.nearest_nodes(G, X=bus_ll.geometry.x.values, Y=bus_ll.geometry.y.values))  # 노드 매핑

    subway_nodes = []  # 지하철역에 매핑된 그래프 노드 id
    if len(sub_ll) > 0:  # 지하철역이 있으면
        subway_nodes = list(ox.distance.nearest_nodes(G, X=sub_ll.geometry.x.values, Y=sub_ll.geometry.y.values))  # 노드 매핑

    # ─────────────────────────────────────────────────────
    # (B-1) isochrone 커버리지(정류장 기준 300/500m)  ※ 기존 기능 유지
    # ─────────────────────────────────────────────────────
    gdf_bus_sel2 = gdf_bus_sel.copy()  # 버스 선택본 복사
    gdf_bus_sel2["stop_type"] = "bus"  # 정류장 타입 표기

    gdf_sub_sel2 = gdf_sub_sel.copy()  # 지하철 선택본 복사
    gdf_sub_sel2["stop_type"] = "subway"  # 역 타입 표기

    gdf_stops = gpd.GeoDataFrame(  # 버스+지하철을 하나로 합친 정류장 레이어
        pd.concat([gdf_bus_sel2, gdf_sub_sel2], ignore_index=True),  # 행 결합
        geometry="geometry",  # 지오메트리 컬럼 지정
        crs=TARGET_CRS,  # 좌표계 지정
    )

    gdf_stops_ll = gdf_stops.to_crs(MAP_CRS).copy()  # 정류장(4326) 변환

    if len(gdf_stops_ll) > 0:  # 정류장이 하나라도 있으면
        gdf_stops_ll["v_node"] = ox.distance.nearest_nodes(  # 각 정류장을 그래프 노드에 매핑
            G, X=gdf_stops_ll.geometry.x.values, Y=gdf_stops_ll.geometry.y.values
        )

    iso_polys = []  # 정류장별 isochrone 폴리곤을 담는 리스트

    for _, r in gdf_stops_ll.iterrows():  # 정류장을 하나씩 순회
        v = int(r["v_node"])  # 출발 노드 id
        iso_cutoff = BUS_BUFFER_M if r["stop_type"] == "bus" else SUB_BUFFER_M  # 타입별 cutoff 거리

        try:  # ego_graph 생성 시도
            Gsub_iso = nx.ego_graph(  # 해당 노드에서 cutoff 이내 서브그래프 추출
                G, v, radius=float(iso_cutoff), distance="length", undirected=True
            )
        except Exception:  # 실패하면
            continue  # 다음 정류장으로

        if Gsub_iso.number_of_edges() == 0:  # 엣지가 없다면(유효 커버 없음)
            continue  # 다음으로

        _, gdf_edges = ox.graph_to_gdfs(  # 서브그래프를 GeoDataFrame으로 변환
            Gsub_iso, nodes=True, edges=True, fill_edge_geometry=True
        )

        poly_m = unary_union(  # 도로 엣지들을 합치고 도로폭 보정을 위해 버퍼 적용
            gdf_edges.to_crs(TARGET_CRS).geometry.buffer(EDGE_BUFFER_M)
        )

        if poly_m is not None and (not poly_m.is_empty):  # 폴리곤이 유효하면
            iso_polys.append(poly_m)  # 리스트에 추가

    cover_iso = unary_union(iso_polys) if len(iso_polys) > 0 else None  # isochrone 커버 유니온
    uncov_iso = sel_union.difference(cover_iso) if cover_iso else sel_union  # 비커버(행정동-커버), 없으면 전체

    # =========================================================
    # 7) KPI + TOP 격자(버퍼/네트워크 각각)
    # =========================================================
    admin_area = sel_union.area  # 행정동 전체 면적(m^2)

    buf_mask = (  # 버퍼 비커버 내에 있는 격자 중심점 마스크
        gdf_grid_sel["centroid_m"].within(uncov_buf)
        if (uncov_buf and not uncov_buf.is_empty)
        else pd.Series(False, index=gdf_grid_sel.index)
    )

    iso_mask = (  # 네트워크 비커버 내에 있는 격자 중심점 마스크
        gdf_grid_sel["centroid_m"].within(uncov_iso)
        if (uncov_iso and not uncov_iso.is_empty)
        else pd.Series(False, index=gdf_grid_sel.index)
    )

    buf_pop = float(gdf_grid_sel.loc[buf_mask, "pop"].sum())  # 버퍼 비커버 인구 합
    iso_pop = float(gdf_grid_sel.loc[iso_mask, "pop"].sum())  # 네트워크 비커버 인구 합
    total_pop = float(gdf_grid_sel["pop"].sum())  # 전체 인구 합

    buf_area = float(uncov_buf.area) if (uncov_buf and not uncov_buf.is_empty) else 0.0  # 버퍼 비커버 면적
    iso_area = float(uncov_iso.area) if (uncov_iso and not uncov_iso.is_empty) else 0.0  # 네트워크 비커버 면적

    false_covered = (~buf_mask) & iso_mask  # 버퍼에서는 커버인데 네트워크에서는 비커버인 격자
    additional_pop = float(gdf_grid_sel.loc[false_covered, "pop"].sum())  # 추가 발견 비커버 인구

    top_buf = None  # 버퍼 비커버 최대인구 격자(행)
    top_iso = None  # 네트워크 비커버 최대인구 격자(행)

    if (uncov_buf is not None) and (not uncov_buf.is_empty):  # 버퍼 비커버가 유효하면
        cands = gdf_grid_sel[gdf_grid_sel["centroid_m"].within(uncov_buf)]  # 후보 격자
        if len(cands) > 0:  # 후보가 있으면
            top_buf = cands.loc[cands["pop"].idxmax()].copy()  # pop 최대 격자 선택

    if (uncov_iso is not None) and (not uncov_iso.is_empty):  # 네트워크 비커버가 유효하면
        cands = gdf_grid_sel[gdf_grid_sel["centroid_m"].within(uncov_iso)]  # 후보 격자
        if len(cands) > 0:  # 후보가 있으면
            top_iso = cands.loc[cands["pop"].idxmax()].copy()  # pop 최대 격자 선택

    # =========================================================
    # 8) TOP 격자 → 각 버스정류장/지하철역 "최단거리(m)" 계산 (경로 라인X)
    # =========================================================
    dist_top_buf_bus = {}  # 버퍼 TOP에서 각 버스노드까지 최단거리(m) {node:dist}
    dist_top_buf_sub = {}  # 버퍼 TOP에서 각 지하철노드까지 최단거리(m) {node:dist}
    dist_top_iso_bus = {}  # 네트워크 TOP에서 각 버스노드까지 최단거리(m) {node:dist}
    dist_top_iso_sub = {}  # 네트워크 TOP에서 각 지하철노드까지 최단거리(m) {node:dist}

    def _all_shortest_lengths_from_src(G, src):  # src에서 모든 노드까지 최단거리(가중치=length) 딕셔너리 생성
        try:
            lengths = nx.single_source_dijkstra_path_length(G, src, weight="length")  # cutoff 없이 전체 최단거리 계산
            return lengths  # {node: distance_m} 반환
        except Exception:
            return {}  # 실패하면 빈 dict 반환

    if top_buf is not None:  # 버퍼 TOP 격자가 있으면
        top_buf_cent_ll = gpd.GeoSeries([top_buf["centroid_m"]], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # TOP 중심점(4326)
        src = ox.distance.nearest_nodes(G, X=float(top_buf_cent_ll.x), Y=float(top_buf_cent_ll.y))  # src 노드 매핑
        lengths = _all_shortest_lengths_from_src(G, src)  # src→모든 노드 최단거리 계산(한 번)
        dist_top_buf_bus = {bn: float(lengths[bn]) for bn in bus_nodes if bn in lengths}  # 버스노드만 추출
        dist_top_buf_sub = {sn: float(lengths[sn]) for sn in subway_nodes if sn in lengths}  # 지하철노드만 추출

    if top_iso is not None:  # 네트워크 TOP 격자가 있으면
        top_iso_cent_ll = gpd.GeoSeries([top_iso["centroid_m"]], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # TOP 중심점(4326)
        src = ox.distance.nearest_nodes(G, X=float(top_iso_cent_ll.x), Y=float(top_iso_cent_ll.y))  # src 노드 매핑
        lengths = _all_shortest_lengths_from_src(G, src)  # src→모든 노드 최단거리 계산(한 번)
        dist_top_iso_bus = {bn: float(lengths[bn]) for bn in bus_nodes if bn in lengths}  # 버스노드만 추출
        dist_top_iso_sub = {sn: float(lengths[sn]) for sn in subway_nodes if sn in lengths}  # 지하철노드만 추출

    # ─────────────────────────────────────────────────────
    # (8-1) 거리표(DataFrame) 만들기: 마커에 붙일 인덱스 정렬을 위해 "행 순서 기준"으로 매핑
    # ─────────────────────────────────────────────────────
    bus_node_series = pd.Series(bus_nodes, index=bus_ll.index) if len(bus_ll) > 0 else pd.Series(dtype="object")  # 버스 row→node 매핑
    sub_node_series = pd.Series(subway_nodes, index=sub_ll.index) if len(sub_ll) > 0 else pd.Series(dtype="object")  # 지하철 row→node 매핑

    df_bus_dist_buf = pd.DataFrame()  # 버퍼 TOP→버스 거리표
    df_sub_dist_buf = pd.DataFrame()  # 버퍼 TOP→지하철 거리표
    df_bus_dist_iso = pd.DataFrame()  # 네트워크 TOP→버스 거리표
    df_sub_dist_iso = pd.DataFrame()  # 네트워크 TOP→지하철 거리표

    if len(bus_ll) > 0 and top_buf is not None:  # 버스가 있고 버퍼 TOP이 있으면
        df_bus_dist_buf = bus_ll.copy()  # 표시용 테이블 생성(geometry 포함)
        df_bus_dist_buf["v_node"] = bus_node_series  # 노드 id 컬럼 추가
        df_bus_dist_buf["dist_m"] = df_bus_dist_buf["v_node"].map(dist_top_buf_bus)  # 최단거리(m) 매핑
        df_bus_dist_buf = df_bus_dist_buf.dropna(subset=["dist_m"]).sort_values("dist_m")  # 도달 가능 대상만 + 거리순 정렬

    if len(sub_ll) > 0 and top_buf is not None:  # 지하철이 있고 버퍼 TOP이 있으면
        df_sub_dist_buf = sub_ll.copy()  # 표시용 테이블 생성(geometry 포함)
        df_sub_dist_buf["v_node"] = sub_node_series  # 노드 id 컬럼 추가
        df_sub_dist_buf["dist_m"] = df_sub_dist_buf["v_node"].map(dist_top_buf_sub)  # 최단거리(m) 매핑
        df_sub_dist_buf = df_sub_dist_buf.dropna(subset=["dist_m"]).sort_values("dist_m")  # 도달 가능 대상만 + 거리순 정렬

    if len(bus_ll) > 0 and top_iso is not None:  # 버스가 있고 네트워크 TOP이 있으면
        df_bus_dist_iso = bus_ll.copy()  # 표시용 테이블 생성(geometry 포함)
        df_bus_dist_iso["v_node"] = bus_node_series  # 노드 id 컬럼 추가
        df_bus_dist_iso["dist_m"] = df_bus_dist_iso["v_node"].map(dist_top_iso_bus)  # 최단거리(m) 매핑
        df_bus_dist_iso = df_bus_dist_iso.dropna(subset=["dist_m"]).sort_values("dist_m")  # 도달 가능 대상만 + 거리순 정렬

    if len(sub_ll) > 0 and top_iso is not None:  # 지하철이 있고 네트워크 TOP이 있으면
        df_sub_dist_iso = sub_ll.copy()  # 표시용 테이블 생성(geometry 포함)
        df_sub_dist_iso["v_node"] = sub_node_series  # 노드 id 컬럼 추가
        df_sub_dist_iso["dist_m"] = df_sub_dist_iso["v_node"].map(dist_top_iso_sub)  # 최단거리(m) 매핑
        df_sub_dist_iso = df_sub_dist_iso.dropna(subset=["dist_m"]).sort_values("dist_m")  # 도달 가능 대상만 + 거리순 정렬

    # =========================================================
    # 9) 표시용(4326) geometry 준비
    # =========================================================
    cover_buf_ll = None  # 버퍼 커버(4326)
    uncov_buf_ll = None  # 버퍼 비커버(4326)
    cover_iso_ll = None  # 네트워크 커버(4326)
    uncov_iso_ll = None  # 네트워크 비커버(4326)

    if cover_buf is not None:  # 커버 버퍼가 있으면
        cover_buf_ll = gpd.GeoSeries([cover_buf.intersection(sel_union)], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 행정동 내로 클립
    if uncov_buf is not None and (not uncov_buf.is_empty):  # 비커버 버퍼가 유효하면
        uncov_buf_ll = gpd.GeoSeries([uncov_buf], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 4326 변환

    if cover_iso is not None:  # 커버 isochrone이 있으면
        cover_iso_ll = gpd.GeoSeries([cover_iso.intersection(sel_union).simplify(5)], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 단순화 후 클립
    if uncov_iso is not None and (not uncov_iso.is_empty):  # 비커버 isochrone이 유효하면
        uncov_iso_ll = gpd.GeoSeries([uncov_iso.simplify(5)], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 단순화 후 변환

    top_buf_ll = None  # TOP 버퍼 격자(4326)
    top_iso_ll = None  # TOP 네트워크 격자(4326)

    if top_buf is not None:  # TOP 버퍼 격자가 있으면
        top_buf_ll = gpd.GeoDataFrame([top_buf], geometry="geometry", crs=TARGET_CRS).to_crs(MAP_CRS)  # GeoDF로 만들어 4326 변환
    if top_iso is not None:  # TOP 네트워크 격자가 있으면
        top_iso_ll = gpd.GeoDataFrame([top_iso], geometry="geometry", crs=TARGET_CRS).to_crs(MAP_CRS)  # GeoDF로 만들어 4326 변환

    kpi = dict(  # KPI 딕셔너리 구성
        region_nm=region_nm,  # 행정동명
        buf_uncov_km2=buf_area / 1e6,  # 버퍼 비커버 면적(km2)
        iso_uncov_km2=iso_area / 1e6,  # 네트워크 비커버 면적(km2)
        buf_uncov_pop=buf_pop,  # 버퍼 비커버 인구
        iso_uncov_pop=iso_pop,  # 네트워크 비커버 인구
        buf_ratio=(buf_area / admin_area) if admin_area > 0 else 0,  # 버퍼 비커버 비율
        iso_ratio=(iso_area / admin_area) if admin_area > 0 else 0,  # 네트워크 비커버 비율
        additional_pop=additional_pop,  # 추가 발견 비커버 인구
        total_pop=total_pop,  # 총 인구
    )

# =========================================================
# 5) KPI 출력
# =========================================================
st.markdown("---")  # 구분선
st.subheader(f"KPI 비교 ({kpi['region_nm']})")  # KPI 섹션 타이틀
c1, c2, c3, c4 = st.columns(4)  # 4열 레이아웃

with c1:  # 첫 번째 KPI 카드
    st.metric(  # KPI 표시
        label="비커버 면적(네트워크)",  # 라벨
        value=f"{kpi['iso_uncov_km2']:.3f} km\u00b2",  # 값
        delta=f"{kpi['iso_uncov_km2'] - kpi['buf_uncov_km2']:+.3f} km\u00b2 (네트워크 − 버퍼)",  # 비교(네트워크-버퍼)
        delta_color="inverse",  # 차이 색상 반전
    )

with c2:  # 두 번째 KPI 카드
    st.metric(  # KPI 표시
        label="비커버 인구(네트워크)",  # 라벨
        value=f"{kpi['iso_uncov_pop']:,.0f} 명",  # 값
        delta=f"{kpi['iso_uncov_pop'] - kpi['buf_uncov_pop']:+,.0f} 명",  # 비교
        delta_color="inverse",  # 차이 색상 반전
    )

with c3:  # 세 번째 KPI 카드
    st.metric(  # KPI 표시
        label="비커버 비율(네트워크)",  # 라벨
        value=f"{kpi['iso_ratio']:.1%}",  # 값
        delta=f"{(kpi['iso_ratio'] - kpi['buf_ratio'])*100:+.1f} %p",  # 비교(%p)
        delta_color="inverse",  # 차이 색상 반전
    )

with c4:  # 네 번째 KPI 카드
    st.metric(  # KPI 표시
        label="추가 발견 비커버 인구",  # 라벨
        value=f"{kpi['additional_pop']:,.0f} 명",  # 값
        help="직선 버퍼로는 커버된 것처럼 보이지만, 네트워크로는 도달 불가능한(비커버) 인구",  # 도움말
    )

# =========================================================
# 6) TOP 격자 → 각 정류장/역 최단거리 표 출력 (전부)
# =========================================================
st.markdown("---")  # 구분선
st.subheader("TOP 격자 → 각 버스정류장/지하철역 네트워크 최단거리(m)")  # 섹션 타이틀

tab1, tab2 = st.tabs(["직선 버퍼 TOP 기준", "네트워크 TOP 기준"])  # 두 기준 탭 분리

with tab1:  # 버퍼 TOP 기준 탭
    st.caption("버퍼 비커버 최대인구 TOP 격자에서 각 정류장/역까지의 네트워크 최단거리(m)")  # 설명
    c1, c2 = st.columns(2)  # 2열 배치
    with c1:  # 왼쪽
        st.markdown("**버스정류장 거리표**")  # 제목
        if len(df_bus_dist_buf) > 0:  # 테이블이 있으면
            st.dataframe(df_bus_dist_buf.drop(columns=["geometry"], errors="ignore")[["dist_m"] + [c for c in df_bus_dist_buf.columns if c not in ["geometry", "dist_m"]]][:])  # 전체 출력
        else:  # 테이블이 없으면
            st.info("버퍼 TOP 기준으로 도달 가능한 버스정류장이 없습니다.")  # 안내
    with c2:  # 오른쪽
        st.markdown("**지하철역 거리표**")  # 제목
        if len(df_sub_dist_buf) > 0:  # 테이블이 있으면
            st.dataframe(df_sub_dist_buf.drop(columns=["geometry"], errors="ignore")[["dist_m"] + [c for c in df_sub_dist_buf.columns if c not in ["geometry", "dist_m"]]][:])  # 전체 출력
        else:  # 테이블이 없으면
            st.info("버퍼 TOP 기준으로 도달 가능한 지하철역이 없습니다.")  # 안내

with tab2:  # 네트워크 TOP 기준 탭
    st.caption("네트워크 비커버 최대인구 TOP 격자에서 각 정류장/역까지의 네트워크 최단거리(m)")  # 설명
    c1, c2 = st.columns(2)  # 2열 배치
    with c1:  # 왼쪽
        st.markdown("**버스정류장 거리표**")  # 제목
        if len(df_bus_dist_iso) > 0:  # 테이블이 있으면
            st.dataframe(df_bus_dist_iso.drop(columns=["geometry"], errors="ignore")[["dist_m"] + [c for c in df_bus_dist_iso.columns if c not in ["geometry", "dist_m"]]][:])  # 전체 출력
        else:  # 테이블이 없으면
            st.info("네트워크 TOP 기준으로 도달 가능한 버스정류장이 없습니다.")  # 안내
    with c2:  # 오른쪽
        st.markdown("**지하철역 거리표**")  # 제목
        if len(df_sub_dist_iso) > 0:  # 테이블이 있으면
            st.dataframe(df_sub_dist_iso.drop(columns=["geometry"], errors="ignore")[["dist_m"] + [c for c in df_sub_dist_iso.columns if c not in ["geometry", "dist_m"]]][:])  # 전체 출력
        else:  # 테이블이 없으면
            st.info("네트워크 TOP 기준으로 도달 가능한 지하철역이 없습니다.")  # 안내

# =========================================================
# 7) 지도 생성(스크립트)
#    - 클러스터 없음
#    - 버스/지하철 아이콘: FontAwesome
#    - 마커 tooltip/popup에 "TOP→정류장 최단거리(m)" 표시
# =========================================================
def number_badge_html(n, bg):  # 숫자 배지(동그라미) HTML 생성 함수
    return f"""
    <div style="
      width:28px;height:28px;border-radius:999px;
      background:{bg};color:#fff;font-weight:800;font-size:14px;
      display:flex;align-items:center;justify-content:center;
      border:2px solid #fff; box-shadow:0 2px 8px rgba(0,0,0,0.35);
    ">{n}</div>
    """  # 스타일 포함 HTML 반환

bus_icon = folium.Icon(color="blue", icon="bus", prefix="fa")  # 버스 마커 아이콘
sub_icon = folium.Icon(color="orange", icon="subway", prefix="fa")  # 지하철 마커 아이콘

def add_base_layers(m, bus_node_to_dist=None, sub_node_to_dist=None, bus_nodes_by_row=None, sub_nodes_by_row=None, dist_label="TOP→최단거리"):  # 공통 레이어(경계+정류장/역) 추가 함수
    folium.GeoJson(  # 행정동 경계 폴리곤
        sel_ll,  # 4326 행정동
        name="행정동 경계",  # 레이어명
        style_function=lambda x: {"fillOpacity": 0.03, "color": "#444", "weight": 3},  # 스타일
        tooltip=folium.GeoJsonTooltip(fields=["region_nm"], aliases=["행정동"]),  # 툴팁
    ).add_to(m)  # 지도에 추가

    # ── 버스정류장(클러스터 없음) ──
    for idx, r in bus_ll.iterrows():  # 버스정류장을 행 단위로 순회
        v = bus_nodes_by_row.get(idx) if bus_nodes_by_row is not None else None  # 해당 마커의 그래프 노드 id
        d = bus_node_to_dist.get(v) if (bus_node_to_dist is not None and v is not None) else None  # TOP→해당 정류장 최단거리(m)
        d_txt = f"{dist_label}: {d:,.1f} m" if d is not None else f"{dist_label}: 도달불가"  # 거리 텍스트 구성
        tip = f"버스정류장 | {r.get('정류소명','')}<br>{d_txt}"  # tooltip HTML 구성
        folium.Marker(  # 마커 생성
            location=[r.geometry.y, r.geometry.x],  # lat, lon
            tooltip=folium.Tooltip(tip, sticky=True),  # 툴팁(HTML)
            popup=folium.Popup(tip, max_width=320),  # 팝업(HTML)
            icon=bus_icon,  # 아이콘
        ).add_to(m)  # 지도에 추가

    # ── 지하철역(클러스터 없음) ──
    for idx, r in sub_ll.iterrows():  # 지하철역을 행 단위로 순회
        v = sub_nodes_by_row.get(idx) if sub_nodes_by_row is not None else None  # 해당 마커의 그래프 노드 id
        d = sub_node_to_dist.get(v) if (sub_node_to_dist is not None and v is not None) else None  # TOP→해당 역 최단거리(m)
        d_txt = f"{dist_label}: {d:,.1f} m" if d is not None else f"{dist_label}: 도달불가"  # 거리 텍스트 구성
        tip = f"지하철역<br>{d_txt}"  # tooltip HTML 구성
        folium.Marker(  # 마커 생성
            location=[r.geometry.y, r.geometry.x],  # lat, lon
            tooltip=folium.Tooltip(tip, sticky=True),  # 툴팁(HTML)
            popup=folium.Popup(tip, max_width=280),  # 팝업(HTML)
            icon=sub_icon,  # 아이콘
        ).add_to(m)  # 지도에 추가

def add_top_grid(m, top_ll, poly_color, label):  # TOP 격자만 표시(경로 없음)
    if top_ll is not None and len(top_ll) > 0:  # TOP 격자가 있으면
        r = top_ll.iloc[0]  # 첫 행(단일)
        pop = float(r.get("pop", 0))  # 인구
        gid = r.get("gid", "")  # gid
        tip = f"{label} | gid={gid} | pop={pop:,.0f}"  # 툴팁 텍스트

        folium.GeoJson(  # TOP 격자 폴리곤 그리기
            {"type": "Feature", "properties": {}, "geometry": mapping(r.geometry)},  # GeoJSON
            name=f"{label} TOP 격자",  # 레이어명
            style_function=lambda x: {"fillOpacity": 0.50, "fillColor": poly_color, "color": poly_color, "weight": 3},  # 스타일
            tooltip=tip,  # 툴팁
        ).add_to(m)  # 지도에 추가

        c = r.geometry.centroid  # 격자 중심점
        folium.Marker(  # 숫자 뱃지 마커(1)
            location=[c.y, c.x],  # lat, lon
            tooltip=tip,  # 툴팁
            icon=folium.DivIcon(html=number_badge_html(1, poly_color)),  # 배지 HTML
        ).add_to(m)  # 지도에 추가

# ─────────────────────────────────────────────────────────
# 8) (좌) 버퍼 지도 생성
# ─────────────────────────────────────────────────────────
m_buf = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")  # 베이스 지도 생성

bus_nodes_by_row = bus_node_series.to_dict() if len(bus_node_series) > 0 else {}  # 버스 row→node dict
sub_nodes_by_row = sub_node_series.to_dict() if len(sub_node_series) > 0 else {}  # 지하철 row→node dict

add_base_layers(  # 공통 레이어 + (버퍼 TOP 기준) 거리 표시 마커
    m_buf,  # 지도
    bus_node_to_dist=dist_top_buf_bus,  # 버퍼 TOP→버스 거리 dict
    sub_node_to_dist=dist_top_buf_sub,  # 버퍼 TOP→지하철 거리 dict
    bus_nodes_by_row=bus_nodes_by_row,  # row→node 매핑
    sub_nodes_by_row=sub_nodes_by_row,  # row→node 매핑
    dist_label="버퍼TOP→최단거리",  # 라벨
)

if cover_buf_ll is not None and (not cover_buf_ll.is_empty):  # 버퍼 커버가 있으면
    folium.GeoJson(  # 커버 폴리곤
        mapping(cover_buf_ll),  # GeoJSON 변환
        name="커버(버퍼)",  # 레이어명
        style_function=lambda x: {"fillOpacity": 0.22, "fillColor": "#28a745", "color": "#28a745", "weight": 1},  # 스타일
    ).add_to(m_buf)  # 지도에 추가

if uncov_buf_ll is not None and (not uncov_buf_ll.is_empty):  # 버퍼 비커버가 있으면
    folium.GeoJson(  # 비커버 폴리곤
        mapping(uncov_buf_ll),  # GeoJSON 변환
        name="비커버(버퍼)",  # 레이어명
        style_function=lambda x: {"fillOpacity": 0.32, "fillColor": "#cc0000", "color": "#cc0000", "weight": 2},  # 스타일
    ).add_to(m_buf)  # 지도에 추가

add_top_grid(m_buf, top_buf_ll, poly_color="#ff6600", label="버퍼 비커버 최대인구")  # 버퍼 TOP 격자 표시

folium.LayerControl(collapsed=False).add_to(m_buf)  # 레이어 컨트롤
m_buf.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])  # 행정동 bounds로 확대

# ─────────────────────────────────────────────────────────
# 9) (우) 네트워크 지도 생성
# ─────────────────────────────────────────────────────────
m_iso = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")  # 베이스 지도 생성

add_base_layers(  # 공통 레이어 + (네트워크 TOP 기준) 거리 표시 마커
    m_iso,  # 지도
    bus_node_to_dist=dist_top_iso_bus,  # 네트워크 TOP→버스 거리 dict
    sub_node_to_dist=dist_top_iso_sub,  # 네트워크 TOP→지하철 거리 dict
    bus_nodes_by_row=bus_nodes_by_row,  # row→node 매핑
    sub_nodes_by_row=sub_nodes_by_row,  # row→node 매핑
    dist_label="네트워크TOP→최단거리",  # 라벨
)

if cover_iso_ll is not None and (not cover_iso_ll.is_empty):  # 네트워크 커버가 있으면
    folium.GeoJson(  # 커버 폴리곤
        mapping(cover_iso_ll),  # GeoJSON 변환
        name="커버(Isochrone)",  # 레이어명
        style_function=lambda x: {"fillOpacity": 0.18, "fillColor": "#0066ff", "color": "#0066ff", "weight": 1},  # 스타일
    ).add_to(m_iso)  # 지도에 추가

if uncov_iso_ll is not None and (not uncov_iso_ll.is_empty):  # 네트워크 비커버가 있으면
    folium.GeoJson(  # 비커버 폴리곤
        mapping(uncov_iso_ll),  # GeoJSON 변환
        name="비커버(Isochrone)",  # 레이어명
        style_function=lambda x: {"fillOpacity": 0.28, "fillColor": "#7a00cc", "color": "#7a00cc", "weight": 2},  # 스타일
    ).add_to(m_iso)  # 지도에 추가

add_top_grid(m_iso, top_iso_ll, poly_color="#e91e63", label="네트워크 비커버 최대인구")  # 네트워크 TOP 격자 표시

folium.LayerControl(collapsed=False).add_to(m_iso)  # 레이어 컨트롤
m_iso.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])  # 행정동 bounds로 확대

# =========================================================
# 10) 화면 배치
# =========================================================
st.markdown("---")  # 구분선
col_l, col_r = st.columns(2, gap="large")  # 좌/우 2열 레이아웃

with col_l:  # 왼쪽
    st.subheader("직선 버퍼 기반 분석")  # 서브타이틀
    st_folium(m_buf, width=None, height=MAP_HEIGHT_PX, key="map_buf", returned_objects=[])  # 지도 렌더

with col_r:  # 오른쪽
    st.subheader("네트워크(Isochrone) 기반 분석")  # 서브타이틀
    st_folium(m_iso, width=None, height=MAP_HEIGHT_PX, key="map_iso", returned_objects=[])  # 지도 렌더

with st.expander("분석 방법론 비교"):  # 접기/펼치기 섹션
    st.markdown(  # 비교 표 출력
        """
| 항목 | 직선 버퍼 | 네트워크 기반 (Isochrone) |
|------|-----------|--------------------------|
| 방식 | 정류장 중심 원형 버퍼 (300 m / 500 m) | OSMnx 도보 네트워크 + ego_graph(거리=length) + 도로폭(25m) 버퍼 |
| 장점 | 계산 빠름, 직관적 | 실제 보행 경로 반영 + 네트워크 거리 산출 가능 |
| 단점 | 장애물/단절 미반영 | OSM 다운로드/계산 필요 |
| 비커버 판단 | 원 바깥 = 비커버 | 도보 네트워크로 도달 불가 = 비커버 |
| TOP 기준 결과 | TOP 격자에서 각 정류장/역까지 최단거리(m) 표시 | TOP 격자에서 각 정류장/역까지 최단거리(m) 표시 |
        """
    )
