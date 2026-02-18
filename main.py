# =========================================================
# 0) 라이브러리
# =========================================================
import os, warnings                          # os: 경로 처리 / warnings: 경고 처리
warnings.filterwarnings("ignore")            # 모든 경고 메시지 출력 억제

import pandas as pd                          # 표 형태 데이터 처리
import geopandas as gpd                      # 공간 벡터 데이터 처리
import folium                                # 웹 기반 인터랙티브 지도 시각화
import streamlit as st                       # Streamlit 웹 앱 UI 프레임워크
from streamlit_folium import st_folium       # Streamlit 안에서 Folium 지도 렌더링
import osmnx as ox                           # OpenStreetMap 도보 네트워크 다운로드/처리
import networkx as nx                        # 그래프 자료구조 및 최단경로 알고리즘
from shapely.ops import unary_union          # 여러 Shapely geometry를 하나로 합치는 함수
from shapely.geometry import mapping, LineString  # GeoJSON 변환 / 선 도형 생성

# =========================================================
# 1) 상수 / 경로
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))         # 이 스크립트 파일이 위치한 폴더 (절대경로)
DATA_DIR = os.path.join(BASE_DIR, "data")                     # 데이터 파일 보관 하위 폴더

ADMIN_SHP  = os.path.join(DATA_DIR, "BND_ADM_DONG_PG.gpkg")  # 행정동 경계 파일 (GeoPackage)
BUS_XLSX   = os.path.join(DATA_DIR, "서울시버스정류소위치정보(20260108).xlsx")  # 버스정류장 위치 엑셀
SUBWAY_CSV = os.path.join(DATA_DIR, "서울교통공사_1_8호선 역사 좌표(위경도) 정보_20250814.csv")  # 지하철역 CSV
GRID_SHP   = os.path.join(DATA_DIR, "nlsp_021001001.shp")    # 인구 격자(100m×100m) 쉐이프파일

TARGET_IDS = {"11210630": "남현동", "11210540": "청림동"}     # 분석 대상 행정동 코드 → 이름 매핑

TARGET_CRS = 5179   # 분석·거리 계산용 좌표계: EPSG:5179 (한국 중부원점, 단위: m)
MAP_CRS    = 4326   # 지도 표시용 좌표계: EPSG:4326 (위경도, WGS84)

BUS_BUFFER_M   = 300.0    # 버스정류장 도보 커버 기준 거리 (m)
SUB_BUFFER_M   = 500.0    # 지하철역 도보 커버 기준 거리 (m)
GRAPH_BUFFER_M = 1500.0   # OSM 네트워크 다운로드 범위: 행정동 경계 바깥 여유 버퍼 (m)
EDGE_BUFFER_M  = 25.0     # isochrone 폴리곤 생성 시 도로 폭 보정 버퍼 (m)

MAP_HEIGHT_PX = 650       # 지도 위젯 표시 높이 (픽셀)

# ── "최종 TOP 격자 → 각 정류장/역" 최단경로 표시 제어 상수 ──
DRAW_ALL_ROUTES = True    # True: 최단경로 라인을 지도에 표시 / False: 표시 안 함
MAX_DRAW_ROUTES = 300     # 그릴 최대 경로 개수 (정류장이 많을 때 성능 안전장치)
ROUTE_WEIGHT    = 4       # 경로 라인 두께 (픽셀)
ROUTE_OPACITY   = 0.85    # 경로 라인 투명도 (0.0 완전투명 ~ 1.0 완전불투명)
ROUTE_COLOR     = "#111111"  # 경로 라인 색상 (거의 검정)

# =========================================================
# 2) 페이지 설정
# =========================================================
st.set_page_config(page_title="대중교통 커버리지 비교", layout="wide")  # 브라우저 탭 제목 + 와이드 레이아웃

st.markdown(                             # CSS 스타일 주입 (마진·폰트 조정)
    """
    <style>
      .block-container { padding: 1.2rem 1.0rem 1.6rem 1.0rem; max-width: none; }
      h1, h2, h3 { letter-spacing: -0.02em; }
      .stApp h1, div[data-testid="stMarkdownContainer"] h1 { text-align: center; width: 100%; }
      div[data-testid="stMarkdownContainer"] h1 { margin-top: 0.2rem; margin-bottom: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,              # HTML/CSS 태그 적용 허용
)

st.title("대중교통 커버리지 분석: 직선 버퍼 vs 네트워크 기반")  # 페이지 메인 제목
st.caption(                              # 부제목 설명 텍스트
    "버스 300 m / 지하철 500 m 기준 · "
    "'최종 TOP 격자'에서 각 버스정류장/지하철역까지 최단경로(라인) 표시"
)

# =========================================================
# 3) 행정동 선택 드롭다운
# =========================================================
st.markdown("---")                       # 수평 구분선

rid = st.selectbox(                      # 행정동 선택 드롭다운 위젯
    "행정동 선택",                        # 위젯 라벨
    options=list(TARGET_IDS.keys()),     # 선택 옵션: 행정동 코드 목록
    format_func=lambda x: f"{TARGET_IDS.get(x, x)} ({x})",  # 표시 형식: "이름 (코드)"
    index=0,                             # 기본 선택 인덱스 (0번째)
)
st.caption(f"선택 행정동: {TARGET_IDS.get(rid)}")  # 현재 선택된 행정동 이름 표시

# =========================================================
# 4) 데이터 로드 + 분석
# =========================================================

with st.spinner("데이터 로드/분석 중... (OSM 네트워크 다운로드 포함)"):  # 계산 중 스피너 표시

    # ─────────────────────────────────────────────────────
    # (1) 행정동 로드 및 선택
    # ─────────────────────────────────────────────────────
    gdf_admin = gpd.read_file(ADMIN_SHP)                               # 전체 행정동 경계 레이어 로드
    gdf_admin["region_id"] = gdf_admin["ADM_CD"].astype(str).str.strip()  # 행정동 코드 → 문자열 + 앞뒤 공백 제거
    gdf_admin["region_nm"] = gdf_admin["ADM_NM"].astype(str).str.strip()  # 행정동 이름 → 문자열 + 공백 제거
    gdf_admin = gdf_admin.to_crs(TARGET_CRS)                           # 분석 좌표계(5179)로 변환

    gdf_sel = gdf_admin[gdf_admin["region_id"] == rid].copy()         # 선택한 행정동만 필터링
    if len(gdf_sel) == 0:                                              # 일치하는 행정동이 없으면
        st.error("선택한 행정동을 찾을 수 없습니다.")                    # 오류 메시지 출력
        st.stop()                                                       # 실행 중단

    region_nm = gdf_sel["region_nm"].iloc[0]                          # 행정동 이름 문자열 추출
    sel_union = unary_union(gdf_sel.geometry)                         # 멀티폴리곤을 단일 폴리곤으로 합침

    sel_ll  = gdf_sel.to_crs(MAP_CRS)                                 # 지도 표시용(4326) GeoDataFrame
    bounds  = sel_ll.total_bounds                                      # 경계 좌표 배열 [minx, miny, maxx, maxy]
    center  = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]  # 지도 중심점 [위도, 경도]

    # ─────────────────────────────────────────────────────
    # (2) 버스정류장 로드 및 행정동 내부 필터
    # ─────────────────────────────────────────────────────
    bus_raw = pd.read_excel(BUS_XLSX)                                  # 버스정류장 엑셀 로드
    bus_raw["X좌표"] = pd.to_numeric(bus_raw["X좌표"], errors="coerce")  # X좌표(경도) → 숫자 (실패 시 NaN)
    bus_raw["Y좌표"] = pd.to_numeric(bus_raw["Y좌표"], errors="coerce")  # Y좌표(위도) → 숫자
    bus_raw = bus_raw.dropna(subset=["X좌표", "Y좌표"])               # 좌표 결측 행 제거

    gdf_bus = gpd.GeoDataFrame(                                        # 버스정류장 GeoDataFrame 생성
        bus_raw,
        geometry=gpd.points_from_xy(bus_raw["X좌표"], bus_raw["Y좌표"]),  # (경도, 위도) → Point 객체
        crs=MAP_CRS,                                                   # 입력 좌표계: WGS84(4326)
    ).to_crs(TARGET_CRS)                                               # 분석 좌표계(5179)로 변환

    gdf_bus_sel = gdf_bus[gdf_bus.geometry.within(sel_union)].copy()  # 행정동 경계 안에 있는 정류장만 추출

    # ─────────────────────────────────────────────────────
    # (3) 지하철역 로드 및 행정동 내부 필터
    # ─────────────────────────────────────────────────────
    try:
        sub_raw = pd.read_csv(SUBWAY_CSV, encoding="utf-8")            # UTF-8 인코딩으로 CSV 로드 시도
    except UnicodeDecodeError:
        sub_raw = pd.read_csv(SUBWAY_CSV, encoding="cp949")            # 실패 시 CP949(한글 윈도우) 재시도

    sub_raw["경도"] = pd.to_numeric(sub_raw["경도"], errors="coerce")  # 경도 → 숫자
    sub_raw["위도"] = pd.to_numeric(sub_raw["위도"], errors="coerce")  # 위도 → 숫자
    sub_raw = sub_raw.dropna(subset=["경도", "위도"])                  # 좌표 결측 제거

    gdf_sub = gpd.GeoDataFrame(                                        # 지하철역 GeoDataFrame 생성
        sub_raw,
        geometry=gpd.points_from_xy(sub_raw["경도"], sub_raw["위도"]),  # (경도, 위도) → Point 객체
        crs=MAP_CRS,                                                   # 입력 좌표계: WGS84(4326)
    ).to_crs(TARGET_CRS)                                               # 분석 좌표계로 변환

    gdf_sub_sel = gdf_sub[gdf_sub.geometry.within(sel_union)].copy()  # 행정동 내부 역만 추출

    # ─────────────────────────────────────────────────────
    # (4) 인구격자 로드 및 행정동으로 클립
    # ─────────────────────────────────────────────────────
    gdf_grid = gpd.read_file(GRID_SHP).to_crs(TARGET_CRS)             # 격자 쉐이프파일 로드 + 좌표계 변환
    gdf_grid["gid"] = gdf_grid["gid"].astype(str)                     # 격자 ID → 문자열

    # "val" 컬럼이 있으면 인구로 사용, 없으면 0으로 채움
    _pop_col = "val" if "val" in gdf_grid.columns else None           # 인구 원본 컬럼 이름 확인
    gdf_grid["pop"] = (                                                # 인구 컬럼 생성
        pd.to_numeric(gdf_grid[_pop_col], errors="coerce").fillna(0.0)  # 숫자 변환 + 결측→0
        if _pop_col else 0.0                                           # 컬럼 없으면 전부 0
    )

    gdf_grid_sel = gpd.clip(                                           # 행정동 경계로 격자 클립
        gdf_grid[gdf_grid.geometry.intersects(sel_union)],             # 교차 후보 먼저 추려 속도 향상
        gdf_sel,                                                        # 클립 기준 경계
    )[["gid", "pop", "geometry"]].copy()                               # 필요한 컬럼만 유지

    gdf_grid_sel["centroid_m"] = gdf_grid_sel.geometry.centroid       # 각 격자의 중심점 계산 (5179 좌표)

    # =========================================================
    # 5) (A) 직선 버퍼 커버리지 계산
    # =========================================================
    bufs = []                                                           # 버퍼 폴리곤 수집 목록

    if len(gdf_bus_sel) > 0:                                           # 행정동 내 버스정류장이 있으면
        bufs.append(unary_union(gdf_bus_sel.geometry.buffer(BUS_BUFFER_M)))  # 300m 원형 버퍼 합산

    if len(gdf_sub_sel) > 0:                                           # 행정동 내 지하철역이 있으면
        bufs.append(unary_union(gdf_sub_sel.geometry.buffer(SUB_BUFFER_M)))  # 500m 원형 버퍼 합산

    cover_buf = unary_union(bufs) if bufs else None                    # 커버 영역 = 버퍼들의 합집합
    uncov_buf = sel_union.difference(cover_buf) if cover_buf else sel_union  # 비커버 = 행정동 - 커버

    # =========================================================
    # 6) (B) 네트워크(Isochrone) 커버리지 계산
    # =========================================================
    poly_graph_ll = (                                                  # OSM 다운로드 범위 (행정동 + 여유 버퍼)
        gpd.GeoSeries([sel_union.buffer(GRAPH_BUFFER_M)], crs=TARGET_CRS)  # 5179 기준 여유 버퍼
        .to_crs(MAP_CRS)                                               # OSMnx 요구: 4326으로 변환
        .iloc[0]                                                        # GeoSeries → 단일 geometry
    )

    ox.settings.log_console = False                                    # OSMnx 콘솔 로그 비활성화
    G = ox.graph_from_polygon(poly_graph_ll, network_type="walk", simplify=True)  # 도보 네트워크 다운로드

    bus_ll = gdf_bus_sel.to_crs(MAP_CRS).copy()                       # 버스정류장 4326 변환 (nearest_nodes 용)
    sub_ll = gdf_sub_sel.to_crs(MAP_CRS).copy()                       # 지하철역 4326 변환

    bus_nodes = []                                                      # 버스정류장 최근접 노드 ID 목록
    if len(bus_ll) > 0:                                                # 버스정류장이 있으면
        bus_nodes = list(ox.distance.nearest_nodes(                    # 각 정류장에 가장 가까운 노드 찾기
            G, X=bus_ll.geometry.x.values, Y=bus_ll.geometry.y.values
        ))

    subway_nodes = []                                                   # 지하철역 최근접 노드 ID 목록
    if len(sub_ll) > 0:                                                # 지하철역이 있으면
        subway_nodes = list(ox.distance.nearest_nodes(
            G, X=sub_ll.geometry.x.values, Y=sub_ll.geometry.y.values
        ))

    # ── 버스 + 지하철 합치기 (isochrone 계산용) ──
    gdf_bus_sel2 = gdf_bus_sel.copy()                                  # 버스 복사본 (원본 수정 방지)
    gdf_bus_sel2["stop_type"] = "bus"                                  # 정류장 타입 레이블 부여

    gdf_sub_sel2 = gdf_sub_sel.copy()                                  # 지하철 복사본
    gdf_sub_sel2["stop_type"] = "subway"                               # 역 타입 레이블 부여

    gdf_stops = gpd.GeoDataFrame(                                      # 버스 + 지하철 통합 GeoDataFrame
        pd.concat([gdf_bus_sel2, gdf_sub_sel2], ignore_index=True),   # 두 데이터프레임 행 방향 결합
        geometry="geometry", crs=TARGET_CRS,                           # geometry 컬럼 및 좌표계 지정
    )

    gdf_stops_ll = gdf_stops.to_crs(MAP_CRS).copy()                   # 4326으로 변환 (nearest_nodes 용)

    if len(gdf_stops_ll) > 0:                                          # 정류장이 하나라도 있으면
        gdf_stops_ll["v_node"] = ox.distance.nearest_nodes(           # 각 정류장의 최근접 노드 ID 매핑
            G, X=gdf_stops_ll.geometry.x.values, Y=gdf_stops_ll.geometry.y.values
        )

    iso_polys = []                                                      # 정류장별 isochrone 폴리곤 수집 목록

    for _, r in gdf_stops_ll.iterrows():                               # 각 정류장/역 순회
        v = int(r["v_node"])                                            # 해당 정류장의 시작 노드 ID (정수 변환)
        iso_cutoff = BUS_BUFFER_M if r["stop_type"] == "bus" else SUB_BUFFER_M  # 타입별 반경

        try:
            Gsub_iso = nx.ego_graph(                                   # 반경 내 도달 가능한 서브 그래프 생성
                G, v, radius=float(iso_cutoff), distance="length", undirected=True
            )
        except Exception:                                              # 노드 없음 등 예외 발생 시
            continue                                                    # 다음 정류장으로 건너뜀

        if Gsub_iso.number_of_edges() == 0:                           # 도달 가능한 엣지가 없으면
            continue                                                    # 건너뜀

        _, gdf_edges = ox.graph_to_gdfs(                              # 서브 그래프 → 엣지 GeoDataFrame 변환
            Gsub_iso, nodes=True, edges=True, fill_edge_geometry=True  # fill_edge_geometry: 곡선 도로 형상 포함
        )

        poly_m = unary_union(                                          # 엣지에 도로폭 버퍼 적용 후 합집합 → isochrone 폴리곤
            gdf_edges.to_crs(TARGET_CRS).geometry.buffer(EDGE_BUFFER_M)
        )

        if poly_m is not None and not poly_m.is_empty:                # 유효한 폴리곤이면
            iso_polys.append(poly_m)                                   # 목록에 추가

    cover_iso = unary_union(iso_polys) if iso_polys else None          # isochrone 커버 = 모든 폴리곤 합집합
    uncov_iso = sel_union.difference(cover_iso) if cover_iso else sel_union  # isochrone 비커버 = 행정동 - 커버

    # =========================================================
    # 7) KPI + 비커버 최대 인구 격자 (TOP)
    # =========================================================
    admin_area = sel_union.area                                        # 행정동 전체 면적 (m²)

    # 각 격자 중심점이 비커버 영역 안에 있는지 판단 (불리언 마스크)
    buf_mask = (
        gdf_grid_sel["centroid_m"].within(uncov_buf)
        if (uncov_buf is not None and not uncov_buf.is_empty)
        else pd.Series(False, index=gdf_grid_sel.index)               # 비커버가 없으면 전부 False
    )
    iso_mask = (
        gdf_grid_sel["centroid_m"].within(uncov_iso)
        if (uncov_iso is not None and not uncov_iso.is_empty)
        else pd.Series(False, index=gdf_grid_sel.index)
    )

    buf_pop  = float(gdf_grid_sel.loc[buf_mask, "pop"].sum())         # 버퍼 비커버 내 총 인구
    iso_pop  = float(gdf_grid_sel.loc[iso_mask, "pop"].sum())         # isochrone 비커버 내 총 인구
    total_pop = float(gdf_grid_sel["pop"].sum())                      # 행정동 전체 인구

    buf_area = float(uncov_buf.area) if (uncov_buf and not uncov_buf.is_empty) else 0.0  # 버퍼 비커버 면적 (m²)
    iso_area = float(uncov_iso.area) if (uncov_iso and not uncov_iso.is_empty) else 0.0  # isochrone 비커버 면적

    false_covered  = (~buf_mask) & iso_mask                           # 버퍼로는 커버지만 네트워크로는 비커버인 격자
    additional_pop = float(gdf_grid_sel.loc[false_covered, "pop"].sum())  # 추가 발견 비커버 인구

    top_buf = None   # 버퍼 비커버 중 인구 최대 격자 (Series)
    top_iso = None   # 네트워크 비커버 중 인구 최대 격자

    if uncov_buf is not None and not uncov_buf.is_empty:              # 버퍼 비커버가 유효하면
        cands = gdf_grid_sel[gdf_grid_sel["centroid_m"].within(uncov_buf)]  # 비커버 내 격자 후보 추출
        if len(cands) > 0:                                             # 후보가 있으면
            top_buf = cands.loc[cands["pop"].idxmax()].copy()         # 인구 최대 격자 선택

    if uncov_iso is not None and not uncov_iso.is_empty:              # isochrone 비커버가 유효하면
        cands = gdf_grid_sel[gdf_grid_sel["centroid_m"].within(uncov_iso)]
        if len(cands) > 0:
            top_iso = cands.loc[cands["pop"].idxmax()].copy()

    # =========================================================
    # 8) TOP 격자 → 모든 정류장/역까지 최단경로 라인 계산
    #    - 버퍼 지도용: top_buf 에서 출발
    #    - 네트워크 지도용: top_iso 에서 출발
    # =========================================================

    def _build_routes(top_row):
        """
        top_row: 출발점이 될 격자 행 (Series, centroid_m 컬럼 포함)
        반환: LineString 목록 (지도에 그릴 최단경로들)
        """
        routes = []                                                    # 결과 경로 목록
        if top_row is None:                                            # 출발 격자가 없으면 빈 목록 반환
            return routes

        # 격자 중심점을 4326으로 변환 후 그래프 노드에 매핑
        cent_ll = (
            gpd.GeoSeries([top_row["centroid_m"]], crs=TARGET_CRS)
            .to_crs(MAP_CRS)
            .iloc[0]                                                   # 중심점 Point (4326)
        )
        src_node = ox.distance.nearest_nodes(                          # 출발 노드 ID
            G, X=float(cent_ll.x), Y=float(cent_ll.y)
        )

        # 도착 노드 목록: 버스 노드 + 지하철 노드 (순서 유지하며 중복 제거)
        targets = list(dict.fromkeys(list(bus_nodes) + list(subway_nodes)))
        if len(targets) > MAX_DRAW_ROUTES:                             # 경로가 너무 많으면
            targets = targets[:MAX_DRAW_ROUTES]                        # 앞부분만 사용 (성능 안전장치)

        for tn in targets:                                             # 각 도착 노드에 대해
            if tn == src_node:                                         # 출발 == 도착이면 경로 불필요
                continue

            try:
                path_nodes = nx.shortest_path(                         # Dijkstra 최단경로 노드 리스트
                    G, source=src_node, target=tn, weight="length"
                )
            except nx.NetworkXNoPath:                                  # 연결되지 않은 경우
                continue
            except Exception:                                          # 기타 예외
                continue

            if len(path_nodes) < 2:                                    # 노드가 1개 이하면 선 생성 불가
                continue

            # ── 경로를 LineString으로 변환 ──────────────────────────────
            # ※ ox.utils_graph.route_to_gdf()는 OSMnx v2.0+에서 제거됨
            #   엣지별 geometry를 직접 읽어 이어 붙이고, 없으면 노드 좌표 직선으로 대체
            try:
                edge_geoms = []                                        # 엣지 geometry 수집 목록
                for u, v in zip(path_nodes[:-1], path_nodes[1:]):     # 인접 노드 쌍(u→v) 순회
                    edge_dict = G.get_edge_data(u, v)                  # u→v 엣지 데이터 가져오기
                    if edge_dict is None:                              # 엣지 없으면 스킵
                        continue
                    # 멀티그래프: 같은 방향 엣지 여러 개 → 가장 짧은 것 선택
                    best = min(edge_dict.values(), key=lambda e: e.get("length", float("inf")))
                    geom = best.get("geometry")                        # 엣지 형상 (LineString or None)
                    if geom is not None and not geom.is_empty:         # 형상이 있으면 사용
                        edge_geoms.append(geom)
                    else:                                              # 없으면 노드 좌표로 직선 생성
                        n1, n2 = G.nodes[u], G.nodes[v]               # 시작/끝 노드 속성
                        edge_geoms.append(LineString([(n1["x"], n1["y"]), (n2["x"], n2["y"])]))

                if not edge_geoms:                                     # 형상 없으면 건너뜀
                    continue

                line = unary_union(edge_geoms)                         # 모든 엣지 형상 합치기

            except Exception:                                          # 예외 시 노드 좌표 직선으로 폴백
                try:
                    coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in path_nodes]
                    line = LineString(coords)                           # 직선 연결 LineString
                except Exception:
                    continue                                            # 그래도 실패하면 건너뜀

            if line is None or line.is_empty:                          # 빈 geometry면 건너뜀
                continue

            # geometry 타입별 저장
            if line.geom_type == "LineString":                         # 단일 선
                routes.append(line)
            elif line.geom_type == "MultiLineString":                  # 복합 선: 모든 파트 저장
                for part in line.geoms:
                    if part is not None and not part.is_empty and len(part.coords) >= 2:
                        routes.append(part)

        return routes                                                  # 완성된 경로 목록 반환

    # 버퍼 지도용: 버퍼 TOP 격자 → 각 정류장/역 최단경로
    buf_routes = _build_routes(top_buf) if DRAW_ALL_ROUTES else []    # DRAW_ALL_ROUTES=False면 빈 목록

    # 네트워크 지도용: 네트워크(iso) TOP 격자 → 각 정류장/역 최단경로
    iso_routes = _build_routes(top_iso) if DRAW_ALL_ROUTES else []    # top_iso 없으면 빈 목록 반환

    # =========================================================
    # 9) 지도 표시용(4326) geometry 변환
    # =========================================================
    cover_buf_ll = None    # 버퍼 커버 영역 (4326)
    uncov_buf_ll = None    # 버퍼 비커버 영역 (4326)
    cover_iso_ll = None    # isochrone 커버 영역 (4326)
    uncov_iso_ll = None    # isochrone 비커버 영역 (4326)

    if cover_buf is not None:                                          # 버퍼 커버가 있으면
        cover_buf_ll = (
            gpd.GeoSeries([cover_buf.intersection(sel_union)], crs=TARGET_CRS)  # 행정동 경계로 clip
            .to_crs(MAP_CRS).iloc[0]                                   # 4326 변환 → 단일 geometry
        )

    if uncov_buf is not None and not uncov_buf.is_empty:              # 버퍼 비커버가 있으면
        uncov_buf_ll = (
            gpd.GeoSeries([uncov_buf], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
        )

    if cover_iso is not None:                                          # isochrone 커버가 있으면
        cover_iso_ll = (
            gpd.GeoSeries([cover_iso.intersection(sel_union).simplify(5)], crs=TARGET_CRS)  # 단순화 + clip
            .to_crs(MAP_CRS).iloc[0]
        )

    if uncov_iso is not None and not uncov_iso.is_empty:              # isochrone 비커버가 있으면
        uncov_iso_ll = (
            gpd.GeoSeries([uncov_iso.simplify(5)], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]
        )

    top_buf_ll = None    # 버퍼 TOP 격자 (4326 GeoDataFrame)
    top_iso_ll = None    # 네트워크 TOP 격자 (4326)

    if top_buf is not None:                                            # 버퍼 TOP이 있으면
        top_buf_ll = gpd.GeoDataFrame([top_buf], geometry="geometry", crs=TARGET_CRS).to_crs(MAP_CRS)

    if top_iso is not None:                                            # 네트워크 TOP이 있으면
        top_iso_ll = gpd.GeoDataFrame([top_iso], geometry="geometry", crs=TARGET_CRS).to_crs(MAP_CRS)

    kpi = dict(                                                        # KPI 값 딕셔너리 생성
        region_nm      = region_nm,                                    # 행정동 이름
        buf_uncov_km2  = buf_area / 1e6,                               # 버퍼 비커버 면적 (m² → km²)
        iso_uncov_km2  = iso_area / 1e6,                               # isochrone 비커버 면적 (km²)
        buf_uncov_pop  = buf_pop,                                      # 버퍼 비커버 인구 (명)
        iso_uncov_pop  = iso_pop,                                      # isochrone 비커버 인구
        buf_ratio      = buf_area / admin_area if admin_area > 0 else 0,  # 버퍼 비커버 면적 비율
        iso_ratio      = iso_area / admin_area if admin_area > 0 else 0,  # isochrone 비커버 비율
        additional_pop = additional_pop,                               # 추가 발견 비커버 인구
        total_pop      = total_pop,                                    # 행정동 전체 인구
        n_buf_routes   = len(buf_routes),                              # 버퍼 지도 경로 개수
        n_iso_routes   = len(iso_routes),                              # 네트워크 지도 경로 개수
    )

# =========================================================
# 10) KPI 출력
# =========================================================
st.markdown("---")                                                     # 수평 구분선
st.subheader(f"KPI 비교 ({kpi['region_nm']})")                         # KPI 섹션 제목

c1, c2, c3, c4 = st.columns(4)                                        # 4등분 컬럼 레이아웃

with c1:                                                               # 첫 번째 KPI: 비커버 면적
    st.metric(
        label="비커버 면적(네트워크)",                                   # 지표 이름
        value=f"{kpi['iso_uncov_km2']:.3f} km²",                       # 현재 값
        delta=f"{kpi['iso_uncov_km2'] - kpi['buf_uncov_km2']:+.3f} km² (네트워크 − 버퍼)",  # 차이 값
        delta_color="inverse",                                         # 증가가 빨간색 (나쁜 방향)
    )

with c2:                                                               # 두 번째 KPI: 비커버 인구
    st.metric(
        label="비커버 인구(네트워크)",
        value=f"{kpi['iso_uncov_pop']:,.0f} 명",
        delta=f"{kpi['iso_uncov_pop'] - kpi['buf_uncov_pop']:+,.0f} 명",
        delta_color="inverse",
    )

with c3:                                                               # 세 번째 KPI: 비커버 비율
    st.metric(
        label="비커버 비율(네트워크)",
        value=f"{kpi['iso_ratio']:.1%}",
        delta=f"{(kpi['iso_ratio'] - kpi['buf_ratio']) * 100:+.1f} %p",  # 백분율포인트 차이
        delta_color="inverse",
    )

with c4:                                                               # 네 번째 KPI: 추가 발견 인구
    st.metric(
        label="추가 발견 비커버 인구",
        value=f"{kpi['additional_pop']:,.0f} 명",
        help="직선 버퍼로는 커버된 것처럼 보이지만, 실제 도보 네트워크로는 도달 불가한 인구",  # 마우스오버 도움말
    )

st.caption(                                                            # 보조 설명 텍스트
    f"버퍼 TOP 기준 경로: {kpi['n_buf_routes']}개 | "
    f"네트워크 TOP 기준 경로: {kpi['n_iso_routes']}개"
)

# =========================================================
# 11) 지도 생성 함수
# =========================================================

def _number_badge(n, bg):
    """숫자 배지 DivIcon HTML 생성 (n: 표시할 숫자, bg: 배경색)"""
    return f"""
    <div style="
      width:28px; height:28px; border-radius:50%;
      background:{bg}; color:#fff; font-weight:800; font-size:14px;
      display:flex; align-items:center; justify-content:center;
      border:2px solid #fff; box-shadow:0 2px 8px rgba(0,0,0,0.35);
    ">{n}</div>
    """                                                                # 원형 숫자 배지 HTML 반환


def _bus_icon_html(name=""):
    """버스정류장 DivIcon HTML — 파란 원형 배지 + 🚌 이모지"""
    label = name[:6] if name else "버스"                              # 정류장명 최대 6자 (오버플로 방지)
    return f"""
    <div style="
      display:flex; flex-direction:column; align-items:center;
      filter: drop-shadow(0 2px 4px rgba(0,0,0,0.45));
    ">
      <div style="
        width:34px; height:34px; border-radius:50%;
        background:linear-gradient(145deg,#2979ff,#0047cc);
        color:#fff; font-size:18px;
        display:flex; align-items:center; justify-content:center;
        border:2.5px solid #fff;
      ">🚌</div>
      <div style="
        margin-top:3px; padding:1px 5px; font-size:9.5px; font-weight:700;
        background:rgba(41,121,255,0.92); color:#fff;
        border-radius:3px; white-space:nowrap; max-width:72px;
        overflow:hidden; text-overflow:ellipsis;
      ">{label}</div>
    </div>"""                                                          # 원형 아이콘 + 이름 라벨 HTML 반환


def _sub_icon_html(name=""):
    """지하철역 DivIcon HTML — 주황 원형 배지 + 🚇 이모지"""
    label = name[:6] if name else "지하철"                            # 역명 최대 6자
    return f"""
    <div style="
      display:flex; flex-direction:column; align-items:center;
      filter: drop-shadow(0 2px 5px rgba(0,0,0,0.5));
    ">
      <div style="
        width:38px; height:38px; border-radius:50%;
        background:linear-gradient(145deg,#ff7043,#e64a19);
        color:#fff; font-size:20px;
        display:flex; align-items:center; justify-content:center;
        border:2.5px solid #fff;
      ">🚇</div>
      <div style="
        margin-top:3px; padding:1px 5px; font-size:9.5px; font-weight:700;
        background:rgba(230,74,25,0.92); color:#fff;
        border-radius:3px; white-space:nowrap; max-width:76px;
        overflow:hidden; text-overflow:ellipsis;
      ">{label}</div>
    </div>"""                                                          # 원형 아이콘 + 역명 라벨 HTML 반환


def _add_base_layers(m):
    """지도에 공통 레이어 추가: 행정동 경계, 버스정류장 마커, 지하철역 마커"""

    folium.GeoJson(                                                    # 행정동 경계 레이어
        sel_ll,                                                        # 4326 GeoDataFrame
        name="행정동 경계",
        style_function=lambda x: {"fillOpacity": 0.03, "color": "#444", "weight": 3},  # 반투명 채움, 진한 테두리
        tooltip=folium.GeoJsonTooltip(fields=["region_nm"], aliases=["행정동"]),  # 마우스오버 툴팁
    ).add_to(m)

    for _, r in bus_ll.iterrows():                                     # 버스정류장 마커 순회
        stop_name = str(r.get("정류소명", ""))                         # 정류장명 문자열 추출
        folium.Marker(                                                 # 마커 생성
            location=[r.geometry.y, r.geometry.x],                    # [위도, 경도]
            tooltip=f"🚌 버스정류장 | {stop_name}",                   # 마우스오버 툴팁
            icon=folium.DivIcon(                                       # 커스텀 HTML 아이콘
                html=_bus_icon_html(stop_name),                        # 버스 아이콘 HTML
                icon_size=(80, 55),                                    # 아이콘 크기 (가로, 세로)
                icon_anchor=(40, 55),                                  # 기준점: 아이콘 하단 중앙
            ),
        ).add_to(m)

    for _, r in sub_ll.iterrows():                                     # 지하철역 마커 순회
        # 역명 컬럼 탐색 (CSV 컬럼명이 다를 수 있으므로 후보 순서대로 시도)
        sta_name = str(
            r.get("역명") or r.get("station_nm") or r.get("역사명") or ""
        )
        folium.Marker(                                                 # 마커 생성
            location=[r.geometry.y, r.geometry.x],                    # [위도, 경도]
            tooltip=f"🚇 지하철역 | {sta_name}",                      # 마우스오버 툴팁
            icon=folium.DivIcon(                                       # 커스텀 HTML 아이콘
                html=_sub_icon_html(sta_name),                         # 지하철 아이콘 HTML
                icon_size=(84, 60),                                    # 아이콘 크기
                icon_anchor=(42, 60),                                  # 기준점: 아이콘 하단 중앙
            ),
        ).add_to(m)


def _add_top_grid(m, top_ll, poly_color, label):
    """비커버 최대 인구 격자 폴리곤 + 배지 마커를 지도에 추가"""
    if top_ll is None or len(top_ll) == 0:                            # TOP 격자가 없으면 종료
        return

    r   = top_ll.iloc[0]                                              # 단일 격자 행 추출
    pop = float(r.get("pop", 0))                                      # 인구 값 (float 변환)
    gid = r.get("gid", "")                                            # 격자 ID
    tip = f"{label} | gid={gid} | pop={pop:,.0f}"                     # 툴팁 문자열

    folium.GeoJson(                                                    # 격자 폴리곤 강조 표시
        {"type": "Feature", "properties": {}, "geometry": mapping(r.geometry)},  # GeoJSON 형식
        name=f"{label} TOP 격자",
        style_function=lambda x, c=poly_color: {                      # c=poly_color로 클로저 캡처
            "fillOpacity": 0.50, "fillColor": c, "color": c, "weight": 3,
        },
        tooltip=tip,
    ).add_to(m)

    c = r.geometry.centroid                                            # 격자 중심점 좌표
    folium.Marker(                                                     # 숫자 배지 마커
        location=[c.y, c.x],                                          # [위도, 경도]
        tooltip=tip,
        icon=folium.DivIcon(html=_number_badge(1, poly_color)),       # 커스텀 HTML 아이콘
    ).add_to(m)


def _add_routes(m, routes, name="최종 격자→정류장/역 최단경로"):
    """최단경로 라인 목록을 지도에 추가 (레이어 그룹으로 묶음)"""
    if not routes:                                                     # 경로 목록이 비어있으면 종료
        return

    fg = folium.FeatureGroup(name=name, show=True)                    # 레이어 컨트롤에 표시될 그룹

    for ls in routes:                                                  # 각 경로 LineString 순회
        if ls is None or ls.is_empty:                                  # 비어있으면 건너뜀
            continue
        coords = list(ls.coords)                                       # 좌표 목록 [(lon, lat), ...]
        if len(coords) < 2:                                            # 좌표가 2개 미만이면 선 불가
            continue

        folium.PolyLine(                                               # 경로 라인 추가
            [(lat, lon) for lon, lat in coords],                       # Folium 요구 형식: [(위도, 경도), ...]
            weight=ROUTE_WEIGHT,                                       # 라인 두께
            opacity=ROUTE_OPACITY,                                     # 투명도
            color=ROUTE_COLOR,                                         # 색상
        ).add_to(fg)

    fg.add_to(m)                                                       # 완성된 그룹을 지도에 추가


# =========================================================
# 12) 직선 버퍼 지도 생성
# =========================================================
m_buf = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")  # 지도 객체 생성

_add_base_layers(m_buf)                                                # 공통 레이어 추가

if cover_buf_ll is not None and not cover_buf_ll.is_empty:            # 버퍼 커버가 있으면
    folium.GeoJson(
        mapping(cover_buf_ll), name="커버(버퍼)",
        style_function=lambda x: {"fillOpacity": 0.22, "fillColor": "#28a745", "color": "#28a745", "weight": 1},  # 녹색 반투명
    ).add_to(m_buf)

if uncov_buf_ll is not None and not uncov_buf_ll.is_empty:            # 버퍼 비커버가 있으면
    folium.GeoJson(
        mapping(uncov_buf_ll), name="비커버(버퍼)",
        style_function=lambda x: {"fillOpacity": 0.32, "fillColor": "#cc0000", "color": "#cc0000", "weight": 2},  # 빨간색
    ).add_to(m_buf)

_add_top_grid(m_buf, top_buf_ll,   poly_color="#ff6600", label="버퍼 비커버 최대인구")  # 버퍼 TOP 격자

# =========================================================
# 13) 네트워크(Isochrone) 지도 생성
# =========================================================
m_iso = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")  # 지도 객체 생성

_add_base_layers(m_iso)                                                # 공통 레이어 추가

if cover_iso_ll is not None and not cover_iso_ll.is_empty:            # isochrone 커버가 있으면
    folium.GeoJson(
        mapping(cover_iso_ll), name="커버(Isochrone)",
        style_function=lambda x: {"fillOpacity": 0.18, "fillColor": "#0066ff", "color": "#0066ff", "weight": 1},  # 파란색
    ).add_to(m_iso)

if uncov_iso_ll is not None and not uncov_iso_ll.is_empty:            # isochrone 비커버가 있으면
    folium.GeoJson(
        mapping(uncov_iso_ll), name="비커버(Isochrone)",
        style_function=lambda x: {"fillOpacity": 0.28, "fillColor": "#7a00cc", "color": "#7a00cc", "weight": 2},  # 보라색
    ).add_to(m_iso)

_add_top_grid(m_iso, top_iso_ll, poly_color="#e91e63", label="네트워크 비커버 최대인구")  # 네트워크 TOP 격자

# =========================================================
# 14) 각 지도에 TOP 격자 + 경로 표시 (지도별 독립적으로)
# =========================================================

# 버퍼 지도: 버퍼 TOP 격자 + 버퍼 TOP 기준 경로
# (최종 TOP 격자는 표시하지 않음 — 버퍼 분석 결과만 표시)
if DRAW_ALL_ROUTES:                                                    # 경로 표시 ON이면
    _add_routes(m_buf, buf_routes, name="버퍼 TOP→정류장/역 최단경로")  # 버퍼 지도에 버퍼 TOP 경로 추가

# 네트워크 지도: 네트워크 TOP 격자 + 네트워크 TOP 기준 경로
if DRAW_ALL_ROUTES:                                                    # 경로 표시 ON이면
    _add_routes(m_iso, iso_routes, name="네트워크 TOP→정류장/역 최단경로")  # 네트워크 지도에 iso TOP 경로 추가

folium.LayerControl(collapsed=False).add_to(m_buf)                    # 레이어 컨트롤 패널 추가
folium.LayerControl(collapsed=False).add_to(m_iso)

m_buf.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])    # 행정동 경계에 맞게 줌 설정
m_iso.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

# =========================================================
# 15) 화면 배치 (두 지도 나란히)
# =========================================================
st.markdown("---")                                                     # 구분선
col_l, col_r = st.columns(2, gap="large")                             # 좌/우 2열 레이아웃

with col_l:                                                            # 왼쪽: 직선 버퍼 분석
    st.subheader("직선 버퍼 기반 분석")
    st_folium(m_buf, width=None, height=MAP_HEIGHT_PX, key="map_buf", returned_objects=[])  # 지도 렌더링

with col_r:                                                            # 오른쪽: 네트워크 분석
    st.subheader("네트워크(Isochrone) 기반 분석")
    st_folium(m_iso, width=None, height=MAP_HEIGHT_PX, key="map_iso", returned_objects=[])

# =========================================================
# 16) 방법론 비교 확장 패널
# =========================================================
with st.expander("분석 방법론 비교"):                                  # 클릭하면 펼쳐지는 패널
    st.markdown(
        """
| 항목 | 직선 버퍼 | 네트워크 기반 (Isochrone) |
|------|-----------|--------------------------|
| **방식** | 정류장 중심 원형 버퍼 (300 m / 500 m) | OSMnx 도보 네트워크 ego_graph + 도로폭 25 m 버퍼 |
| **장점** | 계산 빠름, 직관적 | 실제 보행 경로 반영, 경로 복원 가능 |
| **단점** | 건물·하천·도로 등 장애물 미반영 | OSM 다운로드 필요, 계산 시간 소요 |
| **비커버 판단** | 원 바깥 = 비커버 | 도보 네트워크로 도달 불가 = 비커버 |
| **최단경로 표시** | (공통) 최종 TOP 격자 → 각 정류장/역 최단경로 라인 | (공통) 동일 |
        """
    )
