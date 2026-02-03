import os  # 파일 경로/환경변수 사용
import warnings  # 경고 메시지 제어
warnings.filterwarnings("ignore")  # 경고 메시지 숨김

import numpy as np  # 수치 계산 라이브러리
import pandas as pd  # 데이터프레임 처리 라이브러리
import geopandas as gpd  # 공간 데이터프레임 처리 라이브러리

import streamlit as st  # Streamlit 웹앱 프레임워크
import pydeck as pdk  # pydeck(WebGL 지도) 라이브러리

import folium  # folium(Leaflet 지도) 라이브러리
from streamlit_folium import st_folium  # Streamlit에서 folium 지도 출력

import osmnx as ox  # OpenStreetMap 네트워크 다운로드/가공
import networkx as nx  # 최단경로/다익스트라 계산

from shapely.geometry import Point, mapping  # Point 생성 + GeoJSON 변환(mapping)

# =========================================================  
# 0) PATHS / CONSTANTS  # 섹션 설명 주석
# =========================================================  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 app.py 파일의 절대경로 기준 폴더
DATA_DIR = os.path.join(BASE_DIR, "data")  # data 폴더 경로 구성

GRID_SHP = os.path.join(DATA_DIR, "nlsp_021001001.shp")  # 전수 격자(shp) 경로
UNCOVERED_GPKG = os.path.join(DATA_DIR, "demo_uncovered.gpkg")  # 비커버 폴리곤(gpkg) 경로(없어도 동작)
ADMIN_GPKG = os.path.join(DATA_DIR, "demo_admin.gpkg")  # 행정동 경계(gpkg) 경로(남현동 포함)

GRID_ID_COL = "gid"  # 격자 ID 컬럼명
GRID_POP_COL = "val"  # 격자 인구 컬럼명(없으면 pop=0 처리)

TARGET_CRS = 5179  # 분석용 CRS(미터 기반, 면적/거리 계산 정확)
MAP_CRS = 4326  # 지도 표출용 CRS(위경도)

KPI_RADIUS_M = 1250  # KPI 반경(미터)
NEW_STATION_BUFFER_M = 1250  # 신규 정류장 커버 반경(미터)

WALK_SPEED_MPS = 1.4  # 보행 속도(m/s)
CUTOFF_MIN = 5  # 네트워크 컷오프 시간(분)
CUTOFF_SEC = CUTOFF_MIN * 60  # 네트워크 컷오프 시간(초)

GRAPH_DIST_M = 3500  # OSM 그래프 다운로드 반경(미터)

CARTO_POSITRON_GL = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"  # 토큰 없이도 되는 basemap

MAP_HEIGHT_PX = 720  # ✅ 좌/우 지도 높이를 똑같이 맞추기 위한 고정값(px)

MAPBOX_TOKEN = os.getenv("MAPBOX_API_KEY") or os.getenv("MAPBOX_TOKEN")  # Mapbox 토큰(있으면 사용)
if MAPBOX_TOKEN:  # 토큰이 존재하면
    pdk.settings.mapbox_api_key = MAPBOX_TOKEN  # pydeck에 토큰 설정

# =========================================================  
# 1) Streamlit Page / UI Theme  # 섹션 설명 주석
# =========================================================  
st.set_page_config(page_title="따릉이 신규 정류소 배치를 통한 기대효과 대시보드 - 남현동", layout="wide")  # Streamlit 페이지 설정(가로 폭 넓게)

st.markdown(  # CSS를 주입하기 위한 markdown 호출
    """ 
    <style>
      /* 화면 풀사이즈 채우기: 폭 제한 제거 + 패딩 최소화 */  /* 설명 주석 */
      .block-container {  /* Streamlit 메인 컨테이너 */
        padding-top: 0.8rem;  /* 위쪽 여백 */
        padding-bottom: 1.6rem;  /* 아래쪽 여백 */
        padding-left: 1.0rem;  /* 좌측 여백 */
        padding-right: 1.0rem;  /* 우측 여백 */
        max-width: none;  /* 폭 제한 제거(기본 max-width 해제) */
      }  /* block-container 끝 */

      /* 타이틀 자간 조금 줄여서 보기 좋게 */  /* 설명 주석 */
      h1, h2, h3 { letter-spacing: -0.02em; }  /* 타이틀 자간 */

      /* ✅ 메인 타이틀 가운데 정렬(Streamlit DOM에 맞춘 안전한 셀렉터) */  /* 설명 주석 */
      .stApp h1,  /* 구버전/일부 테마 fallback */
      div[data-testid="stMarkdownContainer"] h1,  /* st.title이 들어가는 컨테이너 */
      .stTitle {  /* 일부 버전에서 타이틀 클래스가 따로 붙음 */
        text-align: center;  /* 타이틀 중앙 정렬 */
        width: 100%;  /* 중앙 정렬 안정화 */
      }  /* 타이틀 중앙 정렬 끝 */

      /* ✅ 제목 위/아래 여백 과하면 줄이기(스크린샷처럼 떠 보일 때) */  /* 설명 주석 */
      div[data-testid="stMarkdownContainer"] h1 {
        margin-top: 0.2rem;  /* 위 여백 */
        margin-bottom: 0.8rem;  /* 아래 여백 */
      }  /* 여백 조정 끝 */

      /* 캡션 색상 */  /* 설명 주석 */
      .stCaption { color: #666; }  /* 캡션 색 */

      /* KPI 영역 카드 스타일 */  /* 설명 주석 */
      .kpi-wrap {  /* KPI 카드 */
        padding: 14px 14px 2px 14px;  /* 내부 여백 */
        border: 1px solid #eee;  /* 테두리 */
        border-radius: 14px;  /* 라운드 */
        background: #fafafa;  /* 배경색 */
      }  /* kpi-wrap 끝 */

      /* 지도 카드 스타일 */  /* 설명 주석 */
      .soft-card {  /* 지도 카드 */
        padding: 14px;  /* 내부 여백 */
        border: 1px solid #eee;  /* 테두리 */
        border-radius: 14px;  /* 라운드 */
        background: white;  /* 배경 */
      }  /* soft-card 끝 */

      /* 작은 안내문 */  /* 설명 주석 */
      .small-muted { color:#777; font-size: 0.92rem; }  /* 글자색/크기 */

      /* 구분선 */  /* 설명 주석 */
      .hr { height: 1px; background: #eee; margin: 14px 0; }  /* 구분선 스타일 */
    </style>
    """,  # 문자열 종료
    unsafe_allow_html=True,  # HTML/CSS 허용
)  # st.markdown 종료

st.title("따릉이 신규 정류소 배치를 통한 기대효과 대시보드 - 남현동")  # 페이지 제목 출력

# =========================================================  
# 2) OSM 그래프(캐시)  # 섹션 설명 주석
# - 네트워크 다운로드는 비용이 크므로 캐시 필요  # 설명 주석
# =========================================================  

@st.cache_resource(show_spinner=False)  # 리소스 캐시(그래프 재사용)
def _build_osm_graph_from_point(lat: float, lon: float, dist_m: int, network_type: str = "walk"):  # 그래프 생성 함수
    ox.settings.log_console = False  # OSMnx 로그 비활성화
    G = ox.graph_from_point(  # 중심점 기준으로 그래프 다운로드
        (float(lat), float(lon)),  # (lat, lon)
        dist=int(dist_m),  # 반경(m)
        network_type=network_type,  # walk 네트워크
        simplify=True,  # 단순화
    )  # graph_from_point 종료
    try:  # OSMnx 2.x 호환 시도
        G = ox.distance.add_edge_lengths(G)  # edge length 추가(2.x)
    except Exception:  # 실패하면
        try:  # 1.x 호환 시도
            G = ox.add_edge_lengths(G)  # edge length 추가(1.x)
        except Exception:  # 그래도 실패하면
            pass  # 그냥 통과(이미 length가 있을 수도 있음)
    return G  # 그래프 반환

# =========================================================  
# 3) DATA LOAD (스크립트형)  # 섹션 설명 주석
# =========================================================  

if not os.path.exists(GRID_SHP):  # 격자 파일 존재 확인
    st.error(f"GRID_SHP not found: {GRID_SHP}")  # 에러 표시
    st.stop()  # 앱 중단

if not os.path.exists(ADMIN_GPKG):  # 행정동 파일 존재 확인
    st.error("남현동 행정구역 파일이 필요합니다. data/demo_admin.gpkg 를 넣어주세요.")  # 에러 표시
    st.stop()  # 앱 중단

with st.spinner("격자 로딩 중..."):  # 로딩 스피너
    gdf_grid = gpd.read_file(GRID_SHP)  # 격자 shapefile 로드

if gdf_grid.crs is None:  # CRS가 없으면
    st.error("GRID_SHP CRS is None. (.prj 확인)")  # 에러 표시
    st.stop()  # 앱 중단

gdf_grid = gdf_grid.to_crs(TARGET_CRS)  # 분석 CRS(5179)로 변환

if GRID_ID_COL not in gdf_grid.columns:  # gid 컬럼이 없으면
    st.error(f"GRID_ID_COL='{GRID_ID_COL}' not found in grid")  # 에러 표시
    st.stop()  # 앱 중단

gdf_grid[GRID_ID_COL] = gdf_grid[GRID_ID_COL].astype(str)  # gid를 문자열로 통일

if GRID_POP_COL in gdf_grid.columns:  # val 컬럼이 있으면
    gdf_grid["pop"] = pd.to_numeric(gdf_grid[GRID_POP_COL], errors="coerce").fillna(0).astype(float)  # pop 생성
elif "pop" in gdf_grid.columns:  # pop 컬럼이 이미 있으면
    gdf_grid["pop"] = pd.to_numeric(gdf_grid["pop"], errors="coerce").fillna(0).astype(float)  # pop 정리
else:  # 인구 컬럼이 없다면
    gdf_grid["pop"] = 0.0  # pop을 0으로

gdf_grid["geometry"] = gdf_grid.geometry.buffer(0)  # geometry 정리(자기교차 완화)
gdf_grid = gdf_grid[[GRID_ID_COL, "pop", "geometry"]].copy()  # 필요한 컬럼만 남김

with st.spinner("행정동(남현동) 로딩/선택 중..."):  # 로딩 스피너
    gdf_admin = gpd.read_file(ADMIN_GPKG)  # 행정동 gpkg 로드

if gdf_admin.crs is None:  # CRS가 없으면
    st.error("ADMIN CRS is None.")  # 에러 표시
    st.stop()  # 앱 중단

gdf_admin = gdf_admin.to_crs(TARGET_CRS)  # 분석 CRS(5179)로 변환
gdf_admin["geometry"] = gdf_admin.geometry.buffer(0)  # geometry 정리

NAME_COL_CANDIDATES = [  # 남현동명 컬럼 후보들
    "ADM_NM", "adm_nm", "ADMNM",  # 후보 1
    "region_nm", "REGION_NM",  # 후보 2
    "emd_nm", "EMD_NM",  # 후보 3
    "dong_nm", "DONG_NM",  # 후보 4
    "법정동명", "행정동명",  # 후보 5
]  # 후보 리스트 끝

name_col = None  # 사용할 컬럼명 초기화
for c in NAME_COL_CANDIDATES:  # 후보를 순서대로 확인
    if c in gdf_admin.columns:  # 존재하면
        name_col = c  # 그 컬럼 사용
        break  # 루프 종료

if name_col is None:  # 이름 컬럼을 못 찾으면
    gdf_namhyeon = gdf_admin.iloc[[0]].copy()  # 최소 동작: 첫 행을 남현동으로 간주
else:  # 이름 컬럼이 있으면
    s = gdf_admin[name_col].astype(str)  # 문자열로 변환
    mask = s.str.contains("남현", na=False)  # "남현" 포함 여부
    if mask.sum() == 0:  # 없으면
        mask = s.str.contains("남현동", na=False)  # "남현동"로 재시도
    gdf_namhyeon = gdf_admin.loc[mask].copy() if mask.sum() > 0 else gdf_admin.iloc[[0]].copy()  # 필터링 또는 fallback

with st.spinner("남현동 격자만 clip 중..."):  # 로딩 스피너
    nam_union = gdf_namhyeon.geometry.union_all()  # 남현동 폴리곤 union
    gdf_sub = gdf_grid[gdf_grid.geometry.intersects(nam_union)].copy()  # 교차 격자만 1차 필터
    gdf_grid_nam = gpd.clip(gdf_sub, gdf_namhyeon).copy()  # 실제 clip 수행
    gdf_grid_nam["geometry"] = gdf_grid_nam.geometry.buffer(0)  # geometry 정리

if len(gdf_grid_nam) == 0:  # 결과가 0이면
    st.error("남현동으로 clip된 격자가 0개입니다. 행정구역 파일/CRS/남현동 명칭 컬럼을 확인하세요.")  # 에러
    st.stop()  # 중단

if os.path.exists(UNCOVERED_GPKG):  # 비커버 파일이 있으면
    gdf_unc = gpd.read_file(UNCOVERED_GPKG)  # 로드
    if gdf_unc.crs is None:  # CRS 확인
        st.error("UNCOVERED_GPKG CRS is None.")  # 에러
        st.stop()  # 중단
    gdf_unc = gdf_unc.to_crs(TARGET_CRS)  # CRS 변환
    gdf_unc["geometry"] = gdf_unc.geometry.buffer(0)  # geometry 정리
    gdf_unc = gdf_unc[["geometry"]].copy()  # geometry만 유지
else:  # 비커버 파일이 없으면
    gdf_unc = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=TARGET_CRS)  # 빈 GDF 생성

if len(gdf_unc) == 0:  # 비커버가 없으면
    gdf_grid_nam["is_uncovered"] = False  # 전부 커버 처리
else:  # 비커버가 있으면
    unc_union = gdf_unc.geometry.union_all()  # 비커버 union
    gdf_grid_nam["is_uncovered"] = gdf_grid_nam.geometry.intersects(unc_union)  # 교차 여부로 비커버 플래그 생성

all_gids = gdf_grid_nam[GRID_ID_COL].astype(str).tolist()  # gid 목록 추출

# =========================================================  
# 4) KPI + gid 선택 + 신규 KPI(면적 km² / 인구 추정) 
# =========================================================  

st.subheader("KPI")  # KPI 제목
st.caption("gid를 선택하면 KPI와 좌/우 지도가 동시에 갱신됩니다.")  # 안내문

sel_gid = st.selectbox("남현동 격자 gid 선택", options=all_gids, index=0, key="gid_select")  # 선택박스

row = gdf_grid_nam.loc[gdf_grid_nam[GRID_ID_COL] == str(sel_gid)]  # 선택 gid 행 찾기
if len(row) == 0:  # 없으면
    st.error("선택 gid를 남현동 격자에서 찾지 못했습니다.")  # 에러
    st.stop()  # 중단

sel_poly = row.geometry.iloc[0]  # 선택 격자 폴리곤
sel_center_5179 = sel_poly.centroid  # 선택 격자 중심점(5179)

kpi_circle_5179 = sel_center_5179.buffer(float(KPI_RADIUS_M))  # KPI 반경 원
station_buffer_5179 = sel_center_5179.buffer(float(NEW_STATION_BUFFER_M))  # 신규 커버 반경 원

in_circle = gdf_grid_nam.geometry.intersects(kpi_circle_5179)  # KPI 원과 교차하는 격자
gdf_in = gdf_grid_nam.loc[in_circle, [GRID_ID_COL, "pop", "is_uncovered", "geometry"]].copy()  # KPI 원 내부 격자

total_pop = float(gdf_in["pop"].sum())  # KPI 원 내부 총 인구
unc_pop = float(gdf_in.loc[gdf_in["is_uncovered"] == True, "pop"].sum())  # KPI 원 내부 비커버 인구
unc_rate = (unc_pop / total_pop) if total_pop > 0 else 0.0  # KPI 원 내부 비커버 비율

newly_covered_geom_5179 = None  # 새로 커버된 비커버 geometry 초기화
newly_covered_area_m2 = 0.0  # 새로 커버된 비커버 면적(㎡) 초기화
newly_covered_area_km2 = 0.0  # ✅ 새로 커버된 비커버 면적(km²) 초기화
newly_covered_pop_est = 0.0  # 새로 커버된 비커버 인구(추정) 초기화

if len(gdf_unc) > 0:  # 비커버가 있으면
    nam_union = gdf_namhyeon.geometry.union_all()  # 남현동 union
    unc_union_nam = gdf_unc.geometry.union_all().intersection(nam_union)  # 비커버를 남현동 내부로 제한
    newly_covered_geom_5179 = unc_union_nam.intersection(station_buffer_5179)  # 새로 커버된 비커버 계산
    if newly_covered_geom_5179 is not None and (not newly_covered_geom_5179.is_empty):  # 유효하면
        newly_covered_area_m2 = float(newly_covered_geom_5179.area)  # 면적(㎡)
        newly_covered_area_km2 = newly_covered_area_m2 / 1_000_000.0  # ✅ 면적(km²)로 변환(㎡→km²)

        cand = gdf_grid_nam[gdf_grid_nam["is_uncovered"] == True].copy()  # 비커버 격자만
        cand = cand[cand.geometry.intersects(newly_covered_geom_5179)].copy()  # 새로커버와 교차하는 격자만

        inter_areas, base_areas, pops = [], [], []  # 계산용 리스트 초기화
        for geom, popv in zip(cand.geometry.tolist(), cand["pop"].tolist()):  # 격자 순회
            if geom is None or geom.is_empty:  # geometry 유효성 확인
                continue  # 스킵
            base_area = float(geom.area)  # 격자 면적(㎡)
            if base_area <= 0:  # 면적이 비정상이면
                continue  # 스킵
            inter = geom.intersection(newly_covered_geom_5179)  # 교집합 geometry
            inter_area = float(inter.area) if (inter is not None and (not inter.is_empty)) else 0.0  # 교집합 면적(㎡)
            inter_areas.append(inter_area)  # 저장
            base_areas.append(base_area)  # 저장
            pops.append(float(popv))  # 저장

        if len(pops) > 0:  # 데이터가 있으면
            inter_arr = np.array(inter_areas, dtype=float)  # numpy 변환
            base_arr = np.array(base_areas, dtype=float)  # numpy 변환
            pop_arr = np.array(pops, dtype=float)  # numpy 변환
            ratio = np.clip(inter_arr / base_arr, 0.0, 1.0)  # 면적 비율(0~1)
            newly_covered_pop_est = float((pop_arr * ratio).sum())  # 인구 추정(균등분포 가정)

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)  # KPI 7개 컬럼
c1.metric("선택 gid", str(sel_gid))  # gid
c2.metric("KPI 반경 내 격자 수", f"{len(gdf_in):,}")  # 격자 수
c3.metric("총 인구", f"{total_pop:,.0f}")  # 총 인구
c4.metric("비커버 인구", f"{unc_pop:,.0f}")  # 비커버 인구
c5.metric("비커버 비율", f"{unc_rate*100:.2f}%")  # 비커버 비율
c6.metric("새로 커버된 비커버 면적(km²)", f"{newly_covered_area_km2:,.4f}")  # ✅ km² 단위(소수 4자리)
c7.metric("새로 커버된 비커버 인구(추정)", f"{newly_covered_pop_est:,.0f}")  # 인구 추정

# =========================================================  
# 5) 좌(Pydeck) / 우(Folium)  # 섹션 설명 주석
# =========================================================  

st.markdown("---")  # 구분선
left, right = st.columns([1, 1], gap="large")  # 좌/우 2컬럼 생성

# =========================================================  
# 5-A) LEFT: Pydeck  # 섹션 설명 주석
# =========================================================  

with left:  # 좌측 컬럼 컨텍스트 시작
    st.subheader("인구기반 따릉이 신규 정류소 배치")  # ✅ 파이덱 제목
    st.caption("남현동 격자(3D) + KPI 원 + 신규 커버 원 + 행정동 경계")  # 안내문

    gdf_ll = gdf_in.to_crs(MAP_CRS).copy()  # KPI 내부 격자를 4326 변환

    pop = gdf_ll["pop"].clip(lower=0).astype(float)  # pop 정리
    cap_val = float(pop.quantile(0.995)) if len(pop) > 0 else 0.0  # 극단치 cap 기준
    pop_capped = np.minimum(pop, cap_val) if cap_val > 0 else pop  # cap 적용
    gdf_ll["elev"] = (np.power(pop_capped, 1.80) * 0.02).astype(float)  # 3D 높이 계산

    grid_records = []  # pydeck 폴리곤 레코드
    for gid, popv, is_unc, elev, geom in zip(  # zip으로 동시에 순회
        gdf_ll[GRID_ID_COL].astype(str).tolist(),  # gid
        gdf_ll["pop"].tolist(),  # pop
        gdf_ll["is_uncovered"].tolist(),  # uncovered
        gdf_ll["elev"].tolist(),  # elev
        gdf_ll.geometry.tolist(),  # geometry
    ):
        if geom is None or geom.is_empty:  # geometry 유효성 체크
            continue  # 스킵
        polys = [geom] if geom.geom_type == "Polygon" else (list(geom.geoms) if geom.geom_type == "MultiPolygon" else [])  # 폴리곤 리스트화
        for poly in polys:  # 폴리곤 순회
            coords = list(map(list, poly.exterior.coords))  # [lon,lat] 좌표 리스트로 변환
            grid_records.append({"gid": gid, "pop": float(popv), "is_uncovered": bool(is_unc), "elev": float(elev), "polygon": coords})  # 레코드 추가

    kpi_circle_ll = gpd.GeoSeries([kpi_circle_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # KPI 원 4326
    station_buf_ll = gpd.GeoSeries([station_buffer_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 신규 커버 원 4326
    center_ll = gpd.GeoSeries([sel_center_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 중심점 4326

    gdf_nam_ll = gdf_namhyeon.to_crs(MAP_CRS).copy()  # 남현동 경계 4326
    admin_features = []  # GeoJSON feature 리스트
    for geom in gdf_nam_ll.geometry.tolist():  # geometry 순회
        if geom is None or geom.is_empty:  # 유효성 체크
            continue  # 스킵
        admin_features.append({"type": "Feature", "properties": {"name": "남현동"}, "geometry": mapping(geom)})  # GeoJSON feature 추가
    admin_geojson = {"type": "FeatureCollection", "features": admin_features}  # FeatureCollection 구성

    layer_admin = pdk.Layer(  # 행정동 경계 레이어
        "GeoJsonLayer",  # 레이어 타입
        data=admin_geojson,  # 데이터
        stroked=True,  # 외곽선 표시
        filled=False,  # 내부 채우기 X
        get_line_color=[0, 0, 0, 230],  # 선 색
        line_width_min_pixels=2,  # 선 두께 최소
        line_width_max_pixels=4,  # 선 두께 최대
        pickable=False,  # 클릭 비활성
    )

    layer_grid = pdk.Layer(  # 격자 3D 레이어
        "PolygonLayer",
        data=grid_records,
        get_polygon="polygon",
        extruded=True,
        filled=True,
        stroked=False,
        get_elevation="elev",
        elevation_scale=1,
        get_fill_color="[240, 240, 240, 160]",
        pickable=True,
    )

    layer_kpi = pdk.Layer(  # KPI 원 레이어(좌측은 유지)
        "PolygonLayer",
        data=[{"polygon": list(map(list, kpi_circle_ll.exterior.coords))}],
        get_polygon="polygon",
        filled=False,
        stroked=True,
        get_line_color=[30, 30, 30, 220],
        line_width_min_pixels=2,
        line_width_max_pixels=4,
        pickable=False,
    )

    layer_station_buf = pdk.Layer(  # 신규 커버 원 레이어
        "PolygonLayer",
        data=[{"polygon": list(map(list, station_buf_ll.exterior.coords))}],
        get_polygon="polygon",
        filled=False,
        stroked=True,
        get_line_color=[0, 120, 0, 220],
        line_width_min_pixels=2,
        line_width_max_pixels=4,
        pickable=False,
    )

    layer_station = pdk.Layer(  # 신규 정류장(점)
        "ScatterplotLayer",
        data=[{"lon": float(center_ll.x), "lat": float(center_ll.y)}],
        get_position="[lon, lat]",
        get_radius=70,
        pickable=True,
    )

    view = pdk.ViewState(  # 카메라 시점 설정
        latitude=float(center_ll.y),
        longitude=float(center_ll.x),
        zoom=14,
        pitch=55,
        bearing=20,
    )

    map_style = CARTO_POSITRON_GL if not MAPBOX_TOKEN else "mapbox://styles/mapbox/light-v11"  # 스타일 선택

    deck = pdk.Deck(  # Deck 구성
        layers=[layer_admin, layer_grid, layer_kpi, layer_station_buf, layer_station],
        initial_view_state=view,
        map_style=map_style,
        tooltip={"text": "gid: {gid}\npop: {pop}\nuncovered: {is_uncovered}"},
    )

    st.pydeck_chart(deck, height=MAP_HEIGHT_PX, width="stretch")  # ✅ 좌측 지도 높이 고정

# =========================================================  
# 5-B) RIGHT: Folium (KPI 반경 레이어 제거)  # 섹션 설명 주석
# =========================================================  

with right:  # 우측 컬럼 컨텍스트 시작
    st.subheader("신규 정류소 배치에 따른 커버리지 분석")  # ✅ 폴리움 제목
    st.caption("남현동 경계 + 비커버 + 신규 커버 + 5분 네트워크 (KPI 반경 표시는 제거)")  # 안내문

    center_ll = gpd.GeoSeries([sel_center_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 중심점 4326
    lon, lat = float(center_ll.x), float(center_ll.y)  # lon/lat 추출

    with st.spinner(f"OSM 그래프 다운로드/캐시 확인... (dist={GRAPH_DIST_M}m)"):  # 스피너
        G = _build_osm_graph_from_point(lat=lat, lon=lon, dist_m=int(GRAPH_DIST_M), network_type="walk")  # 그래프 생성

    with st.spinner("그래프 투영(project) + travel_time 세팅..."):  # 스피너
        Gp = ox.project_graph(G)  # 그래프 투영(미터 CRS)
        pt_ll = gpd.GeoSeries([Point(lon, lat)], crs=MAP_CRS)  # 중심점(4326) GeoSeries
        pt_proj = pt_ll.to_crs(Gp.graph["crs"]).iloc[0]  # 그래프 CRS로 변환
        px, py = float(pt_proj.x), float(pt_proj.y)  # 좌표 추출

        for u, v, k, data in Gp.edges(keys=True, data=True):  # 모든 엣지 순회
            length_m = float(data.get("length", 0.0))  # 엣지 길이(m)
            data["travel_time"] = (length_m / float(WALK_SPEED_MPS)) if WALK_SPEED_MPS > 0 else np.inf  # 시간(초)

        source_node = ox.distance.nearest_nodes(Gp, X=px, Y=py)  # 중심점 최근접 노드

    with st.spinner(f"{CUTOFF_MIN}분 네트워크 계산 중..."):  # 스피너
        lengths = nx.single_source_dijkstra_path_length(Gp, int(source_node), cutoff=float(CUTOFF_SEC), weight="travel_time")  # 도달시간 계산
        reachable_nodes = set(lengths.keys())  # 도달 가능한 노드
        SG = Gp.subgraph(reachable_nodes).copy()  # 서브그래프 생성

        gdf_edges = ox.graph_to_gdfs(SG, nodes=False, edges=True, fill_edge_geometry=True)  # 엣지를 GeoDataFrame으로
        if gdf_edges.crs is None:  # CRS가 없으면
            gdf_edges = gdf_edges.set_crs(Gp.graph["crs"])  # CRS 설정
        gdf_edges_ll = gdf_edges.to_crs(MAP_CRS).reset_index(drop=True)  # 4326 변환

    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="cartodbpositron")  # folium 지도 생성

    gdf_nam_ll = gdf_namhyeon.to_crs(MAP_CRS)  # 남현동 경계 4326
    folium.GeoJson(gdf_nam_ll, name="남현동 경계", style_function=lambda x: {"color": "#000000", "weight": 3, "fillOpacity": 0.02}).add_to(m)  # 경계 추가

    if len(gdf_unc) > 0:  # 비커버가 있으면
        nam_union = gdf_namhyeon.geometry.union_all()  # 남현동 union
        unc_union_nam = gdf_unc.geometry.union_all().intersection(nam_union)  # 비커버를 남현동 내부로 제한
        unc_ll = gpd.GeoSeries([unc_union_nam], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 4326 변환
        if not unc_ll.is_empty:  # 비어있지 않으면
            folium.GeoJson({"type": "Feature", "properties": {}, "geometry": unc_ll.__geo_interface__}, name="비커버(남현동)", style_function=lambda x: {"color": "#ff0000", "weight": 2, "fillOpacity": 0.10}).add_to(m)  # 비커버 추가

    folium.Marker(location=[lat, lon], tooltip=f"신규 따릉이 정류장(가정): gid={sel_gid}", icon=folium.Icon(color="green", icon="bicycle", prefix="fa")).add_to(m)  # 신규정류장 마커

    station_buf_ll = gpd.GeoSeries([station_buffer_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 신규 커버 반경 4326
    folium.GeoJson({"type": "Feature", "properties": {}, "geometry": station_buf_ll.__geo_interface__}, name="신규 커버 반경", style_function=lambda x: {"color": "#00aa00", "weight": 2, "fillOpacity": 0.03}).add_to(m)  # 신규 커버 추가

    if newly_covered_geom_5179 is not None and (not newly_covered_geom_5179.is_empty):  # 새로커버가 있으면
        newly_ll = gpd.GeoSeries([newly_covered_geom_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 4326 변환
        folium.GeoJson({"type": "Feature", "properties": {}, "geometry": newly_ll.__geo_interface__}, name="새로 커버된 비커버", style_function=lambda x: {"color": "#008800", "weight": 2, "fillOpacity": 0.25}).add_to(m)  # 새로커버 추가

    if len(gdf_edges_ll) > 0:  # 네트워크가 있으면
        folium.GeoJson(gdf_edges_ll, name=f"5분 네트워크({CUTOFF_MIN}min)", style_function=lambda x: {"color": "#0055ff", "weight": 3, "opacity": 0.85}).add_to(m)  # 네트워크 추가

    # ✅ KPI 반경 레이어는 우측(Folium)에 넣지 않음 → 범례에 "KPI 반경"이 생기지 않음  # 설명 주석

    folium.LayerControl(collapsed=False).add_to(m)  # 레이어 컨트롤 추가
    st_folium(m, width=None, height=MAP_HEIGHT_PX)  # ✅ 우측 지도 높이 고정(좌측과 동일)


