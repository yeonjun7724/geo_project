import os  # 경로/환경변수 처리용
import warnings  # 경고 제어용
warnings.filterwarnings("ignore")  # 불필요 경고 숨김

import numpy as np  # 수치 연산용
import pandas as pd  # 표 데이터 처리용
import geopandas as gpd  # 공간 데이터 처리용

import streamlit as st  # Streamlit UI
import pydeck as pdk  # Pydeck(WebGL 지도)

import folium  # Folium(Leaflet 지도)
from streamlit_folium import st_folium  # Streamlit에서 Folium 렌더링

import osmnx as ox  # OSM 네트워크 다운로드/가공
import networkx as nx  # 최단경로/다익스트라 계산

from shapely.geometry import Point, mapping  # 점/GeoJSON 변환


# =========================================================  # 구분선 주석
# 0) PATHS / CONSTANTS  # 섹션 설명 주석
# =========================================================  # 구분선 주석

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일(app.py) 기준 절대경로
DATA_DIR = os.path.join(BASE_DIR, "data")  # data 폴더 경로

GRID_SHP = os.path.join(DATA_DIR, "nlsp_021001001.shp")  # 전수 격자(남현동만 clip)
UNCOVERED_GPKG = os.path.join(DATA_DIR, "demo_uncovered.gpkg")  # 비커버 폴리곤(없어도 동작)
ADMIN_GPKG = os.path.join(DATA_DIR, "demo_admin.gpkg")  # 남현동 포함 행정동 경계

GRID_ID_COL = "gid"  # 격자 ID 컬럼
GRID_POP_COL = "val"  # 격자 인구 컬럼(없으면 pop=0 처리)

TARGET_CRS = 5179  # 거리/면적 계산용(미터 기반)
MAP_CRS = 4326  # 지도 시각화용(위경도)

KPI_RADIUS_M = 1250  # KPI 반경(원)
NEW_STATION_BUFFER_M = 1250  # 신규 정류장 커버 반경(원)

WALK_SPEED_MPS = 1.4  # 보행 속도(m/s)
CUTOFF_MIN = 5  # 네트워크 컷오프(분)
CUTOFF_SEC = CUTOFF_MIN * 60  # 네트워크 컷오프(초)

GRAPH_DIST_M = 3500  # OSM 그래프 다운로드 반경(미터)

CARTO_POSITRON_GL = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"  # 토큰 없는 GL 스타일

MAPBOX_TOKEN = os.getenv("MAPBOX_API_KEY") or os.getenv("MAPBOX_TOKEN")  # Mapbox 토큰(있으면 사용)
if MAPBOX_TOKEN:  # 토큰이 있으면
    pdk.settings.mapbox_api_key = MAPBOX_TOKEN  # pydeck에 토큰 주입


# =========================================================  # 구분선 주석
# 1) Streamlit Page / UI Theme  # 섹션 설명 주석
# =========================================================  # 구분선 주석

st.set_page_config(page_title="5강 | 남현동만", layout="wide")  # 페이지 타이틀/레이아웃

st.markdown(  # CSS 주입 시작
    "<style>"  # CSS 시작 태그
    ".block-container{padding-top:1.2rem;padding-bottom:2.5rem;max-width:1400px;}"  # 전체 컨테이너 폭/여백
    "h1,h2,h3{letter-spacing:-0.02em;}"  # 타이틀 자간
    ".stCaption{color:#666;}"  # 캡션 색
    ".kpi-wrap{padding:14px 14px 2px 14px;border:1px solid #eee;border-radius:14px;background:#fafafa;}"  # KPI 카드 배경
    ".soft-card{padding:14px;border:1px solid #eee;border-radius:14px;background:white;}"  # 좌/우 카드 배경
    ".small-muted{color:#777;font-size:0.92rem;}"  # 작은 안내문
    ".hr{height:1px;background:#eee;margin:14px 0;}"  # 구분선
    "</style>",  # CSS 종료 태그
    unsafe_allow_html=True,  # HTML 허용
)  # CSS 주입 종료

st.title("🚲 5강 | 남현동만")  # 페이지 제목
st.caption(  # 상단 고정 파라미터 표시
    f"고정값: KPI반경={KPI_RADIUS_M}m | 보행속도={WALK_SPEED_MPS}m/s | "  # 텍스트 1
    f"컷오프={CUTOFF_MIN}분 | 그래프반경={GRAPH_DIST_M}m | 신규 커버반경={NEW_STATION_BUFFER_M}m"  # 텍스트 2
)  # 캡션 종료

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)  # 상단 구분선


# =========================================================  # 구분선 주석
# 2) (최소 함수) OSM 그래프 빌드만 cache_resource로 유지  # 섹션 설명 주석
# - OSM 다운로드는 비용이 크므로 캐시가 실사용에 중요  # 설명 주석
# =========================================================  # 구분선 주석

@st.cache_resource(show_spinner=False)  # 그래프 다운로드 결과를 리소스 캐시
def _build_osm_graph_from_point(lat: float, lon: float, dist_m: int, network_type: str = "walk"):  # OSM 그래프 생성 함수
    ox.settings.log_console = False  # OSMnx 콘솔 로그 끔
    G = ox.graph_from_point(  # 중심점 기준 그래프 다운로드
        (float(lat), float(lon)),  # (위도, 경도)
        dist=int(dist_m),  # 다운로드 반경
        network_type=network_type,  # walk 네트워크
        simplify=True,  # simplify 적용
    )  # graph_from_point 종료
    try:  # OSMnx 2.x 호환 시도
        G = ox.distance.add_edge_lengths(G)  # edge length 추가
    except Exception:  # 실패하면
        try:  # 구버전 호환 시도
            G = ox.add_edge_lengths(G)  # 구버전 함수
        except Exception:  # 그것도 실패하면
            pass  # 그냥 진행(이미 length가 있을 수도 있음)
    return G  # 그래프 반환


# =========================================================  # 구분선 주석
# 3) DATA LOAD (스크립트형)  # 섹션 설명 주석
# =========================================================  # 구분선 주석

if not os.path.exists(GRID_SHP):  # 격자 파일 존재 확인
    st.error(f"GRID_SHP not found: {GRID_SHP}")  # 에러 표시
    st.stop()  # 중단

if not os.path.exists(ADMIN_GPKG):  # 행정동 파일 존재 확인
    st.error("남현동 행정구역 파일이 필요합니다. data/demo_admin.gpkg 를 넣어주세요.")  # 에러 표시
    st.stop()  # 중단

with st.spinner("격자 로딩 중..."):  # 로딩 스피너
    gdf_grid = gpd.read_file(GRID_SHP)  # 격자 로드

if gdf_grid.crs is None:  # CRS 확인
    st.error("GRID_SHP CRS is None. (.prj 확인)")  # 에러
    st.stop()  # 중단

gdf_grid = gdf_grid.to_crs(TARGET_CRS)  # 분석 CRS로 변환

if GRID_ID_COL not in gdf_grid.columns:  # gid 컬럼 확인
    st.error(f"GRID_ID_COL='{GRID_ID_COL}' not found in grid")  # 에러
    st.stop()  # 중단

gdf_grid[GRID_ID_COL] = gdf_grid[GRID_ID_COL].astype(str)  # gid 문자열 통일

if GRID_POP_COL in gdf_grid.columns:  # val 컬럼이 있으면
    gdf_grid["pop"] = pd.to_numeric(gdf_grid[GRID_POP_COL], errors="coerce").fillna(0).astype(float)  # pop 생성
elif "pop" in gdf_grid.columns:  # pop 컬럼이 이미 있으면
    gdf_grid["pop"] = pd.to_numeric(gdf_grid["pop"], errors="coerce").fillna(0).astype(float)  # pop 정리
else:  # 둘 다 없으면
    gdf_grid["pop"] = 0.0  # pop=0

gdf_grid["geometry"] = gdf_grid.geometry.buffer(0)  # geometry 정리(자기교차 등 완화)
gdf_grid = gdf_grid[[GRID_ID_COL, "pop", "geometry"]].copy()  # 필요한 컬럼만 유지

with st.spinner("행정동(남현동) 로딩/선택 중..."):  # 로딩 스피너
    gdf_admin = gpd.read_file(ADMIN_GPKG)  # 행정동 로드

if gdf_admin.crs is None:  # CRS 확인
    st.error("ADMIN CRS is None.")  # 에러
    st.stop()  # 중단

gdf_admin = gdf_admin.to_crs(TARGET_CRS)  # 분석 CRS로 변환
gdf_admin["geometry"] = gdf_admin.geometry.buffer(0)  # geometry 정리

NAME_COL_CANDIDATES = [  # 남현동 이름 컬럼 후보
    "ADM_NM", "adm_nm", "ADMNM",  # 후보 1
    "region_nm", "REGION_NM",  # 후보 2
    "emd_nm", "EMD_NM",  # 후보 3
    "dong_nm", "DONG_NM",  # 후보 4
    "법정동명", "행정동명",  # 후보 5
]  # 후보 리스트 종료

name_col = None  # 선택된 이름 컬럼 초기화
for c in NAME_COL_CANDIDATES:  # 후보 순회
    if c in gdf_admin.columns:  # 존재하면
        name_col = c  # 선택
        break  # 종료

if name_col is None:  # 이름 컬럼을 못 찾으면
    gdf_namhyeon = gdf_admin.iloc[[0]].copy()  # 최소 동작: 첫 행 사용
else:  # 이름 컬럼이 있으면
    s = gdf_admin[name_col].astype(str)  # 문자열 변환
    mask = s.str.contains("남현", na=False)  # "남현" 포함 여부
    if mask.sum() == 0:  # 없으면
        mask = s.str.contains("남현동", na=False)  # "남현동"로 재시도
    if mask.sum() == 0:  # 그래도 없으면
        gdf_namhyeon = gdf_admin.iloc[[0]].copy()  # 최소 동작: 첫 행
    else:  # 있으면
        gdf_namhyeon = gdf_admin.loc[mask].copy()  # 남현동만 필터

with st.spinner("남현동 격자만 clip 중..."):  # 로딩 스피너
    nam_union = gdf_namhyeon.geometry.union_all()  # 남현동 폴리곤 합치기(단일 geometry)
    gdf_sub = gdf_grid[gdf_grid.geometry.intersects(nam_union)].copy()  # 교차 격자 1차 필터(속도)
    gdf_grid_nam = gpd.clip(gdf_sub, gdf_namhyeon).copy()  # 실제 clip
    gdf_grid_nam["geometry"] = gdf_grid_nam.geometry.buffer(0)  # geometry 정리

if len(gdf_grid_nam) == 0:  # clip 결과 0이면
    st.error("남현동으로 clip된 격자가 0개입니다. 행정구역 파일/CRS/남현동 명칭 컬럼을 확인하세요.")  # 에러
    st.stop()  # 중단

if os.path.exists(UNCOVERED_GPKG):  # 비커버 파일이 있으면
    gdf_unc = gpd.read_file(UNCOVERED_GPKG)  # 로드
    if gdf_unc.crs is None:  # CRS 확인
        st.error("UNCOVERED_GPKG CRS is None.")  # 에러
        st.stop()  # 중단
    gdf_unc = gdf_unc.to_crs(TARGET_CRS)  # 분석 CRS
    gdf_unc["geometry"] = gdf_unc.geometry.buffer(0)  # geometry 정리
    gdf_unc = gdf_unc[["geometry"]].copy()  # geometry만 유지
else:  # 파일이 없으면
    gdf_unc = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=TARGET_CRS)  # 빈 GDF 생성

if len(gdf_unc) == 0:  # 비커버가 없으면
    gdf_grid_nam["is_uncovered"] = False  # 전부 커버로 처리
else:  # 비커버가 있으면
    unc_union = gdf_unc.geometry.union_all()  # 비커버 합치기
    gdf_grid_nam["is_uncovered"] = gdf_grid_nam.geometry.intersects(unc_union)  # 교차하면 비커버 True

all_gids = gdf_grid_nam[GRID_ID_COL].astype(str).tolist()  # gid 목록


# =========================================================  # 구분선 주석
# 4) KPI + gid 선택(요청 위치 유지) + 신규 KPI 2종 추가  # 섹션 설명 주석
# =========================================================  # 구분선 주석

st.markdown('<div class="kpi-wrap">', unsafe_allow_html=True)  # KPI 카드 시작
st.subheader("KPI")  # KPI 제목
st.markdown('<div class="small-muted">gid를 선택하면 KPI와 좌/우 지도가 동시에 갱신됩니다.</div>', unsafe_allow_html=True)  # 안내문

sel_gid = st.selectbox("남현동 격자 gid 선택", options=all_gids, index=0, key="gid_select")  # gid 선택 UI

row = gdf_grid_nam.loc[gdf_grid_nam[GRID_ID_COL] == str(sel_gid)]  # 선택 gid 행
if len(row) == 0:  # 없으면
    st.error("선택 gid를 남현동 격자에서 찾지 못했습니다.")  # 에러
    st.stop()  # 중단

sel_poly = row.geometry.iloc[0]  # 선택 격자 폴리곤
sel_center_5179 = sel_poly.centroid  # 격자 중심점(5179)

kpi_circle_5179 = sel_center_5179.buffer(float(KPI_RADIUS_M))  # KPI 원
station_buffer_5179 = sel_center_5179.buffer(float(NEW_STATION_BUFFER_M))  # 신규 커버 원

in_circle = gdf_grid_nam.geometry.intersects(kpi_circle_5179)  # KPI 원과 교차 여부
gdf_in = gdf_grid_nam.loc[in_circle, [GRID_ID_COL, "pop", "is_uncovered", "geometry"]].copy()  # KPI 원 내부 격자

total_pop = float(gdf_in["pop"].sum())  # 총 인구
unc_pop = float(gdf_in.loc[gdf_in["is_uncovered"] == True, "pop"].sum())  # 비커버 인구
cov_pop = total_pop - unc_pop  # 커버 인구
unc_rate = (unc_pop / total_pop) if total_pop > 0 else 0.0  # 비커버 비율

newly_covered_geom_5179 = None  # 새로 커버된 비커버 geometry 초기화
newly_covered_area_m2 = 0.0  # 새로 커버된 비커버 면적(㎡) 초기화
newly_covered_pop_est = 0.0  # 새로 커버된 비커버 인구(추정) 초기화

if len(gdf_unc) > 0:  # 비커버가 있으면
    nam_union = gdf_namhyeon.geometry.union_all()  # 남현동 폴리곤 합
    unc_union_nam = gdf_unc.geometry.union_all().intersection(nam_union)  # 비커버를 남현동 내부로 제한
    newly_covered_geom_5179 = unc_union_nam.intersection(station_buffer_5179)  # (비커버∩남현동)∩신규커버
    if newly_covered_geom_5179 is not None and (not newly_covered_geom_5179.is_empty):  # 유효하면
        newly_covered_area_m2 = float(newly_covered_geom_5179.area)  # 면적(㎡) 계산 (5179=미터)
        # -------------------------------------------------  # 구분 주석
        # 인구(추정) 계산: 비커버 격자(pop)가 격자 내에 균등 분포한다고 가정  # 설명 주석
        # => pop * (면적(격자∩새로커버) / 면적(격자)) 합산  # 설명 주석
        # -------------------------------------------------  # 구분 주석
        cand = gdf_grid_nam[gdf_grid_nam["is_uncovered"] == True].copy()  # 비커버 격자만 후보
        cand = cand[cand.geometry.intersects(newly_covered_geom_5179)].copy()  # 새로커버와 교차하는 격자만
        if len(cand) > 0:  # 후보가 있으면
            inter_areas = []  # 교집합 면적 리스트
            base_areas = []  # 격자 면적 리스트
            pops = []  # pop 리스트
            for geom, popv in zip(cand.geometry.tolist(), cand["pop"].tolist()):  # 각 격자 순회
                if geom is None or geom.is_empty:  # geometry가 비정상이면
                    continue  # 스킵
                base_area = float(geom.area)  # 격자 면적(㎡)
                if base_area <= 0:  # 면적이 0이면
                    continue  # 스킵
                inter = geom.intersection(newly_covered_geom_5179)  # 격자와 새로커버 교집합
                inter_area = float(inter.area) if (inter is not None and (not inter.is_empty)) else 0.0  # 교집합 면적
                inter_areas.append(inter_area)  # 교집합 면적 저장
                base_areas.append(base_area)  # 격자 면적 저장
                pops.append(float(popv))  # pop 저장
            if len(pops) > 0:  # 유효 데이터가 있으면
                inter_arr = np.array(inter_areas, dtype=float)  # numpy 배열 변환
                base_arr = np.array(base_areas, dtype=float)  # numpy 배열 변환
                pop_arr = np.array(pops, dtype=float)  # numpy 배열 변환
                ratio = np.clip(inter_arr / base_arr, 0.0, 1.0)  # 면적 비율(0~1)
                newly_covered_pop_est = float((pop_arr * ratio).sum())  # 비율만큼 pop 배분 후 합


# KPI 카드(7개로 확장)  # 설명 주석
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)  # KPI 컬럼 7개
c1.metric("선택 gid", str(sel_gid))  # gid 표시
c2.metric("KPI 반경 내 격자 수", f"{len(gdf_in):,}")  # 격자 수
c3.metric("총 인구", f"{total_pop:,.0f}")  # 총 인구
c4.metric("비커버 인구", f"{unc_pop:,.0f}")  # 비커버 인구
c5.metric("비커버 비율", f"{unc_rate*100:.2f}%")  # 비커버 비율
c6.metric("새로 커버된 비커버 면적(㎡)", f"{newly_covered_area_m2:,.0f}")  # 새로 커버 면적
c7.metric("새로 커버된 비커버 인구(추정)", f"{newly_covered_pop_est:,.0f}")  # 새로 커버 인구(추정)

st.markdown("</div>", unsafe_allow_html=True)  # KPI 카드 종료


# =========================================================  # 구분선 주석
# 5) 좌(Pydeck) / 우(Folium)  # 섹션 설명 주석
# =========================================================  # 구분선 주석

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)  # 구분선
left, right = st.columns([1, 1], gap="large")  # 좌/우 컬럼


# =========================================================  # 구분선 주석
# 5-A) LEFT: Pydeck  # 섹션 설명 주석
# =========================================================  # 구분선 주석

with left:  # 좌측 영역 시작
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)  # 카드 시작
    st.subheader("좌측: Pydeck")  # 제목
    st.markdown('<div class="small-muted">남현동 격자(3D) + KPI 원 + 신규 커버 원 + 행정동 경계(GeoJSON)</div>', unsafe_allow_html=True)  # 안내문

    gdf_ll = gdf_in.to_crs(MAP_CRS).copy()  # KPI 반경 내 격자를 4326으로 변환

    pop = gdf_ll["pop"].clip(lower=0).astype(float)  # pop 음수 방지 + float
    cap_val = float(pop.quantile(0.995)) if len(pop) > 0 else 0.0  # 상위 0.5% cap
    pop_capped = np.minimum(pop, cap_val) if cap_val > 0 else pop  # cap 적용
    gdf_ll["elev"] = (np.power(pop_capped, 1.80) * 0.02).astype(float)  # 3D 높이(튜닝값)

    grid_records = []  # pydeck 입력용 레코드 리스트
    for gid, popv, is_unc, elev, geom in zip(  # zip 순회 시작
        gdf_ll[GRID_ID_COL].astype(str).tolist(),  # gid 리스트
        gdf_ll["pop"].tolist(),  # pop 리스트
        gdf_ll["is_uncovered"].tolist(),  # 비커버 여부 리스트
        gdf_ll["elev"].tolist(),  # 높이 리스트
        gdf_ll.geometry.tolist(),  # geometry 리스트
    ):  # zip 순회 종료
        if geom is None or geom.is_empty:  # geometry 유효성 체크
            continue  # 스킵
        polys = [geom] if geom.geom_type == "Polygon" else (list(geom.geoms) if geom.geom_type == "MultiPolygon" else [])  # 폴리곤 리스트화
        for poly in polys:  # 폴리곤마다
            coords = list(map(list, poly.exterior.coords))  # exterior 좌표를 [lon,lat] 리스트로
            grid_records.append({  # 레코드 추가
                "gid": gid,  # gid
                "pop": float(popv),  # pop
                "is_uncovered": bool(is_unc),  # 비커버 여부
                "elev": float(elev),  # 높이
                "polygon": coords,  # 폴리곤 좌표
            })  # append 종료

    kpi_circle_ll = gpd.GeoSeries([kpi_circle_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # KPI 원 4326
    station_buf_ll = gpd.GeoSeries([station_buffer_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 신규 커버 원 4326
    center_ll = gpd.GeoSeries([sel_center_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 중심점 4326

    gdf_nam_ll = gdf_namhyeon.to_crs(MAP_CRS).copy()  # 남현동 경계를 4326으로
    admin_features = []  # GeoJSON feature 리스트
    for geom in gdf_nam_ll.geometry.tolist():  # geometry 순회
        if geom is None or geom.is_empty:  # 유효성 체크
            continue  # 스킵
        admin_features.append({  # feature 추가
            "type": "Feature",  # GeoJSON 타입
            "properties": {"name": "남현동"},  # 속성
            "geometry": mapping(geom),  # geometry를 GeoJSON dict로
        })  # append 종료
    admin_geojson = {"type": "FeatureCollection", "features": admin_features}  # FeatureCollection 구성

    layer_admin = pdk.Layer(  # 행정동 경계 레이어
        "GeoJsonLayer",  # 레이어 타입
        data=admin_geojson,  # GeoJSON 데이터
        stroked=True,  # 외곽선 표시
        filled=False,  # 면 채우기 없음
        get_line_color=[0, 0, 0, 230],  # 선 색(검정)
        line_width_min_pixels=2,  # 최소 두께(px)
        line_width_max_pixels=4,  # 최대 두께(px)
        pickable=False,  # 클릭 상호작용 없음
    )  # layer_admin 종료

    layer_grid = pdk.Layer(  # 격자 3D 레이어
        "PolygonLayer",  # 폴리곤 레이어
        data=grid_records,  # 레코드
        get_polygon="polygon",  # 폴리곤 좌표 키
        extruded=True,  # 3D extrusion
        filled=True,  # 채움
        stroked=False,  # 외곽선(격자)은 끔
        get_elevation="elev",  # 높이
        elevation_scale=1,  # 스케일
        get_fill_color="[240, 240, 240, 160]",  # 회색 톤
        pickable=True,  # 툴팁 가능
    )  # layer_grid 종료

    layer_kpi = pdk.Layer(  # KPI 원 레이어
        "PolygonLayer",  # 폴리곤 레이어
        data=[{"polygon": list(map(list, kpi_circle_ll.exterior.coords))}],  # 원 좌표
        get_polygon="polygon",  # 폴리곤 키
        filled=False,  # 채움 없음
        stroked=True,  # 선 표시
        get_line_color=[30, 30, 30, 220],  # 선 색
        line_width_min_pixels=2,  # 두께
        line_width_max_pixels=4,  # 두께
        pickable=False,  # 상호작용 없음
    )  # layer_kpi 종료

    layer_station_buf = pdk.Layer(  # 신규 커버 원 레이어
        "PolygonLayer",  # 폴리곤 레이어
        data=[{"polygon": list(map(list, station_buf_ll.exterior.coords))}],  # 원 좌표
        get_polygon="polygon",  # 폴리곤 키
        filled=False,  # 채움 없음
        stroked=True,  # 선 표시
        get_line_color=[0, 120, 0, 220],  # 초록 선
        line_width_min_pixels=2,  # 두께
        line_width_max_pixels=4,  # 두께
        pickable=False,  # 상호작용 없음
    )  # layer_station_buf 종료

    layer_station = pdk.Layer(  # 정류장 중심점 레이어
        "ScatterplotLayer",  # 점 레이어
        data=[{"lon": float(center_ll.x), "lat": float(center_ll.y)}],  # 중심점
        get_position="[lon, lat]",  # 위치 키
        get_radius=70,  # 반경
        pickable=True,  # 상호작용
    )  # layer_station 종료

    view = pdk.ViewState(  # 카메라 뷰
        latitude=float(center_ll.y),  # 위도
        longitude=float(center_ll.x),  # 경도
        zoom=14,  # 줌
        pitch=55,  # 피치(3D)
        bearing=20,  # 회전
    )  # view 종료

    map_style = CARTO_POSITRON_GL if not MAPBOX_TOKEN else "mapbox://styles/mapbox/light-v11"  # 스타일 선택

    deck = pdk.Deck(  # pydeck deck 구성
        layers=[layer_admin, layer_grid, layer_kpi, layer_station_buf, layer_station],  # 레이어 순서(경계가 위에 보이게)
        initial_view_state=view,  # 초기 뷰
        map_style=map_style,  # 스타일
        tooltip={"text": "gid: {gid}\npop: {pop}\nuncovered: {is_uncovered}"},  # 툴팁
    )  # deck 종료

    st.pydeck_chart(deck, width="stretch")  # pydeck 렌더
    st.markdown("</div>", unsafe_allow_html=True)  # 카드 종료


# =========================================================  # 구분선 주석
# 5-B) RIGHT: Folium + 즉석 5분 네트워크  # 섹션 설명 주석
# =========================================================  # 구분선 주석

with right:  # 우측 영역 시작
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)  # 카드 시작
    st.subheader("우측: Folium")  # 제목
    st.markdown('<div class="small-muted">남현동 경계 + 비커버 + 신규 커버 + 5분 네트워크</div>', unsafe_allow_html=True)  # 안내문

    center_ll = gpd.GeoSeries([sel_center_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 중심점 4326
    lon, lat = float(center_ll.x), float(center_ll.y)  # lon/lat 추출

    with st.spinner(f"OSM 그래프 다운로드/캐시 확인... (dist={GRAPH_DIST_M}m)"):  # 로딩 스피너
        G = _build_osm_graph_from_point(lat=lat, lon=lon, dist_m=int(GRAPH_DIST_M), network_type="walk")  # 그래프 다운로드

    with st.spinner("그래프 투영(project) + travel_time 세팅..."):  # 스피너
        Gp = ox.project_graph(G)  # 그래프를 미터 CRS로 투영
        pt_ll = gpd.GeoSeries([Point(lon, lat)], crs=MAP_CRS)  # 중심점(4326) GeoSeries
        pt_proj = pt_ll.to_crs(Gp.graph["crs"]).iloc[0]  # 그래프 CRS로 변환
        px, py = float(pt_proj.x), float(pt_proj.y)  # 투영 좌표 추출

        for u, v, k, data in Gp.edges(keys=True, data=True):  # 모든 엣지 순회
            length_m = float(data.get("length", 0.0))  # 길이(m)
            data["travel_time"] = (length_m / float(WALK_SPEED_MPS)) if WALK_SPEED_MPS > 0 else np.inf  # 시간(초)

        source_node = ox.distance.nearest_nodes(Gp, X=px, Y=py)  # 중심점에서 가장 가까운 노드

    with st.spinner(f"{CUTOFF_MIN}분 네트워크 계산 중..."):  # 스피너
        lengths = nx.single_source_dijkstra_path_length(  # 다익스트라(단일출발)로 도달시간 계산
            Gp, int(source_node), cutoff=float(CUTOFF_SEC), weight="travel_time"  # 소스/컷오프/가중치
        )  # lengths 계산 종료
        reachable_nodes = set(lengths.keys())  # 도달 가능한 노드 집합
        SG = Gp.subgraph(reachable_nodes).copy()  # 서브그래프(5분 이내)

        gdf_edges = ox.graph_to_gdfs(SG, nodes=False, edges=True, fill_edge_geometry=True)  # edges GeoDataFrame 변환
        if gdf_edges.crs is None:  # CRS가 없으면
            gdf_edges = gdf_edges.set_crs(Gp.graph["crs"])  # 그래프 CRS로 세팅
        gdf_edges_ll = gdf_edges.to_crs(MAP_CRS).reset_index(drop=True)  # 4326 변환

    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="cartodbpositron")  # Folium 맵 생성

    gdf_nam_ll = gdf_namhyeon.to_crs(MAP_CRS)  # 남현동 경계 4326
    folium.GeoJson(  # 남현동 경계 추가
        gdf_nam_ll,  # 데이터
        name="남현동 경계",  # 레이어명
        style_function=lambda x: {"color": "#000000", "weight": 3, "fillOpacity": 0.02},  # 스타일
    ).add_to(m)  # 맵에 추가

    if len(gdf_unc) > 0:  # 비커버가 있으면
        nam_union = gdf_namhyeon.geometry.union_all()  # 남현동 union
        unc_union_nam = gdf_unc.geometry.union_all().intersection(nam_union)  # 비커버를 남현동 내부로 제한
        unc_ll = gpd.GeoSeries([unc_union_nam], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 4326 변환
        if not unc_ll.is_empty:  # 비어있지 않으면
            folium.GeoJson(  # 비커버 레이어
                {"type": "Feature", "properties": {}, "geometry": unc_ll.__geo_interface__},  # GeoJSON
                name="비커버(남현동)",  # 레이어명
                style_function=lambda x: {"color": "#ff0000", "weight": 2, "fillOpacity": 0.10},  # 스타일
            ).add_to(m)  # 추가

    folium.Marker(  # 신규 정류장 마커
        location=[lat, lon],  # 위치
        tooltip=f"신규 따릉이 정류장(가정): gid={sel_gid}",  # 툴팁
        icon=folium.Icon(color="green", icon="bicycle", prefix="fa"),  # 아이콘
    ).add_to(m)  # 추가

    station_buf_ll = gpd.GeoSeries([station_buffer_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 신규커버 원 4326
    folium.GeoJson(  # 신규 커버 반경
        {"type": "Feature", "properties": {}, "geometry": station_buf_ll.__geo_interface__},  # GeoJSON
        name="신규 커버 반경",  # 레이어명
        style_function=lambda x: {"color": "#00aa00", "weight": 2, "fillOpacity": 0.03},  # 스타일
    ).add_to(m)  # 추가

    if newly_covered_geom_5179 is not None and (not newly_covered_geom_5179.is_empty):  # 새로커버가 있으면
        newly_ll = gpd.GeoSeries([newly_covered_geom_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 4326 변환
        folium.GeoJson(  # 새로 커버된 비커버 폴리곤
            {"type": "Feature", "properties": {}, "geometry": newly_ll.__geo_interface__},  # GeoJSON
            name="새로 커버된 비커버",  # 레이어명
            style_function=lambda x: {"color": "#008800", "weight": 2, "fillOpacity": 0.25},  # 스타일
        ).add_to(m)  # 추가

    if len(gdf_edges_ll) > 0:  # 네트워크 엣지가 있으면
        folium.GeoJson(  # 네트워크 레이어
            gdf_edges_ll,  # 엣지
            name=f"5분 네트워크({CUTOFF_MIN}min)",  # 레이어명
            style_function=lambda x: {"color": "#0055ff", "weight": 3, "opacity": 0.85},  # 스타일
        ).add_to(m)  # 추가

    kpi_circle_ll = gpd.GeoSeries([kpi_circle_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # KPI 원 4326
    folium.GeoJson(  # KPI 원 레이어
        {"type": "Feature", "properties": {}, "geometry": kpi_circle_ll.__geo_interface__},  # GeoJSON
        name="KPI 반경",  # 레이어명
        style_function=lambda x: {"color": "#111111", "weight": 2, "opacity": 0.7},  # 스타일
    ).add_to(m)  # 추가

    folium.LayerControl(collapsed=False).add_to(m)  # 레이어 컨트롤
    st_folium(m, width=None, height=680)  # Streamlit 렌더

    st.markdown("</div>", unsafe_allow_html=True)  # 카드 종료
