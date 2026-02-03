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

from shapely.geometry import Point, mapping, LineString  # Point 생성 + GeoJSON 변환 + LineString
import fiona  # GPKG 레이어 목록 확인용(있으면 사용)

# =========================================================
# 0) PATHS / CONSTANTS  # 섹션 설명 주석
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 app.py 파일의 절대경로 기준 폴더
DATA_DIR = os.path.join(BASE_DIR, "data")  # data 폴더 경로 구성

GRID_SHP = os.path.join(DATA_DIR, "nlsp_021001001.shp")  # 전수 격자(shp) 경로
UNCOVERED_GPKG = os.path.join(DATA_DIR, "demo_uncovered.gpkg")  # 비커버 폴리곤(gpkg) 경로(없어도 동작)
ADMIN_GPKG = os.path.join(DATA_DIR, "demo_admin.gpkg")  # 행정동 경계(gpkg) 경로(남현동 포함)

BUS_STOP_CSV = "data/서울시버스정류소위치정보(20260108).csv"  # 업로드된 버스정류장 CSV(스프레드시트 역할)
ROUTES_ALL_GPKG = "data/routes_all.gpkg"  # 업로드된 기존 커버 경로 GPKG(여러 레이어일 수 있음)

GRID_ID_COL = "gid"  # 격자 ID 컬럼명
GRID_POP_COL = "val"  # 격자 인구 컬럼명

TARGET_CRS = 5179  # 분석용 CRS(미터 기반, 면적/거리 계산)
MAP_CRS = 4326  # 지도 표출용 CRS(위경도)

KPI_RADIUS_M = 1250  # KPI 반경(미터)
NEW_STATION_BUFFER_M = 1250  # 신규 정류장 커버 반경(미터)

WALK_SPEED_MPS = 1.4  # 보행 속도(m/s)
CUTOFF_MIN = 5  # 네트워크 컷오프 시간(분)
CUTOFF_SEC = CUTOFF_MIN * 60  # 네트워크 컷오프 시간(초)

GRAPH_DIST_M = 3500  # OSM 그래프 다운로드 반경(미터)
MAP_HEIGHT_PX = 720  # 좌/우 지도 높이(px)

CARTO_POSITRON_GL = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"  # 토큰 없이도 되는 basemap

MAPBOX_TOKEN = os.getenv("MAPBOX_API_KEY") or os.getenv("MAPBOX_TOKEN")  # Mapbox 토큰(있으면 사용)
if MAPBOX_TOKEN:  # 토큰이 존재하면
    pdk.settings.mapbox_api_key = MAPBOX_TOKEN  # pydeck에 토큰 설정

# =========================================================
# 1) Streamlit Page / UI Theme  # 섹션 설명 주석
# =========================================================

st.set_page_config(page_title="따릉이 신규 정류소 배치를 통한 기대효과 대시보드 - 남현동", layout="wide")  # 페이지 설정(가로 레이아웃)

st.markdown(  # CSS를 주입하기 위한 markdown 호출
    """
    <style>
      .block-container { padding: 1.2rem 1.0rem 1.6rem 1.0rem; max-width: none; }  /* 컨테이너 패딩/폭 */
      h1, h2, h3 { letter-spacing: -0.02em; }  /* 타이틀 자간 */
      .stApp h1, div[data-testid="stMarkdownContainer"] h1, .stTitle { text-align: center; width: 100%; }  /* 타이틀 중앙정렬 */
      div[data-testid="stMarkdownContainer"] h1 { margin-top: 0.2rem; margin-bottom: 0.8rem; }  /* 타이틀 여백 */
      .stCaption { color: #666; }  /* 캡션 색상 */
    </style>
    """,  # CSS 문자열
    unsafe_allow_html=True,  # HTML/CSS 허용
)

st.title("따릉이 신규 정류소 배치를 통한 기대효과 대시보드 - 남현동")  # 타이틀 출력

# =========================================================
# 2) DATA LOAD (스크립트형)  # 섹션 설명 주석
# =========================================================

gdf_grid = gpd.read_file(GRID_SHP)  # 격자 shapefile 로드
gdf_grid = gdf_grid.to_crs(TARGET_CRS)  # 분석 CRS로 변환(미터 기반)
gdf_grid[GRID_ID_COL] = gdf_grid[GRID_ID_COL].astype(str)  # gid 문자열로 통일
gdf_grid["pop"] = pd.to_numeric(gdf_grid.get(GRID_POP_COL, 0), errors="coerce").fillna(0).astype(float)  # pop 생성/정리
gdf_grid = gdf_grid[[GRID_ID_COL, "pop", "geometry"]].copy()  # 필요한 컬럼만 유지

gdf_admin = gpd.read_file(ADMIN_GPKG)  # 행정동 gpkg 로드
gdf_admin = gdf_admin.to_crs(TARGET_CRS)  # 분석 CRS로 변환
name_col = next(c for c in ["ADM_NM", "region_nm", "emd_nm", "dong_nm", "법정동명", "행정동명"] if c in gdf_admin.columns)  # 이름 컬럼 선택(있다고 가정)
gdf_namhyeon = gdf_admin[gdf_admin[name_col].astype(str).str.contains("남현", na=False)].copy()  # 남현동 행만 필터
nam_union = gdf_namhyeon.geometry.union_all()  # 남현동 폴리곤을 union으로 합치기

gdf_grid_nam = gpd.clip(gdf_grid[gdf_grid.geometry.intersects(nam_union)], gdf_namhyeon).copy()  # 남현동 격자만 clip
gdf_grid_nam["is_uncovered"] = False  # 기본값으로 커버 처리

if os.path.exists(UNCOVERED_GPKG):  # 비커버 파일이 있으면
    gdf_unc = gpd.read_file(UNCOVERED_GPKG)  # 비커버 gpkg 로드
    gdf_unc = gdf_unc.to_crs(TARGET_CRS)  # 분석 CRS로 변환
    gdf_unc = gdf_unc[["geometry"]].copy()  # geometry만 남김
    unc_union = gdf_unc.geometry.union_all()  # 비커버 폴리곤 union
    gdf_grid_nam["is_uncovered"] = gdf_grid_nam.geometry.intersects(unc_union)  # 비커버 교차 격자만 True
else:  # 비커버 파일이 없으면
    gdf_unc = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=TARGET_CRS)  # 빈 GeoDataFrame 생성

# =========================================================
# 2-B) BUS STOP CSV LOAD (컬럼 고정)  # 섹션 설명 주석
# - CSV 컬럼: NODE_ID, ARS_ID, 정류소명, X좌표, Y좌표, 정류소타입  # 설명 주석
# - X좌표=경도(lon), Y좌표=위도(lat) 로 간주  # 설명 주석
# =========================================================

df_bus = pd.read_csv(BUS_STOP_CSV, encoding="utf-8", low_memory=False)  # 버스정류장 CSV 로드
df_bus["X좌표"] = pd.to_numeric(df_bus["X좌표"], errors="coerce")  # X좌표를 숫자(경도)로 변환
df_bus["Y좌표"] = pd.to_numeric(df_bus["Y좌표"], errors="coerce")  # Y좌표를 숫자(위도)로 변환
df_bus = df_bus.dropna(subset=["X좌표", "Y좌표"]).copy()  # 좌표가 없는 행 제거
df_bus["ARS_ID"] = df_bus["ARS_ID"].astype(str)  # 정류장 ID는 문자열로 통일(표시/툴팁 안정화)
df_bus["정류소명"] = df_bus["정류소명"].astype(str)  # 정류소명 문자열화
df_bus["정류소타입"] = df_bus["정류소타입"].astype(str)  # 타입 문자열화

gdf_bus = gpd.GeoDataFrame(  # 좌표 -> 포인트로 변환하여 GeoDataFrame 생성
    df_bus[["NODE_ID", "ARS_ID", "정류소명", "정류소타입", "X좌표", "Y좌표"]].copy(),  # 필요한 컬럼만
    geometry=gpd.points_from_xy(df_bus["X좌표"], df_bus["Y좌표"]),  # (lon, lat)로 포인트 생성
    crs=MAP_CRS,  # 원본 좌표계는 WGS84(4326)로 가정
)  # gdf_bus 생성 끝
gdf_bus_5179 = gdf_bus.to_crs(TARGET_CRS)  # 공간필터(남현동 내부)용으로 5179 변환
gdf_bus_nam = gdf_bus_5179[gdf_bus_5179.geometry.within(nam_union)].copy()  # 남현동 폴리곤 내부 정류장만 선택

all_gids = gdf_grid_nam[GRID_ID_COL].astype(str).tolist()  # 남현동 격자 gid 목록

# =========================================================
# 3) KPI + gid 선택  # 섹션 설명 주석
# =========================================================

st.subheader("KPI")  # KPI 섹션 제목
st.caption("gid를 선택하면 KPI와 좌/우 지도가 동시에 갱신됩니다.")  # 안내문

sel_gid = st.selectbox("남현동 격자 gid 선택", options=all_gids, index=0, key="gid_select")  # gid 선택 UI

sel_poly = gdf_grid_nam.loc[gdf_grid_nam[GRID_ID_COL] == str(sel_gid), "geometry"].iloc[0]  # 선택 격자 폴리곤
sel_center_5179 = sel_poly.centroid  # 선택 격자 중심점(5179)

kpi_circle_5179 = sel_center_5179.buffer(float(KPI_RADIUS_M))  # KPI 반경 원(5179)
station_buffer_5179 = sel_center_5179.buffer(float(NEW_STATION_BUFFER_M))  # 신규 정류장 커버 반경 원(5179)

gdf_in = gdf_grid_nam[gdf_grid_nam.geometry.intersects(kpi_circle_5179)].copy()  # KPI 반경과 교차하는 격자만
total_pop = float(gdf_in["pop"].sum())  # KPI 반경 내 총 인구
unc_pop = float(gdf_in.loc[gdf_in["is_uncovered"] == True, "pop"].sum())  # KPI 반경 내 비커버 인구
unc_rate = (unc_pop / total_pop) if total_pop > 0 else 0.0  # 비커버 비율(0으로 나눔 방지)

c1, c2, c3, c4, c5 = st.columns(5)  # KPI 카드 5개 컬럼 생성
c1.metric("선택 gid", str(sel_gid))  # 선택 gid 표시
c2.metric("KPI 반경 내 격자 수", f"{len(gdf_in):,}")  # 격자 수 표시
c3.metric("총 인구", f"{total_pop:,.0f}")  # 총 인구 표시
c4.metric("비커버 인구", f"{unc_pop:,.0f}")  # 비커버 인구 표시
c5.metric("비커버 비율", f"{unc_rate*100:.2f}%")  # 비커버 비율 표시

# =========================================================
# 4) 좌(Pydeck) / 우(Folium)  # 섹션 설명 주석
# =========================================================

st.markdown("---")  # 구분선
left, right = st.columns([1, 1], gap="large")  # 좌/우 레이아웃 구성

# =========================================================
# 4-A) LEFT: Pydeck  # 섹션 설명 주석
# - 비커버 격자만 회색/컬러로 보이도록 구성  # 설명 주석
# - 막대(3D) 색상은 노랑->빨강 그라데이션  # 설명 주석
# =========================================================

with left:  # 좌측 컬럼 시작
    st.subheader("인구기반 따릉이 신규 정류소 배치")  # 좌측 제목

    gdf_ll = gdf_in.to_crs(MAP_CRS).copy()  # KPI 내부 격자를 4326으로 변환
    gdf_ll = gdf_ll[gdf_ll["is_uncovered"] == True].copy()  # 비커버 격자만 남김(요구사항)

    pop = gdf_ll["pop"].clip(lower=0).astype(float)  # pop 음수 제거 + float 보정
    cap = float(pop.quantile(0.995)) if len(pop) > 0 else 0.0  # 극단치 상한(cap) 계산
    pop_c = np.minimum(pop, cap) if cap > 0 else pop  # cap 적용(pop_c)
    gdf_ll["elev"] = (np.power(pop_c, 1.80) * 0.02).astype(float)  # 3D 높이(elevation) 계산

    denom = float(pop_c.max() - pop_c.min()) if len(pop_c) > 0 else 0.0  # 정규화 분모
    t = ((pop_c - float(pop_c.min())) / denom).clip(0, 1) if denom > 0 else pd.Series([0.0] * len(pop_c), index=gdf_ll.index)  # 0~1 정규화
    gdf_ll["r"] = 255  # 빨강 채널 고정
    gdf_ll["g"] = (255 * (1.0 - t)).round().astype(int)  # 초록 채널은 t가 커질수록 감소(노랑->빨강)
    gdf_ll["b"] = 0  # 파랑 채널 0 고정
    gdf_ll["a"] = 190  # 알파(투명도) 고정

    grid_records = []  # pydeck PolygonLayer에 넣을 레코드 리스트
    for gid, popv, elev, rr, gg, bb, aa, geom in zip(  # 필요한 값을 한 번에 순회
        gdf_ll[GRID_ID_COL].astype(str).tolist(),  # gid 리스트
        gdf_ll["pop"].tolist(),  # pop 리스트
        gdf_ll["elev"].tolist(),  # elev 리스트
        gdf_ll["r"].tolist(),  # r 리스트
        gdf_ll["g"].tolist(),  # g 리스트
        gdf_ll["b"].tolist(),  # b 리스트
        gdf_ll["a"].tolist(),  # a 리스트
        gdf_ll.geometry.tolist(),  # geometry 리스트
    ):
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)  # Polygon/MultiPolygon 처리
        for poly in polys:  # 폴리곤 단위로 레코드 생성
            grid_records.append({  # 레코드 추가
                "gid": gid,  # gid 저장
                "pop": float(popv),  # pop 저장
                "elev": float(elev),  # elev 저장
                "color": [int(rr), int(gg), int(bb), int(aa)],  # RGBA 저장
                "polygon": list(map(list, poly.exterior.coords)),  # exterior 좌표를 [lon,lat] 리스트로
            })  # 레코드 끝

    center_ll = gpd.GeoSeries([sel_center_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 중심점(4326)
    kpi_circle_ll = gpd.GeoSeries([kpi_circle_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # KPI 원(4326)
    station_buf_ll = gpd.GeoSeries([station_buffer_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 신규 커버 원(4326)

    admin_geojson = {  # 남현동 경계 GeoJSON 구성
        "type": "FeatureCollection",  # GeoJSON 타입
        "features": [  # feature 목록
            {"type": "Feature", "properties": {"name": "남현동"}, "geometry": mapping(geom)}  # geometry를 GeoJSON으로
            for geom in gdf_namhyeon.to_crs(MAP_CRS).geometry.tolist()  # 남현동 폴리곤 순회
        ],  # features 끝
    }  # admin_geojson 끝

    layer_admin = pdk.Layer(  # 남현동 경계 레이어
        "GeoJsonLayer",  # 레이어 타입
        data=admin_geojson,  # 데이터
        stroked=True,  # 외곽선 표시
        filled=False,  # 내부 채우기 없음
        get_line_color=[0, 0, 0, 230],  # 선 색상(검정)
        line_width_min_pixels=2,  # 최소 두께
        pickable=False,  # 클릭 비활성
    )  # layer_admin 끝

    layer_grid = pdk.Layer(  # 비커버 격자 3D 레이어
        "PolygonLayer",  # 레이어 타입
        data=grid_records,  # 레코드
        get_polygon="polygon",  # 폴리곤 좌표 키
        extruded=True,  # 3D extrusion 켜기
        filled=True,  # 채우기
        stroked=False,  # 외곽선 끄기
        get_elevation="elev",  # 높이 컬럼
        elevation_scale=1,  # 스케일
        get_fill_color="color",  # 색상 컬럼(노랑->빨강)
        pickable=True,  # 툴팁 표시를 위해 pickable
    )  # layer_grid 끝

    layer_kpi = pdk.Layer(  # KPI 원 레이어
        "PolygonLayer",  # 레이어 타입
        data=[{"polygon": list(map(list, kpi_circle_ll.exterior.coords))}],  # 원 exterior 좌표
        get_polygon="polygon",  # 키
        filled=False,  # 채우기 없음
        stroked=True,  # 외곽선만
        get_line_color=[30, 30, 30, 220],  # 선 색
        line_width_min_pixels=2,  # 선 두께
        pickable=False,  # 클릭 비활성
    )  # layer_kpi 끝

    layer_station_buf = pdk.Layer(  # 신규 커버 반경 원 레이어
        "PolygonLayer",  # 레이어 타입
        data=[{"polygon": list(map(list, station_buf_ll.exterior.coords))}],  # 원 좌표
        get_polygon="polygon",  # 키
        filled=False,  # 채우기 없음
        stroked=True,  # 외곽선
        get_line_color=[0, 120, 0, 220],  # 초록 선
        line_width_min_pixels=2,  # 두께
        pickable=False,  # 클릭 비활성
    )  # layer_station_buf 끝

    layer_station = pdk.Layer(  # 신규 정류장(점) 레이어
        "ScatterplotLayer",  # 레이어 타입
        data=[{"lon": float(center_ll.x), "lat": float(center_ll.y)}],  # 점 데이터
        get_position="[lon, lat]",  # 좌표 키
        get_radius=70,  # 점 반경
        pickable=True,  # 툴팁 표시 가능
    )  # layer_station 끝

    view = pdk.ViewState(  # 카메라 시점
        latitude=float(center_ll.y),  # 위도
        longitude=float(center_ll.x),  # 경도
        zoom=14,  # 줌
        pitch=55,  # 기울기
        bearing=20,  # 회전
    )  # view 끝

    map_style = CARTO_POSITRON_GL if not MAPBOX_TOKEN else "mapbox://styles/mapbox/light-v11"  # 스타일 선택

    deck = pdk.Deck(  # pydeck 데크 구성
        layers=[layer_admin, layer_grid, layer_kpi, layer_station_buf, layer_station],  # 레이어 목록
        initial_view_state=view,  # 초기 시점
        map_style=map_style,  # 지도 스타일
        tooltip={"text": "gid: {gid}\npop: {pop}"},  # 툴팁 텍스트
    )  # deck 끝

    st.pydeck_chart(deck, height=MAP_HEIGHT_PX, width="stretch")  # 좌측 지도 출력

# =========================================================
# 4-B) RIGHT: Folium  # 섹션 설명 주석
# - 버스정류장 아이콘 표시  # 설명 주석
# - 신규 경로: 반경 내 버스정류장이 있으면 + 5분 내 도달 가능한 정류장까지 경로만 표시  #  요구사항 반영
# - 기존 경로: routes_all.gpkg 모든 레이어/칼럼을 합쳐 "기존 커버 경로" 1개만 표시  #  요구사항 반영
# =========================================================

with right:  # 우측 컬럼 시작
    st.subheader("신규 정류소 배치에 따른 커버리지 분석")  # 우측 제목

    center_ll = gpd.GeoSeries([sel_center_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 중심점 4326 변환
    lon, lat = float(center_ll.x), float(center_ll.y)  # folium 중심 좌표(lon/lat)

    # =========================================================
    # 1) Folium 지도 기본 틀(경계/비커버/신규정류소/반경/버스정류장)  # 섹션 설명 주석
    # =========================================================

    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="cartodbpositron")  # Folium 지도 생성(토큰 불필요)

    folium.GeoJson(  # 남현동 경계 폴리곤 레이어
        gdf_namhyeon.to_crs(MAP_CRS),  # 4326 변환
        name="남현동 경계",  # 레이어명
        style_function=lambda x: {"color": "#000000", "weight": 3, "fillOpacity": 0.02},  # 스타일
    ).add_to(m)  # 지도에 추가

    if len(gdf_unc) > 0:  # 비커버 데이터가 있으면
        unc_ll = gpd.GeoSeries(  # 남현동 내부 비커버만 남기기 위한 GeoSeries
            [gdf_unc.geometry.union_all().intersection(nam_union)],  # 남현동과 교집합만
            crs=TARGET_CRS,  # 분석 CRS
        ).to_crs(MAP_CRS).iloc[0]  # 4326 변환
        folium.GeoJson(  # 비커버 레이어 추가
            {"type": "Feature", "properties": {}, "geometry": unc_ll.__geo_interface__},  # GeoJSON
            name="비커버(남현동)",  # 레이어명
            style_function=lambda x: {"color": "#ff0000", "weight": 2, "fillOpacity": 0.10},  # 스타일(빨강)
        ).add_to(m)  # 지도에 추가

    folium.Marker(  # 신규 따릉이 정류장 마커
        location=[lat, lon],  # 위치
        tooltip=f"신규 따릉이 정류장(가정): gid={sel_gid}",  # 툴팁
        icon=folium.Icon(color="green", icon="bicycle", prefix="fa"),  # 아이콘(따릉이)
    ).add_to(m)  # 지도에 추가

    station_buf_ll = gpd.GeoSeries([station_buffer_5179], crs=TARGET_CRS).to_crs(MAP_CRS).iloc[0]  # 신규 커버 반경(4326)
    folium.GeoJson(  # 신규 커버 반경 레이어
        {"type": "Feature", "properties": {}, "geometry": station_buf_ll.__geo_interface__},  # GeoJSON
        name="신규 커버 반경",  # 레이어명
        style_function=lambda x: {"color": "#00aa00", "weight": 2, "fillOpacity": 0.03},  # 스타일(초록)
    ).add_to(m)  # 지도에 추가

    #  버스정류장 마커(남현동 내부 전체)  # 섹션 설명 주석
    gdf_bus_nam_ll = gdf_bus_nam.to_crs(MAP_CRS).copy()  # 남현동 버스정류장을 4326으로 변환
    for ars, nm, typ, geom in zip(  # 버스정류장을 순회
        gdf_bus_nam_ll["ARS_ID"].tolist(),  # 정류장 ID
        gdf_bus_nam_ll["정류소명"].tolist(),  # 정류소명
        gdf_bus_nam_ll["정류소타입"].tolist(),  # 정류소타입
        gdf_bus_nam_ll.geometry.tolist(),  # 포인트 geometry
    ):
        folium.Marker(  # 버스정류장 마커
            location=[float(geom.y), float(geom.x)],  # 위도/경도
            tooltip=f"버스정류장 | ARS_ID={ars} | {nm} | {typ}",  # 툴팁 텍스트
            icon=folium.Icon(color="blue", icon="bus", prefix="fa"),  # 버스 아이콘
        ).add_to(m)  # 지도에 추가

    # =========================================================
    # 2)  신규 경로(최종 요구사항 반영)  # 섹션 설명 주석
    # - "반경 내 버스정류장"이 있을 때만 경로를 만든다  # 설명 주석
    # - 반경 내 정류장들 각각에 대해  # 설명 주석
    #   신규 따릉이 정류소 → 버스정류장 최단경로들을 표시한다  # 설명 주석
    # - 반경 내 정류장이 없으면 신규 경로는 표시하지 않는다  # 설명 주석
    # =========================================================

    gdf_bus_in_radius = gdf_bus_nam[gdf_bus_nam.geometry.intersects(station_buffer_5179)].copy()  #  반경(5179) 내 버스정류장만 추출

    if len(gdf_bus_in_radius) == 0:  # 반경 내 버스정류장이 없으면
        pass  #  신규 경로는 표시하지 않음(요구사항)
    else:  # 반경 내 버스정류장이 있으면
        try:  # 네트워크/경로 계산 전체를 try로 감싸 지도 표시 자체는 유지
            ox.settings.log_console = False  # OSMnx 로그 끄기(콘솔 출력 최소화)

            #  신규 따릉이 정류소(지도 중심점) 기준으로 도보 네트워크 그래프 다운로드  # 설명 주석
            G = ox.graph_from_point(  # OSM 그래프 생성
                (lat, lon),  # 중심점(lat, lon)
                dist=int(GRAPH_DIST_M),  # 다운로드 반경(미터)
                network_type="walk",  # 도보 네트워크
                simplify=True,  # 그래프 단순화
            )  # G 생성 끝

            #  엣지 길이(length) 속성 추가(OSMnx 버전별 API 대응)  # 설명 주석
            try:  # OSMnx 2.x
                G = ox.distance.add_edge_lengths(G)  # length 추가
            except Exception:  # OSMnx 1.x
                G = ox.add_edge_lengths(G)  # length 추가

            #  그래프를 미터 CRS로 투영(거리/시간 계산 안정화)  # 설명 주석
            Gp = ox.project_graph(G)  # 투영 그래프

            #  신규 따릉이 정류소(중심점)를 그래프 CRS로 변환한 뒤 최근접 노드 추출  # 설명 주석
            pt_proj = gpd.GeoSeries([Point(lon, lat)], crs=MAP_CRS).to_crs(Gp.graph["crs"]).iloc[0]  # 중심점을 그래프 CRS로
            px, py = float(pt_proj.x), float(pt_proj.y)  # 투영 좌표
            source_node = ox.distance.nearest_nodes(Gp, X=px, Y=py)  # 출발 노드(최근접)

            #  최단경로 가중치로 쓸 travel_time(초)을 모든 엣지에 부여  # 설명 주석
            for u, v, k, data in Gp.edges(keys=True, data=True):  # 모든 엣지 순회
                data["travel_time"] = float(data.get("length", 0.0)) / float(WALK_SPEED_MPS)  # time(s)=length(m)/speed(m/s)

            #  반경 내 버스정류장을 그래프 CRS로 투영 후, 각 정류장 최근접 노드 추출  # 설명 주석
            bus_pts_proj = gdf_bus_in_radius.to_crs(Gp.graph["crs"]).copy()  # 정류장 투영
            bus_nodes = []  # 정류장 노드 리스트
            for p in bus_pts_proj.geometry.tolist():  # 정류장 포인트 순회
                bn = ox.distance.nearest_nodes(Gp, X=float(p.x), Y=float(p.y))  # 정류장 최근접 노드
                bus_nodes.append(int(bn))  # 정수로 저장

            #  신규 따릉이 정류소 → 반경 내 버스정류장 최단경로를 모두 생성  # 설명 주석
            link_features = []  # 경로 feature 리스트
            max_routes = 80  # 데모 렌더링 성능을 위한 상한(원하면 늘려도 됨)

            for bn in bus_nodes[:max_routes]:  # 반경 내 정류장 노드 순회(상한 적용)
                try:  # 개별 경로 실패가 전체 실패로 번지지 않게 방어
                    route = nx.shortest_path(Gp, int(source_node), int(bn), weight="travel_time")  # travel_time 기준 최단경로(노드열)
                    line = LineString([(float(Gp.nodes[n]["x"]), float(Gp.nodes[n]["y"])) for n in route])  # 노드열→LineString
                    line_ll = gpd.GeoSeries([line], crs=Gp.graph["crs"]).to_crs(MAP_CRS).iloc[0]  # 표시용 4326 변환
                    link_features.append({"type": "Feature", "properties": {}, "geometry": line_ll.__geo_interface__})  # feature 추가
                except Exception:  # 연결 불가(그래프 단절 등) 케이스
                    continue  # 해당 정류장은 스킵

            if len(link_features) > 0:  # 경로가 하나라도 있으면
                folium.GeoJson(  # 신규→버스정류장 연결경로 레이어
                    {"type": "FeatureCollection", "features": link_features},  # GeoJSON FeatureCollection
                    name="신규→반경내 버스정류장 최단경로",  # 레이어명
                    style_function=lambda x: {"color": "#ff9900", "weight": 4, "opacity": 0.85},  # 스타일(주황)
                ).add_to(m)  # 지도에 추가
            else:  # 반경 내 정류장은 있지만 경로가 0개면
                pass  # 표시할 경로가 없으므로 아무 것도 추가하지 않음

        except Exception as e:  # 네트워크/라우팅 실패 시
            st.warning(f"OSMnx/NetworkX 경로 계산이 실패했습니다. (지도는 정상 표시) | 에러: {type(e).__name__}: {e}")  # 경고 출력

    # =========================================================
    # 3)  기존 경로(routes_all.gpkg)  # 섹션 설명 주석
    # - 모든 레이어/칼럼을 읽어도 최종은 geometry만 남겨서  # 설명 주석
    # - '기존 커버 경로' 1개 레이어로만 표시  # 설명 주석
    # =========================================================

    gdf_routes_all = None  # 합쳐진 결과를 담을 변수

    if os.path.exists(ROUTES_ALL_GPKG):  # 파일이 있으면
        try:  # 레이어 목록 확인 시도
            layers = fiona.listlayers(ROUTES_ALL_GPKG)  # 레이어 이름 리스트
        except Exception:  # 실패하면(환경에 따라 fiona listlayers 실패 가능)
            layers = [None]  # 레이어 미지정으로 한 번만 읽기

        gdfs = []  # 레이어별 GeoDataFrame 담기
        for lyr in layers:  # 레이어를 순회
            try:  # 레이어별 로드 실패 방어
                gdf_tmp = gpd.read_file(ROUTES_ALL_GPKG, layer=lyr) if lyr else gpd.read_file(ROUTES_ALL_GPKG)  # 레이어별 로드
                if gdf_tmp is None or len(gdf_tmp) == 0:  # 비어있으면
                    continue  # 스킵
                if gdf_tmp.crs is None:  # CRS가 없으면
                    gdf_tmp = gdf_tmp.set_crs(TARGET_CRS)  # 5179로 가정(데모 기준)
                gdf_tmp = gdf_tmp[["geometry"]].copy()  #  geometry만 남김(스키마 충돌 제거)
                gdfs.append(gdf_tmp)  # 리스트에 추가
            except Exception:  # 읽기 실패하면
                continue  # 다음 레이어로

        if len(gdfs) > 0:  # 하나라도 있으면
            gdf_routes_all = pd.concat(gdfs, ignore_index=True)  # 모두 합치기
            gdf_routes_all = gpd.GeoDataFrame(gdf_routes_all, geometry="geometry", crs=gdfs[0].crs)  # GeoDataFrame 보정
            gdf_routes_all = gdf_routes_all.to_crs(MAP_CRS)  # 4326 변환(표시용)

            folium.GeoJson(  # 기존 커버 경로 레이어(1개만)
                gdf_routes_all,  # 합쳐진 경로
                name="기존 커버 경로",  # 레이어명
                style_function=lambda x: {"color": "#7a7a7a", "weight": 5, "opacity": 0.70},  # 스타일(회색)
            ).add_to(m)  # 지도에 추가
    else:  # 파일이 없으면
        st.info("routes_all.gpkg 파일이 없어 기존 커버 경로를 표시하지 않습니다.")  # 안내 출력

    # =========================================================
    # 4) 레이어 컨트롤 + 출력  # 섹션 설명 주석
    # =========================================================

    folium.LayerControl(collapsed=False).add_to(m)  # 레이어 컨트롤 추가(펼친 상태)
    st_folium(m, width=None, height=MAP_HEIGHT_PX, key=f"folium_{sel_gid}")  # Folium 지도 출력(우측)


