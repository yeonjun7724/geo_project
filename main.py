# app.py
# ------------------------------------------------------------
# Streamlit 지도 시각화 튜토리얼 (함수 없이, 스텝바이스텝 스크립트)
# - STEP 1: 데이터 로드 & 미리보기
# - STEP 2: 산점 지도
# - STEP 3: 성능 제어(표본·반경·불투명도)
# - STEP 4: 히트맵 전환
# - STEP 5: 경로(PathLayer, Mapbox Directions로 실제 도로 경로)
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import json, os, requests

# 0) 페이지·레이아웃 기본 설정 --------------------------------
st.set_page_config(page_title="Streamlit 지도 시각화 — Step-by-Step", layout="wide")

# 0-1) (선택) Mapbox Directions 토큰: 경로 단계(STEP 5)에서만 사용
#      OSM 배경지도는 토큰 없이 동작합니다.
MAPBOX_TOKEN = "pk.eyJ1Ijoia2lteWVvbmp1biIsImEiOiJjbWVnZHNyZmsxMTVpMmtzZzMzMTU5ZGFyIn0.esI42zH2s8c_Dy26yj4uHw"

# 1) 사이드바: 수업 단계 선택 ----------------------------------
st.sidebar.markdown("### 강의 단계 (STEP)")
STEP = st.sidebar.selectbox(
    "시연 단계",
    ["1) 데이터 로드 & 미리보기",
     "2) 산점 지도",
     "3) 성능 제어(표본·반경·불투명도)",
     "4) 히트맵 전환",
     "5) 경로(PathLayer, Mapbox Directions)"],
    index=0
)

# 2) 사이드바: 데이터 업로드 또는 서울 25개 자치구 샘플 ---------
upl = st.sidebar.file_uploader("CSV / JSON / GEOJSON 업로드", type=["csv","json","geojson"])
use_seoul_sample = st.sidebar.checkbox("서울 25개 자치구 샘플(point_sample.json) 사용", value=(upl is None))

# 3) 데이터 준비 (업로드 > 샘플JSON > 임시 난수) -----------------
if upl is not None:
    name = upl.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(upl)
    else:
        # JSON/NDJSON/GeoJSON 포괄 처리
        try:
            df = pd.read_json(upl, lines=False)
        except ValueError:
            upl.seek(0)
            df = pd.read_json(upl, lines=True)
        except Exception:
            upl.seek(0)
            data = json.load(upl)
            if isinstance(data, dict) and "features" in data:
                rows = []
                for feat in data["features"]:
                    geom = feat.get("geometry", {})
                    props = feat.get("properties", {}) or {}
                    if geom.get("type") == "Point":
                        lon, lat = geom.get("coordinates", [None, None])
                        rows.append({"lat": lat, "lon": lon, **props})
                df = pd.DataFrame(rows)
            else:
                df = pd.DataFrame(data)
elif use_seoul_sample:
    # 실행 폴더에 point_sample.json(서울 25개 자치구 중심점)이 있으면 사용
    try:
        with open("point_sample.json", "r", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))
    except:
        # 샘플 JSON이 없다면 임시 난수로 대체(데모용)
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "lat": 37.55 + 0.03*rng.standard_normal(200),
            "lon": 126.98 + 0.04*rng.standard_normal(200),
            "weight": rng.integers(1, 5, 200),
            "label": "random"
        })
else:
    # 기존 CSV 샘플 파일이 있으면 사용
    try:
        df = pd.read_csv("points_sample.csv")
    except:
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "lat": 37.55 + 0.1*rng.random(1000),
            "lon": 126.97 + 0.1*rng.random(1000),
            "weight": rng.integers(1, 5, 1000),
            "label": "random"
        })

# 4) 열 표준화 및 좌표 유효성 체크 ------------------------------
#    다양한 컬럼명을 'lat','lon'으로 통일하고 잘못된 좌표 제거
df = df.rename(columns={c.lower(): "lat" if c.lower() in ["lat","latitude","위도"]
                                   else "lon" if c.lower() in ["lon","lng","longitude","경도"]
                                   else c for c in df.columns})
df = df.dropna(subset=["lat","lon"])
df = df[(df["lat"].between(-90,90)) & (df["lon"].between(-180,180))]

# 5) 타이틀 및 데이터 유효성 ------------------------------------
st.title("Streamlit 지도 시각화 — Step-by-Step")
if len(df) == 0:
    st.error("표시할 점 데이터가 없습니다.")
    st.stop()

# 6) 지도 초기 뷰포트(중앙/줌) ----------------------------------
mid_lat, mid_lon = df["lat"].median(), df["lon"].median()
zoom_guess = 11 if df["lat"].std() < 0.2 and df["lon"].std() < 0.2 else 9
view = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=zoom_guess, bearing=0, pitch=0)

# 7) (항상 켜는) OSM 배경지도 -----------------------------------
#    토큰 불필요. 모든 단계에서 첫 레이어로 추가하여 백그라운드 고정.
osm = pdk.Layer(
    "TileLayer",
    data="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    minZoom=0, maxZoom=19, tileSize=256, opacity=1.0
)

if STEP.startswith("1)"):
    st.subheader("1. 데이토 로드 & 불러오기")
    st.write("아래 테이블의 필수 열은 'lat', 'lon' 입니다. (서울 샘플이면 'gu', 'name' 컬럼이 함께 존재)")
    st.dataframe(df.head(25), use_container_width=True)
    st.stop()

# 9) 공통: (있으면) 카테고리/구 선택 필터 -----------------------
right = st.sidebar
if "gu" in df.columns:
    cats = ["<전체>"] + df["gu"].astype(str).unique().tolist()
    sel = right.selectbox("자치구(선택)", cats, index=0)
else:
    sel = "<전체>"

df_view = df.copy()
if sel != "<전체>" and "gu" in df_view.columns:
    df_view = df_view[df_view["gu"].astype(str) == sel]

if STEP.startswith("2)"):
    st.subheader("2. 산점지도")
    point_tooltip = {"text": "📍 {gu} {name}\n(lat: {lat}, lon: {lon})"} if "gu" in df.columns else {"text": "lat: {lat}\nlon: {lon}"}
    scatter = pdk.Layer(
        "ScatterplotLayer",
        data = df_view,
        get_position = '[lon, lat]',
        get_radius = 80 if 'gu' in df.columns else 40,
        get_fill_color = [255, 140, 0, 200],
        pickable = True 
    )

    st.pydeck_chart(
        pdk.Deck(
            layers = [osm, scatter],
            initial_view_state = view,
            tooltip = point_tooltip
        ),
        use_container_width = True
    )

    st.info("설명: 가장 기본적인 점의 표현입니다. 마우스를 올리면 포인트 상세(구, 이름)가 보입니다.")
    st.stop()

max_n = min(20000, len(df_view))
sample_n = right.slider("표본 수", 25, max(200, max_n), min(500, max_n), step=25)
radius = right.slider("점 반경(px)", 5, 200, 60 if 'gu' in df.columns else 40, step=5)
opacity = right.slider("점 불투명도(%)", 10, 100, 80, step=5) / 100

df_view2 = df_view.sample(sample_n, random_state=42) if len(df_view) > sample_n else df_view

if STEP.startswith("3)"):
    st.subheader("3. 성능 제어(표본, 반경, 불투명도)")
    point_tooltip = {"text": "📍 {gu} {name}\n(lat: {lat}, lon: {lon})"} if "gu" in df.columns else {"text": "lat: {lat}\nlon: {lon}"}

    scatter = pdk.Layer(
        "ScatterplotLayer",
        data = df_view2,
        get_position = '[lon, lat]',
        get_radius = radius,
        get_fill_color = [255, 140, 0, int(255*opacity)],
        pickable = True 
    )

    st.pydeck_chart(pdk.Deck(
        layers = [osm, scatter], 
        initial_view_state = view,
        tooltip = point_tooltip
    ),
    use_container_width = True
    )

    st.info("설명: 대용량일수록 먼저 표본을 줄이고, 이후 반경, 투명도를 조절해 가독성과 성능을 균형 있게 맞춥니다.")
    st.stop()

if STEP.startswith("4)"):
    st.subheader("4. 히트맵 전환")
    df_heat = df_view2.assign(_w = df_view2["weight"] if "weight" in df_view2.columns else 1)   
    heat = pdk.Layer(
        "HeatmapLayer",
        data = df_heat,
        get_position = '[lon, lat]',
        get_weight = "_w",
        radiusPixels = radius
    )

    st.pydeck_chart(pdk.Deck(
        layers = [osm, heat],
        initial_view_state = view,
    ),
    use_container_width = True
    )

    st.info("설명: 밀도가 높은 영역을 한눈에 확인하려면 히트맵이 효과적입니다. 반경을 키우면 더 부드러운 분포가 됩니다")
    st.stop()

# 13) STEP 5 — 경로(PathLayer, Mapbox Directions) ----------------
if STEP.startswith("5)"):
    st.subheader("⑤ 경로(PathLayer, Mapbox Directions)")
    st.write("**사용법**: 출발지 → (경유지들) → 도착지 순서로 포인트를 선택하세요. Mapbox Directions가 실제 도로 경로를 그립니다.")

    # 13-1) 선택 UI: 최대 5개 포인트(출발/경유/도착)
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index":"id"})
    # 라벨 구성: id + (구/이름) 표시
    label_col = "gu" if "gu" in df.columns else ("name" if "name" in df.columns else None)
    display = df["id"].astype(str) + ((" — " + df[label_col].astype(str)) if label_col else "")
    selected = st.multiselect("포인트 선택(순서 중요 — 출발→경유→도착, 최대 5개)", display.tolist(), max_selections=5)

    # 13-2) 스타일 UI
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        width_px = st.slider("경로 두께(px)", 2, 12, 6)
    with col2:
        color_hex = st.color_picker("경로 색상", "#0066FF")
    with col3:
        opacity_px = st.slider("경로 불투명도(%)", 30, 100, 90, step=5)

    # HEX → RGBA
    hx = color_hex.lstrip("#")
    rgb = [int(hx[i:i+2], 16) for i in (0,2,4)]
    rgba = [rgb[0], rgb[1], rgb[2], int(255*(opacity_px/100))]

    # 13-3) 항상 점(배경) + OSM 타일 추가
    layers = [osm]
    point_tooltip = {"text": "📍 {gu} {name}\n(lat: {lat}, lon: {lon})"} if "gu" in df.columns else {"text": "lat: {lat}\nlon: {lon}"}
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=df_view2,
            get_position='[lon, lat]',
            get_radius=60,
            get_fill_color=[255,140,0,130],
            pickable=True
        )
    )

    # 13-4) 경로 생성(선택 + 토큰 필요)
    if selected:
        if MAPBOX_TOKEN.endswith("_입력"):
            st.warning("경로를 보려면 MAPBOX_TOKEN을 본인 키로 교체하세요.")
        else:
            # 선택된 id 순서대로 좌표/이름 추출
            sel_ids = [int(s.split(" — ")[0]) for s in selected]
            coords = df.set_index("id").loc[sel_ids, ["lon","lat"]].to_numpy().tolist()
            names  = df.set_index("id").loc[sel_ids, ["gu","name"]].fillna("").astype(str).agg(" ".join, axis=1).tolist() \
                     if ("gu" in df.columns or "name" in df.columns) else [str(i) for i in sel_ids]

            # 라벨: "출발 → 도착" (경유지 있으면 'via N')
            start_label = names[0]
            end_label   = names[-1]
            via_cnt     = max(0, len(names)-2)
            route_name  = f"{start_label} → {end_label}" + (f" (via {via_cnt})" if via_cnt>0 else "")

            # Mapbox Directions 호출(실제 도로 경로)
            coord_str = ";".join([f"{c[0]},{c[1]}" for c in coords])
            url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{coord_str}"
            params = {"geometries":"geojson","overview":"full","access_token":MAPBOX_TOKEN}

            try:
                r = requests.get(url, params=params, timeout=10)
                data = r.json()
                if "routes" in data and data["routes"]:
                    line = data["routes"][0]["geometry"]["coordinates"]  # [[lon,lat], ...]
                    # PathLayer에 라벨을 박아 툴팁에서 "어디 → 어디" 확인 가능
                    route_df = pd.DataFrame([{"path_coords": line, "route_name": route_name}])
                    layers.append(
                        pdk.Layer(
                            "PathLayer",
                            data=route_df,
                            get_path="path_coords",
                            get_color=rgba,
                            width_scale=1,
                            width_min_pixels=width_px,
                            width_max_pixels=width_px,
                            pickable=True
                        )
                    )
                    # 경로 툴팁: 어디→어디
                    route_tooltip = {"text": "🛣️ {route_name}"}
                    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip=route_tooltip), use_container_width=True)
                    st.success(f"경로 표시: {route_name}")
                else:
                    st.warning("Mapbox에서 경로를 찾지 못했습니다. 좌표가 너무 가깝거나 도로 연결이 없을 수 있어요.")
            except Exception as e:
                st.error(f"Directions 호출 오류: {e}")
            st.stop()

    # 선택이 없으면 기본 지도만 표시
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip=point_tooltip), use_container_width=True)
    st.info("설명: 포인트를 2개 이상 선택하면 '출발 → 도착' 경로가 그려지고, 경유지를 추가하면 (via N)로 표시됩니다. 경로에 마우스를 올리면 라벨이 툴팁으로 뜹니다.")
