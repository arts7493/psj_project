from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import streamlit as st


APP_TITLE = "광주·전남 AI 미식 관광 대시보드"
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "places.csv"
REQUIRED_COLUMNS = ("카테고리", "이름", "주소")
CATEGORY_ORDER = ("맛집", "카페", "관광명소", "문화공간")


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_places(csv_path: str, modified_time_ns: int) -> pd.DataFrame:
    """장소 CSV를 읽고 화면 필터링에 필요한 지역 컬럼을 만든다.

    modified_time_ns는 CSV를 수정한 뒤 다시 불러올 때 캐시를 갱신하기 위한 값이다.
    """
    del modified_time_ns

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            data = pd.read_csv(csv_path, encoding=encoding, dtype=str)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise ValueError("CSV 문자 인코딩을 확인할 수 없습니다.") from last_error

    data.columns = [str(column).replace("\ufeff", "").strip() for column in data.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"CSV에 필수 컬럼이 없습니다: {missing_text}")

    for column in REQUIRED_COLUMNS:
        data[column] = data[column].fillna("").astype(str).str.strip()

    data = data[(data["이름"] != "") & (data["주소"] != "")].copy()
    data = data.drop_duplicates(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)

    # 대회 범위는 광주광역시와 전라남도이므로 전북 데이터는 화면에서 제외한다.
    out_of_scope = data["주소"].str.contains(r"전라북도|순창군", regex=True, na=False)
    data = data.loc[~out_of_scope].copy()

    data["지역"] = data["주소"].map(extract_region)
    data["카테고리"] = pd.Categorical(
        data["카테고리"],
        categories=list(CATEGORY_ORDER),
        ordered=True,
    )

    return data.sort_values(
        by=["지역", "카테고리", "이름"],
        kind="stable",
    ).reset_index(drop=True)


def extract_region(address: str) -> str:
    """주소 문자열에서 광주 구 또는 전남 시·군 단위 지역명을 추출한다."""
    normalized = re.sub(r"\s+", " ", str(address).strip())
    tokens = normalized.split(" ")

    if normalized.startswith("광주광역시"):
        district = next((token for token in tokens[1:] if token.endswith(("구", "군"))), "")
        return f"광주 {district}".strip()

    # 수집 CSV에 사용된 '전남광주 동구 ...' 형식도 광주로 인식한다.
    if normalized.startswith(("전남광주", "광주 ")):
        district = next((token for token in tokens if token.endswith(("구", "군"))), "")
        return f"광주 {district}".strip()

    if normalized.startswith(("전라남도", "전남 ")):
        municipality = next((token for token in tokens[1:] if token.endswith(("시", "군"))), "")
        return municipality or "전남"

    return "지역 확인 필요"


def apply_filters(
    data: pd.DataFrame,
    regions: list[str],
    categories: list[str],
    keyword: str,
) -> pd.DataFrame:
    """사이드바에서 선택한 조건에 맞춰 장소 목록을 필터링한다."""
    filtered = data[
        data["지역"].isin(regions)
        & data["카테고리"].astype(str).isin(categories)
    ].copy()

    normalized_keyword = keyword.strip()
    if normalized_keyword:
        search_text = (
            filtered["이름"].fillna("")
            + " "
            + filtered["주소"].fillna("")
            + " "
            + filtered["카테고리"].astype(str)
        )
        filtered = filtered[
            search_text.str.contains(normalized_keyword, case=False, regex=False, na=False)
        ]

    return filtered.reset_index(drop=True)


def render_header() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1180px;
                padding-top: 2rem;
                padding-bottom: 3rem;
            }
            .hero-box {
                padding: 1.6rem 1.8rem;
                border: 1px solid rgba(49, 51, 63, 0.12);
                border-radius: 18px;
                background: linear-gradient(135deg, rgba(248, 250, 252, 0.98), rgba(236, 253, 245, 0.92));
                margin-bottom: 1.2rem;
            }
            .hero-label {
                font-size: 0.82rem;
                font-weight: 700;
                letter-spacing: 0.06em;
                color: #047857;
                margin-bottom: 0.35rem;
            }
            .hero-title {
                font-size: clamp(1.8rem, 4vw, 2.7rem);
                font-weight: 800;
                line-height: 1.2;
                margin: 0;
                color: #111827;
            }
            .hero-description {
                margin: 0.8rem 0 0;
                color: #4b5563;
                line-height: 1.65;
            }
            div[data-testid="stMetric"] {
                border: 1px solid rgba(49, 51, 63, 0.10);
                border-radius: 14px;
                padding: 0.8rem 1rem;
                background: rgba(255, 255, 255, 0.72);
            }
        </style>
        <div class="hero-box">
            <div class="hero-label">광주·전남 체류형 미식 관광</div>
            <h1 class="hero-title">맛집의 기다림을 지역 여행으로 바꾸다</h1>
            <p class="hero-description">
                유명 맛집의 대기수요를 주변 관광명소·카페·문화공간으로 분산하기 위한
                AI 관광 코스 서비스의 1차 데이터 대시보드입니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    render_header()

    if not DATA_PATH.exists():
        st.error("장소 CSV 파일을 찾지 못했습니다.")
        st.code("data/places.csv")
        st.stop()

    try:
        places = load_places(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
    except Exception as exc:
        st.error("장소 데이터를 읽는 중 문제가 발생했습니다.")
        st.exception(exc)
        st.stop()

    region_options = sorted(
        places["지역"].dropna().astype(str).unique().tolist(),
        key=lambda value: (0 if value.startswith("광주") else 1, value),
    )
    available_categories = [
        category for category in CATEGORY_ORDER if category in places["카테고리"].astype(str).unique()
    ]

    with st.sidebar:
        st.header("탐색 조건")
        st.caption("CSV를 수정한 뒤 아래 버튼을 눌러 화면에 반영할 수 있습니다.")

        if st.button("CSV 다시 불러오기"):
            load_places.clear()
            st.rerun()

        selected_regions = st.multiselect(
            "지역",
            options=region_options,
            default=region_options,
        )
        selected_categories = st.multiselect(
            "카테고리",
            options=available_categories,
            default=available_categories,
        )
        keyword = st.text_input(
            "장소 검색",
            placeholder="예: 여수, 카페, 오동도",
        )

        st.divider()
        st.caption("현재 단계: CSV 연결 · 지역/카테고리 필터 · 장소 목록")

    filtered = apply_filters(
        places,
        selected_regions,
        selected_categories,
        keyword,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("등록 장소", f"{len(places)}곳")
    metric_columns[1].metric("검색 결과", f"{len(filtered)}곳")
    metric_columns[2].metric("표시 지역", f"{filtered['지역'].nunique()}개")
    metric_columns[3].metric("카테고리", f"{filtered['카테고리'].nunique()}개")

    st.subheader("장소 데이터")
    st.caption("주소에서 지역을 임시 추출해 필터에 사용합니다. 원본 CSV에는 지역 컬럼을 추가하지 않았습니다.")

    if filtered.empty:
        st.warning("선택한 조건에 맞는 장소가 없습니다. 지역·카테고리·검색어를 확인해 주세요.")
    else:
        display_data = filtered[["지역", "카테고리", "이름", "주소"]].copy()
        display_data["카테고리"] = display_data["카테고리"].astype(str)

        st.dataframe(
            display_data,
            width="stretch",
            height=520,
            hide_index=True,
            column_config={
                "지역": st.column_config.TextColumn("지역", width="small"),
                "카테고리": st.column_config.TextColumn("카테고리", width="small"),
                "이름": st.column_config.TextColumn("장소명", width="medium"),
                "주소": st.column_config.TextColumn("주소", width="large"),
            },
        )

    st.divider()
    st.info(
        "다음 단계에서 맛집 대기시간 입력과 주변 장소 추천 조건을 연결합니다. "
        "현재 버전에는 AI API·지도·QR·카메라 기능을 아직 넣지 않았습니다."
    )


if __name__ == "__main__":
    main()
