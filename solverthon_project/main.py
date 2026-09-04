from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, time, timedelta
from hashlib import sha1, sha256
from html import escape
from pathlib import Path
import math
import re
from textwrap import dedent
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from ai_client import generate_gemini_routes, get_gemini_status
from naver_map import (
    build_naver_search_url,
    clear_map_cache,
    estimate_walk,
    geocode,
    geocode_many,
    get_driving_route,
    get_naver_status,
    straight_distance_km,
)
from qr_utils import (
    DEMO_QR_CODES,
    QR_COOLDOWN_SECONDS,
    cooldown_remaining,
    decode_qr_image,
    get_demo_qr_info,
)
from place_images import attach_preview_images, preview_image_data_uri


APP_NAME = "운남화이팅"
APP_SUBTITLE = "전남광주 AI 로컬 미식 코스"
KST = ZoneInfo("Asia/Seoul")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "places.csv"
IMAGES_DIR = BASE_DIR / "images"
REQUIRED_COLUMNS = ("카테고리", "이름", "주소")

HOME = "🏠 홈"
COURSE = "🧭 코스"
CHECKIN = "📷 체크인"
MY = "👤 MY"
NAV_ITEMS = (HOME, COURSE, CHECKIN, MY)

CATEGORY_ICON = {
    "맛집": "🍽️",
    "관광명소": "🌿",
    "문화공간": "🎨",
    "카페": "☕",
    "출발지": "📍",
}

ROUTE_TYPES = (
    "가까운 동네",
    "취향 집중",
    "식사 전후",
)

COURSE_OPTIONS: dict[str, dict[str, int]] = {
    "2시간": {
        "minutes": 120,
        "soft_radius_km": 6,
        "hard_radius_km": 10,
        "places": 2,
        "stay": 20,
        "meal_stay": 50,
    },
    "3시간": {
        "minutes": 180,
        "soft_radius_km": 12,
        "hard_radius_km": 20,
        "places": 3,
        "stay": 25,
        "meal_stay": 60,
    },
    "반나절": {
        "minutes": 300,
        "soft_radius_km": 25,
        "hard_radius_km": 40,
        "places": 4,
        "stay": 30,
        "meal_stay": 60,
    },
    "하루": {
        "minutes": 480,
        "soft_radius_km": 45,
        "hard_radius_km": 65,
        "places": 4,
        "stay": 45,
        "meal_stay": 70,
    },
}

INTEREST_CATEGORY = {
    "자연·산책": "관광명소",
    "문화·전시": "문화공간",
    "카페·디저트": "카페",
    "로컬 감성": "관광명소",
}


st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------------
# 공통 도우미
# -----------------------------------------------------------------------------
def ui(markup: str) -> None:
    body = dedent(markup).strip()
    if hasattr(st, "html"):
        st.html(body)
    else:
        st.markdown(body, unsafe_allow_html=True)


def h(value: Any) -> str:
    return escape(str(value or "").strip())


def short_address(address: str) -> str:
    """CSV의 주소 표현을 임의로 바꾸지 않고 공백만 정리합니다."""
    return " ".join(str(address or "").split())

def optional_text(row: dict[str, Any] | pd.Series, *columns: str) -> str:
    for column in columns:
        value = row.get(column, "")
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def parse_minutes(value: Any, default: int) -> int:
    match = re.search(r"\d+", str(value or ""))
    if not match:
        return default
    return max(10, min(180, int(match.group())))


def format_minutes(value: int) -> str:
    hours, minutes = divmod(max(0, int(value)), 60)
    if hours and minutes:
        return f"{hours}시간 {minutes}분"
    if hours:
        return f"{hours}시간"
    return f"{minutes}분"


def format_clock(value: datetime | None) -> str:
    return value.strftime("%H:%M") if value else "-"


def extract_region(address: str) -> str:
    """
    주소에서 앱에 표시할 지역을 추출합니다.

    광주 주소는 대회 명칭에 맞춰 모두 '전남광주'로 표시하고,
    전남 주소는 시·군 단위로 표시합니다.
    """
    normalized = " ".join(str(address or "").split())

    if "전남광주" in normalized or "광주광역시" in normalized or normalized.startswith("광주 "):
        return "전남광주"

    match = re.search(r"(?:전라남도|전남)\s+([가-힣]+(?:시|군))", normalized)
    if match:
        return match.group(1)

    return "지역 확인 필요"

def extract_subregion(address: str, region: str) -> str:
    """광주는 구, 전남은 읍·면·동을 우선해 하위 지역을 추출합니다."""
    normalized = " ".join(str(address or "").split())

    if region == "전남광주":
        match = re.search(r"([가-힣]+구)", normalized)
        return match.group(1) if match else "전남광주"

    for token in normalized.split():
        if token.endswith(("읍", "면", "동")):
            return token

    return region

def region_sort_key(region: str) -> tuple[int, str]:
    if region == "전남광주":
        return 0, region
    if region == "지역 확인 필요":
        return 2, region
    return 1, region

def set_page(page: str) -> None:
    st.session_state.page = page


def clear_route_runtime() -> None:
    st.session_state.mobility_open = {}
    st.session_state.mobility_cache = {}
    st.session_state.naver_live = {
        "geocoding": "미확인",
        "directions": "미확인",
        "checked_at": "",
        "error": "",
    }


# -----------------------------------------------------------------------------
# CSV
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_places(path: str, modified_time: int) -> pd.DataFrame:
    """
    CSV를 읽고 지역·하위지역 보조 컬럼을 만듭니다.

    CSV에 '지역' 컬럼이 있으면 그 값을 우선 사용하고,
    광주 관련 표기만 앱 명칭인 '전남광주'로 통일합니다.
    """
    del modified_time
    last_error: Exception | None = None

    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            df = pd.read_csv(path, encoding=encoding, dtype=str)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        raise ValueError("CSV를 UTF-8 또는 CP949 형식으로 저장해 주세요.") from last_error

    df.columns = [str(column).replace("\ufeff", "").strip() for column in df.columns]
    df["_csv_line_no"] = range(2, len(df) + 2)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("필수 컬럼이 없습니다: " + ", ".join(missing))

    for column in df.columns:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df = df[(df["이름"] != "") & (df["주소"] != "")].drop_duplicates().copy()

    # 사용자가 관리하는 CSV를 그대로 신뢰하되, 대회 범위 밖 전북 행만 제외합니다.
    excluded = df["주소"].str.contains(
        r"^(?:전라북도|전북\s)",
        regex=True,
        na=False,
    )
    if "지역" in df.columns:
        excluded = excluded | df["지역"].str.contains(
            r"전북|순창",
            regex=True,
            na=False,
        )
    df = df[~excluded].copy()

    derived_region = df["주소"].map(extract_region)
    if "지역" in df.columns:
        original_region = df["지역"].fillna("").astype(str).str.strip()
        df["지역"] = original_region.where(original_region != "", derived_region)
        df.loc[df["지역"].str.contains("광주", na=False), "지역"] = "전남광주"
    else:
        df["지역"] = derived_region

    df["하위지역"] = df.apply(
        lambda row: extract_subregion(row["주소"], row["지역"]),
        axis=1,
    )

    category_order = {"맛집": 0, "관광명소": 1, "문화공간": 2, "카페": 3}
    df["_order"] = df["카테고리"].map(category_order).fillna(99)

    df = (
        df.sort_values(["지역", "_order", "이름"], kind="stable")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    return attach_preview_images(df, IMAGES_DIR)

def get_places() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error("`data/places.csv`를 찾지 못했습니다.")
        st.stop()

    try:
        return load_places(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
    except Exception as exc:
        st.error("CSV를 읽는 중 오류가 발생했습니다.")
        st.exception(exc)
        st.stop()


# -----------------------------------------------------------------------------
# 세션 상태
# -----------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "page": HOME,
        "points": 0,
        "preferences": {},
        "selected_restaurant": "",
        "routes": [],
        "route_index": 0,
        "route_generation": 0,
        "route_variation": 0,
        "saved": [],
        "checkins": [],
        "ai_meta": {},
        "map_meta": {},
        "mobility_open": {},
        "mobility_cache": {},
        "naver_live": {
            "geocoding": "미확인",
            "directions": "미확인",
            "checked_at": "",
            "error": "",
        },
        "qr_last_scan": {},
        "qr_widget_nonce": 0,
        "qr_feedback": None,
        "qr_last_input_sig": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = deepcopy(value)

    if st.session_state.page not in NAV_ITEMS:
        st.session_state.page = HOME

def reset_demo() -> None:
    st.session_state.clear()
    st.rerun()


# -----------------------------------------------------------------------------
# 추천 후보 준비
# -----------------------------------------------------------------------------
def interest_score(item: dict[str, Any], interests: list[str]) -> int:
    wanted = {
        INTEREST_CATEGORY[value]
        for value in interests
        if value in INTEREST_CATEGORY
    }
    category = str(item.get("카테고리") or "")
    text = " ".join(
        [
            category,
            optional_text(item, "태그", "키워드", "설명", "한줄소개"),
        ]
    )

    score = 2 if category in wanted else 0
    for interest in interests:
        keyword = interest.split("·")[0]
        if keyword and keyword in text:
            score += 1
    return score


def companion_score(item: dict[str, Any], companion: str) -> int:
    text = " ".join(
        [
            optional_text(item, "추천동행", "동행", "태그", "키워드", "설명"),
            str(item.get("카테고리") or ""),
        ]
    )
    if not text.strip():
        return 0
    return 2 if companion in text else 0


def prepare_candidate_pool(
    places: pd.DataFrame,
    restaurant_name: str,
    preferences: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    region = preferences["region"]
    restaurant_rows = places[
        (places["지역"] == region)
        & (places["카테고리"] == "맛집")
        & (places["이름"] == restaurant_name)
    ]

    if restaurant_rows.empty:
        return None, [], {"error": "선택한 맛집을 CSV에서 찾지 못했습니다."}

    restaurant = restaurant_rows.iloc[0].to_dict()
    restaurant.update(
        {
            "name": restaurant_name,
            "address": restaurant["주소"],
            "category": "맛집",
            "subregion": restaurant["하위지역"],
        }
    )

    raw_candidates: list[dict[str, Any]] = []
    for item in places[
        (places["지역"] == region) & (places["카테고리"] != "맛집")
    ].to_dict("records"):
        item = dict(item)
        item["interest_score"] = interest_score(
            item,
            list(preferences.get("interests", [])),
        )
        item["companion_score"] = companion_score(item, preferences["companion"])
        raw_candidates.append(item)

    if not raw_candidates:
        return restaurant, [], {"error": "선택 지역에 주변 장소가 없습니다."}

    config = COURSE_OPTIONS[preferences["course_label"]]
    soft_radius = float(config["soft_radius_km"])
    hard_radius = float(config["hard_radius_km"])
    naver_status = get_naver_status()

    if not naver_status["configured"]:
        anchor_area = restaurant["하위지역"]
        for item in raw_candidates:
            item.update(
                {
                    "name": item["이름"],
                    "address": item["주소"],
                    "category": item["카테고리"],
                    "subregion": item["하위지역"],
                    "same_subregion": item["하위지역"] == anchor_area,
                    "distance_km": None,
                }
            )

        raw_candidates.sort(
            key=lambda item: (
                not bool(item.get("same_subregion")),
                -int(item.get("interest_score") or 0),
                -int(item.get("companion_score") or 0),
                str(item.get("이름") or ""),
            )
        )

        return (
            restaurant,
            raw_candidates[:12],
            {
                "configured": False,
                "geocoding_status": "키 없음",
                "geocoded_count": 0,
                "candidate_count": len(raw_candidates),
                "soft_radius_km": soft_radius,
                "hard_radius_km": hard_radius,
                "error": "네이버 지도 키가 없어 행정구역 기준으로 후보를 정렬했습니다.",
            },
        )

    anchor_geo = geocode(restaurant["주소"])
    restaurant.update(anchor_geo)
    restaurant["subregion"] = anchor_geo.get("dongmyun") or restaurant["하위지역"]

    if not anchor_geo.get("ok"):
        for item in raw_candidates:
            item.update(
                {
                    "name": item["이름"],
                    "address": item["주소"],
                    "category": item["카테고리"],
                    "subregion": item["하위지역"],
                    "same_subregion": item["하위지역"] == restaurant["하위지역"],
                    "distance_km": None,
                }
            )

        raw_candidates.sort(
            key=lambda item: (
                not bool(item.get("same_subregion")),
                -int(item.get("interest_score") or 0),
            )
        )

        return (
            restaurant,
            raw_candidates[:12],
            {
                "configured": True,
                "geocoding_status": "실패",
                "geocoded_count": 0,
                "candidate_count": len(raw_candidates),
                "soft_radius_km": soft_radius,
                "hard_radius_km": hard_radius,
                "error": str(anchor_geo.get("error") or "선택한 식당 좌표 변환 실패"),
            },
        )

    enriched = geocode_many(raw_candidates)
    usable: list[dict[str, Any]] = []
    failed_same_area: list[dict[str, Any]] = []
    errors: list[str] = []

    for item in enriched:
        item["name"] = item.get("이름", "")
        item["address"] = item.get("주소", "")
        item["category"] = item.get("카테고리", "장소")
        item["subregion"] = item.get("dongmyun") or item.get("하위지역") or region
        item["same_subregion"] = item["subregion"] == restaurant["subregion"]

        if item.get("ok"):
            distance = straight_distance_km(restaurant, item)
            item["distance_km"] = round(distance, 2) if distance is not None else None
            if distance is not None and distance <= hard_radius:
                usable.append(item)
        else:
            item["distance_km"] = None
            if item["same_subregion"]:
                failed_same_area.append(item)
            errors.append(f"{item.get('이름')}: {item.get('error') or '좌표 변환 실패'}")

    usable.sort(
        key=lambda item: (
            not bool(item.get("same_subregion")),
            float(item.get("distance_km") or 9999) > soft_radius,
            float(item.get("distance_km") or 9999),
            -int(item.get("interest_score") or 0),
            -int(item.get("companion_score") or 0),
        )
    )

    # 좌표 실패 장소는 같은 하위지역인 경우에만 부족분 보완용으로 쓴다.
    pool = usable[:12]
    if len(pool) < 4:
        for item in failed_same_area:
            if item not in pool:
                pool.append(item)
            if len(pool) >= 4:
                break

    geocoded_count = sum(1 for item in enriched if item.get("ok"))
    status = "성공" if geocoded_count == len(enriched) else "일부 성공"

    return (
        restaurant,
        pool,
        {
            "configured": True,
            "geocoding_status": status,
            "geocoded_count": geocoded_count,
            "candidate_count": len(enriched),
            "pool_count": len(pool),
            "soft_radius_km": soft_radius,
            "hard_radius_km": hard_radius,
            "error": " | ".join(errors[:4]),
        },
    )


# -----------------------------------------------------------------------------
# 기본 추천과 AI 추천
# -----------------------------------------------------------------------------
def fallback_title(route_type: str, companion: str) -> str:
    table = {
        "혼자": {
            "가까운 동네": "혼자 천천히 즐기는 동네 한 바퀴",
            "취향 집중": "내 취향대로 고르는 남도 산책",
            "식사 전후": "혼자서도 여유로운 미식 하루",
        },
        "연인": {
            "가까운 동네": "둘이 가볍게 즐기는 동네 데이트",
            "취향 집중": "취향을 나누는 남도 데이트",
            "식사 전후": "산책과 식사를 잇는 둘만의 코스",
        },
        "가족": {
            "가까운 동네": "가족과 편안한 동네 나들이",
            "취향 집중": "온 가족 취향을 담은 남도 여행",
            "식사 전후": "식사와 관광을 잇는 가족 하루",
        },
        "친구": {
            "가까운 동네": "친구와 가볍게 도는 동네 코스",
            "취향 집중": "친구와 취향대로 즐기는 남도 여행",
            "식사 전후": "맛과 이야기가 이어지는 친구 코스",
        },
    }
    return table.get(companion, {}).get(route_type, f"{route_type} 추천 코스")


def fallback_summary(route_type: str, companion: str) -> str:
    return {
        "가까운 동네": "오늘 방문할 식당과 가까운 장소를 우선해 이동 부담을 줄였어요.",
        "취향 집중": f"{companion} 여행과 선택한 관심 분야를 중심으로 골랐어요.",
        "식사 전후": "식사 전 관광과 식사 후 휴식을 균형 있게 연결했어요.",
    }.get(route_type, "선택한 조건에 맞춘 지역 코스예요.")


def balanced_order(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        groups.setdefault(str(item.get("category") or item.get("카테고리") or "장소"), []).append(item)

    for values in groups.values():
        values.sort(key=lambda item: float(item.get("distance_km") or 9999))

    result: list[dict[str, Any]] = []
    while any(groups.values()):
        changed = False
        for category in ("관광명소", "문화공간", "카페", "장소"):
            if groups.get(category):
                result.append(groups[category].pop(0))
                changed = True
        if not changed:
            break
    return result


def fallback_routes(
    candidates: list[dict[str, Any]],
    preferences: dict[str, Any],
) -> list[dict[str, Any]]:
    count = min(
        COURSE_OPTIONS[preferences["course_label"]]["places"],
        len(candidates),
    )

    near = sorted(
        candidates,
        key=lambda item: (
            not bool(item.get("same_subregion")),
            item.get("distance_km") is None,
            float(item.get("distance_km") or 9999),
        ),
    )
    taste = sorted(
        candidates,
        key=lambda item: (
            -int(item.get("interest_score") or 0),
            -int(item.get("companion_score") or 0),
            item.get("distance_km") is None,
            float(item.get("distance_km") or 9999),
        ),
    )
    balanced = balanced_order(near)

    selections = [near[:count], taste[:count], balanced[:count]]

    # 후보가 충분하면 세 코스가 완전히 같은 조합이 되지 않도록
    # 마지막 장소를 다음 후보로 교체한다.
    used_signatures: set[tuple[str, ...]] = set()
    for selection_index, selected in enumerate(selections):
        signature = tuple(str(item.get("name") or item.get("이름") or "") for item in selected)
        if signature in used_signatures and len(candidates) > count and selected:
            replacement_pool = [
                item
                for item in candidates
                if str(item.get("name") or item.get("이름") or "") not in signature
            ]
            if replacement_pool:
                selected = selected[:-1] + [replacement_pool[(selection_index - 1) % len(replacement_pool)]]
                selections[selection_index] = selected
                signature = tuple(str(item.get("name") or item.get("이름") or "") for item in selected)
        used_signatures.add(signature)

    results: list[dict[str, Any]] = []
    for route_type, selected in zip(ROUTE_TYPES, selections):
        results.append(
            {
                "route_type": route_type,
                "title": fallback_title(route_type, preferences["companion"]),
                "summary": fallback_summary(route_type, preferences["companion"]),
                "places": [
                    {
                        **item,
                        "name": item.get("name") or item.get("이름", ""),
                        "category": item.get("category") or item.get("카테고리", "장소"),
                        "reason": fallback_summary(route_type, preferences["companion"]),
                    }
                    for item in selected
                ],
            }
        )
    return results


def point_distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    value = straight_distance_km(first, second)
    return value if value is not None else 9999.0


def greedy_order(
    points: list[dict[str, Any]],
    start: dict[str, Any] | None,
    end: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if len(points) <= 1:
        return points

    if any(point.get("lat") is None or point.get("lng") is None for point in points):
        return points

    remaining = points.copy()
    ordered: list[dict[str, Any]] = []
    current = start if start and start.get("lat") is not None else None

    while remaining:
        if current is None:
            # 종착점(맛집)이 있으면 종착점과 가까운 장소부터 역으로 고른다.
            next_item = min(
                remaining,
                key=lambda item: point_distance(item, end) if end else 0,
            )
        else:
            next_item = min(remaining, key=lambda item: point_distance(current, item))

        ordered.append(next_item)
        remaining.remove(next_item)
        current = next_item

    return ordered


def split_before_count(meal_time: str, count: int) -> int:
    if count <= 1:
        return count
    if meal_time == "점심":
        return 1
    if meal_time == "저녁":
        return max(1, count - 1)
    return max(1, count // 2)


def rough_travel_minutes(first: dict[str, Any], second: dict[str, Any]) -> int:
    distance = straight_distance_km(first, second)
    if distance is None:
        return 10
    return max(5, math.ceil(distance * 1.25 / 35 * 60))


def estimate_route_minutes(origin: dict[str, Any] | None, stops: list[dict[str, Any]]) -> int:
    total = sum(int(stop.get("stay_minutes") or 0) for stop in stops)
    points: list[dict[str, Any]] = []
    if origin:
        points.append(origin)
    points.extend(stops)

    for first, second in zip(points, points[1:]):
        total += rough_travel_minutes(first, second)
    return total


def route_id(route: dict[str, Any]) -> str:
    names = "|".join(stop["name"] for stop in route["stops"])
    raw = f"{route['region']}|{route['route_type']}|{route['title']}|{names}"
    return sha1(raw.encode("utf-8")).hexdigest()[:14]




def max_cafe_count(course_label: str) -> int:
    return 1 if course_label in ("2시간", "3시간") else 2


def candidate_priority(item: dict[str, Any], route_type: str) -> tuple[Any, ...]:
    category = str(item.get("category") or item.get("카테고리") or "장소")
    distance = float(item.get("distance_km") or 9999)
    interest = -int(item.get("interest_score") or 0)
    companion = -int(item.get("companion_score") or 0)
    same_area = not bool(item.get("same_subregion") or item.get("same_area"))
    if route_type == "가까운 동네":
        return (same_area, distance, interest, companion, category)
    if route_type == "취향 집중":
        return (interest, companion, same_area, distance, category)
    category_order = {"관광명소": 0, "문화공간": 1, "카페": 2, "장소": 3}
    return (category_order.get(category, 9), same_area, distance, interest, companion)



def reorder_to_reduce_repetition(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = items.copy()
    result: list[dict[str, Any]] = []
    while remaining:
        prev = str(result[-1].get("category") or result[-1].get("카테고리") or "") if result else ""
        pick_index = 0
        for index, item in enumerate(remaining):
            category = str(item.get("category") or item.get("카테고리") or "")
            if category != prev:
                pick_index = index
                break
        result.append(remaining.pop(pick_index))
    return result



def normalize_selected_places(
    selected: list[dict[str, Any]],
    candidate_pool: list[dict[str, Any]],
    preferences: dict[str, Any],
    route_type: str,
) -> list[dict[str, Any]]:
    if not selected:
        return []

    target_count = len(selected)
    limit = max_cafe_count(preferences["course_label"])
    used_names = {str(item.get("name") or item.get("이름") or "").strip() for item in selected}

    def category_of(item: dict[str, Any]) -> str:
        return str(item.get("category") or item.get("카테고리") or "장소")

    non_cafe_candidates = [
        item for item in sorted(candidate_pool, key=lambda row: candidate_priority(row, route_type))
        if category_of(item) != "카페" and str(item.get("name") or item.get("이름") or "").strip() not in used_names
    ]

    cafes = [item for item in selected if category_of(item) == "카페"]
    if len(cafes) > limit:
        kept = []
        cafe_kept = 0
        for item in selected:
            if category_of(item) == "카페":
                if cafe_kept < limit:
                    kept.append(item)
                    cafe_kept += 1
                elif non_cafe_candidates:
                    replacement = non_cafe_candidates.pop(0)
                    used_names.add(str(replacement.get("name") or replacement.get("이름") or "").strip())
                    kept.append(replacement)
                else:
                    kept.append(item)
            else:
                kept.append(item)
        selected = kept

    scenic_count = sum(1 for item in selected if category_of(item) in {"관광명소", "문화공간"})
    if scenic_count == 0:
        replacements = [
            item for item in sorted(candidate_pool, key=lambda row: candidate_priority(row, route_type))
            if category_of(item) in {"관광명소", "문화공간"}
            and str(item.get("name") or item.get("이름") or "").strip() not in {str(row.get("name") or row.get("이름") or "").strip() for row in selected}
        ]
        if replacements:
            replace_index = next((i for i, item in enumerate(selected) if category_of(item) == "카페"), len(selected)-1)
            selected[replace_index] = replacements[0]

    selected = sorted(selected, key=lambda item: candidate_priority(item, route_type))
    selected = reorder_to_reduce_repetition(selected)

    # 점심 코스의 첫 장소는 가능하면 카페가 아닌 장소로 둡니다.
    if preferences.get("meal_time") == "점심" and selected:
        first_cat = category_of(selected[0])
        if first_cat == "카페":
            non_cafe_index = next(
                (i for i, item in enumerate(selected) if category_of(item) != "카페"),
                None,
            )
            if non_cafe_index not in (None, 0):
                selected[0], selected[non_cafe_index] = selected[non_cafe_index], selected[0]

    # 카페 과다 보정 과정에서 candidate_pool의 원본 행이 들어오면
    # AI가 만든 reason 필드가 없을 수 있으므로 화면용 필드를 항상 완성합니다.
    normalized: list[dict[str, Any]] = []
    for item in selected[:target_count]:
        name = str(item.get("name") or item.get("이름") or "").strip()
        if not name:
            continue

        category = str(item.get("category") or item.get("카테고리") or "장소").strip()
        normalized.append(
            {
                **item,
                "name": name,
                "address": str(item.get("address") or item.get("주소") or "").strip(),
                "category": category,
                "reason": str(
                    item.get("reason")
                    or fallback_summary(route_type, preferences["companion"])
                ).strip(),
            }
        )

    return normalized

def assemble_routes(
    raw_routes: list[dict[str, Any]],
    restaurant: dict[str, Any],
    candidate_pool: list[dict[str, Any]],
    preferences: dict[str, Any],
    origin: dict[str, Any] | None,
    source: str,
) -> list[dict[str, Any]]:
    config = COURSE_OPTIONS[preferences["course_label"]]
    routes: list[dict[str, Any]] = []

    for index, raw in enumerate(raw_routes[:3]):
        route_type = str(raw.get("route_type") or ROUTE_TYPES[min(index, 2)])
        selected: list[dict[str, Any]] = []
        used: set[str] = set()

        for item in raw.get("places", []):
            name = str(item.get("name") or item.get("이름") or "").strip()
            if not name or name in used:
                continue
            used.add(name)
            selected.append(
                {
                    **item,
                    "name": name,
                    "address": str(item.get("address") or item.get("주소") or ""),
                    "category": str(item.get("category") or item.get("카테고리") or "장소"),
                    "reason": str(item.get("reason") or fallback_summary(route_type, preferences["companion"])),
                }
            )

        if not selected:
            continue

        selected = normalize_selected_places(selected, candidate_pool, preferences, route_type)
        if not selected:
            continue

        before_count = split_before_count(preferences["meal_time"], len(selected))
        before = selected[:before_count]
        after = selected[before_count:]
        before = greedy_order(before, origin, restaurant)
        after = greedy_order(after, restaurant, None)

        stops: list[dict[str, Any]] = []
        for item in before:
            stops.append(
                {
                    **item,
                    "phase": "식사 전",
                    "stay_minutes": parse_minutes(
                        optional_text(item, "체류시간", "예상체류시간", "소요시간"),
                        config["stay"],
                    ),
                }
            )

        stops.append(
            {
                **restaurant,
                "name": restaurant["name"],
                "address": restaurant["주소"],
                "category": "맛집",
                "phase": "식사 장소",
                "reason": "오늘 방문할 식당을 일정의 기준점으로 연결했어요.",
                "stay_minutes": parse_minutes(
                    optional_text(restaurant, "체류시간", "예상체류시간"),
                    config["meal_stay"],
                ),
            }
        )

        for item in after:
            stops.append(
                {
                    **item,
                    "phase": "식사 후",
                    "stay_minutes": parse_minutes(
                        optional_text(item, "체류시간", "예상체류시간", "소요시간"),
                        config["stay"],
                    ),
                }
            )

        route = {
            "route_type": route_type,
            "title": str(raw.get("title") or fallback_title(route_type, preferences["companion"])),
            "summary": str(raw.get("summary") or fallback_summary(route_type, preferences["companion"])),
            "region": preferences["region"],
            "restaurant": restaurant["name"],
            "companion": preferences["companion"],
            "meal_time": preferences["meal_time"],
            "course_label": preferences["course_label"],
            "course_minutes": preferences["course_minutes"],
            "start_time": preferences["start_time"],
            "interests": list(preferences.get("interests", [])),
            "origin": origin if origin and origin.get("ok") else None,
            "stops": stops,
            "source": source,
        }
        route["estimated_minutes"] = estimate_route_minutes(route["origin"], stops)
        route["id"] = route_id(route)
        routes.append(route)

    return routes


def create_recommendations(
    places: pd.DataFrame,
    restaurant_name: str,
    preferences: dict[str, Any],
    variation: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    restaurant, candidates, map_meta = prepare_candidate_pool(
        places,
        restaurant_name,
        preferences,
    )

    if restaurant is None or not candidates:
        return [], {"source": "", "error": "추천 가능한 주변 장소가 없습니다."}, map_meta

    origin: dict[str, Any] | None = None
    origin_address = str(preferences.get("origin_address") or "").strip()
    if origin_address:
        origin_geo = geocode(origin_address)
        origin = {
            **origin_geo,
            "name": "출발지",
            "address": origin_address,
            "category": "출발지",
            "stay_minutes": 0,
        }
        if not origin_geo.get("ok"):
            map_meta["origin_error"] = origin_geo.get("error") or "출발지 좌표 변환 실패"

    target_count = min(
        COURSE_OPTIONS[preferences["course_label"]]["places"],
        len(candidates),
    )

    result = generate_gemini_routes(
        region=preferences["region"],
        restaurant=restaurant_name,
        course_label=preferences["course_label"],
        course_minutes=preferences["course_minutes"],
        companion=preferences["companion"],
        meal_time=preferences["meal_time"],
        interests=preferences.get("interests", []),
        candidates=candidates,
        place_count=target_count,
        variation=variation,
    )

    if result["success"]:
        routes = assemble_routes(
            result["routes"],
            restaurant,
            candidates,
            preferences,
            origin,
            "gemini",
        )
        if routes:
            return (
                routes,
                {
                    "source": "gemini",
                    "model": result.get("model", ""),
                    "latency_seconds": result.get("latency_seconds", 0.0),
                    "repair_count": result.get("repair_count", 0),
                    "attempts": result.get("attempts", []),
                    "error": "",
                    "timed_out": False,
                },
                map_meta,
            )

    fallback = fallback_routes(candidates, preferences)
    routes = assemble_routes(
        fallback,
        restaurant,
        candidates,
        preferences,
        origin,
        "python",
    )
    return (
        routes,
        {
            "source": "python",
            "model": result.get("model", ""),
            "latency_seconds": result.get("latency_seconds", 0.0),
            "repair_count": 0,
            "attempts": result.get("attempts", []),
            "error": result.get("error", "Gemini 추천을 적용하지 못했습니다."),
            "timed_out": bool(result.get("timed_out")),
        },
        map_meta,
    )


def active_route() -> dict[str, Any] | None:
    if not st.session_state.routes:
        return None
    index = max(0, min(st.session_state.route_index, len(st.session_state.routes) - 1))
    return st.session_state.routes[index]


def save_active_route() -> None:
    route = active_route()
    if not route:
        return

    saved_ids = {item["id"] for item in st.session_state.saved}
    if route["id"] in saved_ids:
        st.toast("이미 저장한 코스예요.", icon="ℹ️")
        return

    saved = deepcopy(route)
    saved["saved_at"] = datetime.now(KST).strftime("%m.%d %H:%M")
    st.session_state.saved.insert(0, saved)
    st.toast("MY 일정에 저장했어요.", icon="✅")


# -----------------------------------------------------------------------------
# 이동정보
# -----------------------------------------------------------------------------
def ensure_coords(point: dict[str, Any]) -> dict[str, Any]:
    try:
        float(point["lat"])
        float(point["lng"])
        if point.get("ok", True):
            return dict(point)
    except (KeyError, TypeError, ValueError):
        pass

    address = str(point.get("address") or point.get("주소") or "").strip()
    return {**point, **geocode(address)}


def mobility_key(route_id_value: str, leg_index: int, start: dict[str, Any], goal: dict[str, Any]) -> str:
    raw = "|".join(
        (
            route_id_value,
            str(leg_index),
            str(start.get("name") or start.get("이름") or ""),
            str(goal.get("name") or goal.get("이름") or ""),
        )
    )
    return sha1(raw.encode("utf-8")).hexdigest()[:16]


def toggle_mobility(key: str) -> None:
    """이동시간 카드의 열림 상태만 바꾸고 코스 페이지를 유지합니다."""
    st.session_state.page = COURSE
    current = bool(st.session_state.mobility_open.get(key))
    st.session_state.mobility_open[key] = not current

def calculate_mobility(start: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
    checked_at = datetime.now(KST)
    start_geo = ensure_coords(start)
    goal_geo = ensure_coords(goal)

    if not start_geo.get("ok") or not goal_geo.get("ok"):
        error = "출발지 또는 도착지의 좌표를 찾지 못했습니다."
        st.session_state.naver_live = {
            "geocoding": "실패",
            "directions": "미확인",
            "checked_at": checked_at.strftime("%H:%M"),
            "error": error,
        }
        return {
            "start": start_geo,
            "goal": goal_geo,
            "driving": {"ok": False, "error": error},
            "walking": {"ok": False, "error": error},
            "checked_at": checked_at,
        }

    driving = get_driving_route(start_geo, goal_geo)
    walking = estimate_walk(start_geo, goal_geo)
    st.session_state.naver_live = {
        "geocoding": "성공",
        "directions": "성공" if driving.get("ok") else "실패",
        "checked_at": checked_at.strftime("%H:%M"),
        "error": "" if driving.get("ok") else str(driving.get("error") or "이동정보 조회 실패"),
    }

    return {
        "start": start_geo,
        "goal": goal_geo,
        "driving": driving,
        "walking": walking,
        "checked_at": checked_at,
    }


def render_mobility_result(result: dict[str, Any]) -> None:
    """자동차·택시·도보 정보를 잘리지 않는 반응형 카드로 표시합니다."""
    driving = result["driving"]
    walking = result["walking"]
    checked_at: datetime = result["checked_at"]

    start_name = str(result["start"].get("name") or result["start"].get("이름") or "출발지")
    goal_name = str(result["goal"].get("name") or result["goal"].get("이름") or "도착지")

    if driving.get("ok"):
        duration = int(driving["duration_min"])
        distance = float(driving["distance_km"])
        arrival = checked_at + timedelta(minutes=duration)
        taxi_fare = int(driving.get("taxi_fare") or 0)
        taxi_value = f"약 {taxi_fare:,}원" if taxi_fare else "요금 미제공"

        driving_html = f"""
        <div class="mobility-source">Ⓝ 네이버 지도에서 가져온 정보입니다.</div>
        <div class="mobility-grid">
            <div class="mobility-mode car">
                <div class="mobility-label">🚗 자동차</div>
                <div class="mobility-value">{duration}분</div>
                <div class="mobility-detail">{distance:.1f}km</div>
                <div class="mobility-arrival">지금 출발 시 {arrival.strftime('%H:%M')} 도착 예상</div>
            </div>
            <div class="mobility-mode taxi">
                <div class="mobility-label">🚕 택시</div>
                <div class="mobility-value taxi-value">{h(taxi_value)}</div>
                <div class="mobility-detail">차량 기준 약 {duration}분</div>
                <div class="mobility-arrival">예상 요금은 교통 상황에 따라 달라질 수 있어요.</div>
            </div>
        </div>
        """
    else:
        driving_html = f"""
        <div class="mobility-error">
            네이버 지도에서 자동차 이동정보를 불러오지 못했어요.<br>
            <span>{h(driving.get('error') or '오류 정보 없음')}</span>
        </div>
        """

    if walking.get("ok"):
        walk_minutes = int(walking["duration_min"])
        walk_distance = float(walking["distance_km"])
        walk_arrival = checked_at + timedelta(minutes=walk_minutes)
        walking_html = f"""
        <div class="mobility-mode walk">
            <div class="mobility-label">🚶 도보 추정</div>
            <div class="mobility-value">{h(format_minutes(walk_minutes))}</div>
            <div class="mobility-detail">약 {walk_distance:.1f}km</div>
            <div class="mobility-arrival">지금 출발 시 {walk_arrival.strftime('%H:%M')} 도착 예상</div>
        </div>
        """
    else:
        walking_html = """
        <div class="mobility-mode walk">
            <div class="mobility-label">🚶 도보 추정</div>
            <div class="mobility-value small">계산 불가</div>
        </div>
        """

    ui(
        f"""
        <div class="mobility-card">
            <div class="mobility-route">{h(start_name)} → {h(goal_name)}</div>
            {driving_html}
            {walking_html}
            <div class="mobility-note">도보 시간은 좌표 간 거리를 보정한 추정값으로 실제 보행 경로와 다를 수 있어요.</div>
        </div>
        """
    )

def render_stop_actions(
    route: dict[str, Any],
    leg_index: int,
    start: dict[str, Any] | None,
    goal: dict[str, Any],
) -> None:
    """장소 정보와 해당 구간 이동시간을 두 개의 나란한 버튼으로 제공합니다."""
    search_url = build_naver_search_url(goal["name"], goal["address"])
    left, right = st.columns(2, gap="small")

    with left:
        st.link_button("Ⓝ 장소·예약 보기", search_url, width="stretch")

    if start is None:
        with right:
            st.button(
                "코스 시작 장소",
                key=f"first_stop_{route['id']}_{leg_index}",
                disabled=True,
                width="stretch",
            )
        return

    key = mobility_key(route["id"], leg_index, start, goal)
    is_open = bool(st.session_state.mobility_open.get(key))

    with right:
        st.button(
            "이동시간 닫기" if is_open else "🕒 이동시간 보기",
            key=f"toggle_mobility_{key}",
            width="stretch",
            on_click=toggle_mobility,
            args=(key,),
        )

    if not st.session_state.mobility_open.get(key):
        return

    if key not in st.session_state.mobility_cache:
        with st.spinner("네이버 지도에서 이동정보를 불러오고 있어요..."):
            st.session_state.mobility_cache[key] = calculate_mobility(start, goal)

    render_mobility_result(st.session_state.mobility_cache[key])

# -----------------------------------------------------------------------------
# 카메라
# -----------------------------------------------------------------------------
def qr_scan_event_id(payload: str, scanned_at: datetime, place_name: str) -> str:
    raw = f"{payload}|{scanned_at.isoformat()}|{place_name}"
    return sha1(raw.encode("utf-8")).hexdigest()[:16]



def attempt_qr_checkin(stop: dict[str, Any], image_bytes: bytes) -> dict[str, Any]:
    """업로드 또는 카메라 이미지의 시연용 QR을 확인하고 포인트를 적립합니다."""
    decoded = decode_qr_image(image_bytes)
    if not decoded.get("ok"):
        st.session_state.qr_widget_nonce += 1
        return {
            "kind": "error",
            "message": str(decoded.get("error") or "QR 코드를 읽지 못했습니다."),
        }

    payload = str(decoded.get("payload") or "").strip()
    qr_info = get_demo_qr_info(payload)
    if qr_info is None:
        st.session_state.qr_widget_nonce += 1
        return {
            "kind": "error",
            "message": "운남화이팅 시연용 QR이 아닙니다. 제공된 시연용 QR 이미지를 사용해 주세요.",
        }

    now = datetime.now(KST)
    last_scanned_at = st.session_state.qr_last_scan.get(payload)
    remaining = cooldown_remaining(last_scanned_at, now=now)
    if remaining > 0:
        st.session_state.qr_widget_nonce += 1
        return {
            "kind": "cooldown",
            "message": (
                "한 번 찍은 QR은 쿨타임이 끝나야 다시 찍을 수 있습니다. "
                f"{remaining}초 후 다시 시도해 주세요."
            ),
            "remaining": remaining,
            "qr_code": qr_info["code"],
        }

    points = int(qr_info.get("points") or 100)
    st.session_state.qr_last_scan[payload] = now.isoformat()
    st.session_state.checkins.insert(
        0,
        {
            "id": qr_scan_event_id(payload, now, stop["name"]),
            "qr_code": qr_info["code"],
            "qr_label": qr_info["label"],
            "name": stop["name"],
            "category": stop["category"],
            "checked_at": now.strftime("%m.%d %H:%M:%S"),
            "points": points,
            "image_hash": sha256(image_bytes).hexdigest()[:12],
        },
    )
    st.session_state.points += points
    st.session_state.qr_widget_nonce += 1

    return {
        "kind": "success",
        "message": f"{qr_info['label']} 확인 완료! {points}P를 적립했어요.",
        "points": points,
        "qr_code": qr_info["code"],
    }


def active_qr_cooldowns() -> list[dict[str, Any]]:
    """현재 세션에서 아직 끝나지 않은 QR 쿨타임 목록을 반환합니다."""
    now = datetime.now(KST)
    active: list[dict[str, Any]] = []

    for payload, scanned_at in st.session_state.qr_last_scan.items():
        remaining = cooldown_remaining(scanned_at, now=now)
        if remaining <= 0:
            continue
        info = DEMO_QR_CODES.get(payload, {})
        active.append(
            {
                "code": info.get("code", "QR"),
                "label": info.get("label", "시연용 QR"),
                "remaining": remaining,
            }
        )

    return sorted(active, key=lambda item: int(item["remaining"]))


# -----------------------------------------------------------------------------
# CSS와 공통 UI
# -----------------------------------------------------------------------------
def inject_css() -> None:
    ui(
        """
        <style>
        :root {
            --g900: #063f38;
            --g800: #075b50;
            --g700: #08796a;
            --g500: #19a98f;
            --g100: #e8f6f2;
            --ink: #17231f;
            --muted: #68766f;
            --line: #dce8e4;
            --paper: #fbfdfc;
            --orange: #ff6d61;
        }

        html, body, [class*="css"] {
            font-family: Pretendard, -apple-system, BlinkMacSystemFont,
                "Segoe UI", "Noto Sans KR", sans-serif;
        }

        html, body, [data-testid="stAppViewContainer"] {
            overflow-x: hidden !important;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 8% 2%, rgba(25,169,143,.16), transparent 28%),
                radial-gradient(circle at 95% 18%, rgba(255,109,97,.08), transparent 24%),
                #e9efec;
        }

        [data-testid="stHeader"], [data-testid="stToolbar"],
        [data-testid="stDecoration"], #MainMenu, footer {
            display: none !important;
        }

        [data-testid="stMainBlockContainer"], .block-container {
            width: 100% !important;
            max-width: 470px !important;
            min-height: 100vh;
            padding: 1rem 1rem calc(10rem + env(safe-area-inset-bottom)) !important;
            background: var(--paper);
            box-shadow: 0 0 50px rgba(25,55,47,.13);
        }

        .app-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: .7rem;
            margin-bottom: 1rem;
        }

        .brand { display: flex; align-items: center; gap: .65rem; min-width: 0; }
        .logo {
            width: 2.7rem; height: 2.7rem; display: grid; place-items: center;
            border-radius: 1rem; color: white; font-weight: 950;
            background: linear-gradient(145deg, var(--g900), var(--g500));
            box-shadow: 0 10px 22px rgba(8,121,106,.24);
        }
        .brand strong { display: block; color: var(--ink); font-size: 1rem; }
        .brand span { display: block; margin-top: .15rem; color: var(--muted); font-size: .65rem; }
        .point-pill {
            flex: none; padding: .52rem .7rem; border: 1px solid var(--line);
            border-radius: 999px; background: white; color: var(--g800);
            font-size: .76rem; font-weight: 900;
        }

        .hero {
            position: relative; overflow: hidden; padding: 1.45rem 1.3rem;
            border-radius: 1.65rem; color: white;
            background: linear-gradient(145deg, var(--g900), var(--g700) 58%, #1ca88f);
            box-shadow: 0 20px 38px rgba(7,91,80,.23);
        }
        .hero::after {
            content: ""; position: absolute; width: 10rem; height: 10rem;
            right: -4rem; top: -4rem; border-radius: 50%;
            background: rgba(255,255,255,.12);
        }
        .hero > * { position: relative; z-index: 1; }
        .hero small {
            display: inline-flex; padding: .32rem .58rem; border-radius: 999px;
            background: rgba(255,255,255,.15); font-size: .65rem; font-weight: 850;
        }
        .hero h1 {
            margin: .75rem 0 .4rem; color: white; font-size: 1.65rem;
            line-height: 1.18; letter-spacing: -.04em;
        }
        .hero p { margin: 0; max-width: 20rem; color: rgba(255,255,255,.84); font-size: .8rem; line-height: 1.5; }

        .stats {
            display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .5rem; margin: .8rem 0 1.1rem;
        }
        .stat {
            min-width: 0; padding: .72rem .3rem; border: 1px solid var(--line);
            border-radius: 1rem; background: white; text-align: center;
        }
        .stat strong { display: block; color: var(--ink); font-size: 1rem; }
        .stat span { color: var(--muted); font-size: .62rem; }

        .section-head {
            display: flex; align-items: flex-end; justify-content: space-between;
            gap: .5rem; margin: 1.2rem 0 .65rem;
        }
        .section-head small { color: var(--g700); font-size: .61rem; font-weight: 950; letter-spacing: .08em; }
        .section-head strong { display: block; margin-top: .12rem; color: var(--ink); font-size: 1.1rem; }
        .section-head span { color: var(--muted); font-size: .63rem; text-align: right; }

        .status-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: .55rem; margin: .75rem 0; }
        .status-card { padding: .75rem; border: 1px solid var(--line); border-radius: 1rem; background: white; }
        .status-card.ok { border-color: #bfe2d8; background: #f3fbf8; }
        .status-card.warn { border-color: #efd7b5; background: #fff9f0; }
        .status-card b { display: block; color: var(--ink); font-size: .72rem; }
        .status-card span { display: block; margin-top: .15rem; color: var(--muted); font-size: .61rem; }

        .st-key-planner {
            padding: 1rem !important; border: 1px solid var(--line) !important;
            border-radius: 1.3rem !important; background: white !important;
            box-shadow: 0 12px 28px rgba(31,62,54,.07) !important;
        }
        .st-key-planner [data-testid="stVerticalBlock"] { gap: .72rem !important; }

        div[data-baseweb="select"] > div, [data-testid="stTextInput"] input {
            min-height: 3rem; border-radius: .9rem !important;
            border-color: #d9e6e1 !important; background: #f7faf9 !important;
        }
        [data-testid="stWidgetLabel"] p { color: #40504a; font-size: .73rem; font-weight: 850; }
        .stButton > button, .stLinkButton > a {
            width: 100%; min-height: 3rem; border-radius: .92rem !important;
            font-weight: 850 !important;
        }
        .stButton > button[kind="primary"] {
            border: 0; color: white; background: linear-gradient(135deg, var(--g900), #0e927a);
            box-shadow: 0 10px 20px rgba(8,120,106,.18);
        }

        .proof {
            display: flex; gap: .7rem; align-items: center; padding: .8rem .85rem;
            margin: .7rem 0; border: 1px solid #cfe9e1; border-radius: 1rem;
            background: #f3fbf8;
        }
        .proof.warn { border-color: #efd7b5; background: #fff9f0; }
        .proof-icon {
            width: 2.2rem; height: 2.2rem; display: grid; place-items: center;
            flex: none; border-radius: .75rem; background: var(--g700); color: white;
        }
        .proof.warn .proof-icon { background: #d88931; }
        .proof b { display: block; color: var(--ink); font-size: .75rem; }
        .proof span { display: block; margin-top: .12rem; color: var(--muted); font-size: .61rem; }

        .route-hero {
            padding: 1.05rem; margin: .75rem 0 1rem; border-radius: 1.3rem;
            color: white; background: linear-gradient(140deg, #183a32, var(--g700));
            box-shadow: 0 14px 28px rgba(20,64,54,.18);
        }
        .route-badge {
            display: inline-flex; padding: .25rem .5rem; border-radius: 999px;
            background: rgba(255,255,255,.14); font-size: .6rem; font-weight: 900;
        }
        .route-hero h2 { margin: .48rem 0 .25rem; color: white; font-size: 1.15rem; }
        .route-hero p { margin: 0; color: rgba(255,255,255,.82); font-size: .72rem; line-height: 1.45; }
        .route-meta { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: .4rem; margin-top: .75rem; }
        .route-meta div { min-width: 0; padding: .45rem .15rem; border-radius: .7rem; background: rgba(255,255,255,.1); text-align: center; }
        .route-meta b { display: block; color: white; font-size: .7rem; white-space: nowrap; }
        .route-meta span { color: rgba(255,255,255,.66); font-size: .54rem; }

        .simple-card, .stop-card, .saved-card, .reward-card, .empty-card {
            box-sizing: border-box; width: 100%; max-width: 100%;
            padding: .9rem; margin-bottom: .7rem; border: 1px solid var(--line);
            border-radius: 1.08rem; background: white;
            box-shadow: 0 8px 20px rgba(36,65,57,.05);
        }
        .stop-layout { display: grid; grid-template-columns: minmax(0, 1fr) 96px; gap: .8rem; align-items: start; }
        .stop-copy-wrap { min-width: 0; }
        .stop-thumb { width: 96px; height: 88px; border-radius: .95rem; overflow: hidden; background: #eef4f1; border: 1px solid #dfe8e4; flex: none; }
        .stop-thumb-image { width: 100%; height: 100%; object-fit: cover; display: block; }
        .stop-thumb-fallback { width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: .22rem; color: #54706a; background: linear-gradient(145deg,#f3f7f6,#edf4f2); }
        .stop-thumb-fallback span { font-size: 1.25rem; }
        .stop-thumb-fallback small { font-size: .55rem; font-weight: 900; }
        .stop-top { display: flex; align-items: center; justify-content: space-between; gap: .45rem; }
        .phase {
            display: inline-flex; padding: .22rem .45rem; border-radius: 999px;
            background: var(--g100); color: var(--g800); font-size: .57rem; font-weight: 950;
        }
        .phase.food { background: #fff0ec; color: #c24f36; }
        .stop-time { color: var(--muted); font-size: .59rem; text-align: right; }
        .stop-card h3 { margin: .43rem 0 .22rem; color: var(--ink); font-size: .92rem; overflow-wrap: anywhere; }
        .stop-card p { margin: 0; color: #51615b; font-size: .68rem; line-height: 1.5; overflow-wrap: anywhere; }
        .address { margin-top: .4rem; color: #7a8782; font-size: .6rem; overflow-wrap: anywhere; }
        .kicker { color: var(--g700); font-size: .6rem; font-weight: 950; }
        .card-title { margin-top: .25rem; color: var(--ink); font-size: .9rem; font-weight: 950; overflow-wrap: anywhere; }
        .card-copy { margin-top: .3rem; color: var(--muted); font-size: .65rem; line-height: 1.45; overflow-wrap: anywhere; }

        .reward-card { border-color: #f0dca8; background: linear-gradient(145deg,#fff9eb,#fff); }
        .reward-card small { color: #a86c00; font-size: .58rem; font-weight: 950; letter-spacing: .08em; }
        .reward-card strong { display: block; margin: .28rem 0 .18rem; color: var(--ink); font-size: .84rem; }
        .reward-card span { color: var(--muted); font-size: .65rem; }
        .empty-card { text-align: center; padding: 1.35rem; }
        .empty-card .emoji { font-size: 1.8rem; }
        .empty-card b { display: block; margin: .35rem 0 .18rem; color: var(--ink); }
        .empty-card span { color: var(--muted); font-size: .72rem; }
        .saved-head { display: flex; justify-content: space-between; gap: .5rem; }
        .saved-time { color: var(--muted); font-size: .58rem; white-space: nowrap; }
        .done { display: inline-flex; margin-top: .45rem; padding: .25rem .48rem; border-radius: 999px; background: var(--g100); color: var(--g800); font-size: .59rem; font-weight: 900; }

        /* 카메라 위젯은 높이·overflow를 강제로 제한하지 않는다. */
        [data-testid="stCameraInput"] { border-radius: 1rem; }

        .safe-space { height: 7.5rem; }
        .st-key-bottom_nav_shell {
            position: fixed !important; left: 50% !important;
            bottom: max(.5rem, env(safe-area-inset-bottom)) !important;
            transform: translateX(-50%) !important; z-index: 99999 !important;
            width: min(438px, calc(100vw - 1.1rem)) !important;
            padding: .4rem !important; border: 1px solid rgba(204,220,214,.96) !important;
            border-radius: 1.18rem !important; background: rgba(255,255,255,.96) !important;
            box-shadow: 0 15px 36px rgba(29,57,49,.2) !important;
            backdrop-filter: blur(16px);
        }
        .st-key-bottom_nav_shell [data-testid="stVerticalBlock"] { gap: 0 !important; }
        .st-key-bottom_nav_shell [data-testid="column"] { padding: 0 .1rem !important; }
        .st-key-bottom_nav_shell .stButton > button {
            min-height: 3rem !important; padding: .35rem .1rem !important;
            font-size: .64rem !important; border-radius: .8rem !important;
        }

        @media (max-width: 640px) {
            [data-testid="stMainBlockContainer"], .block-container {
                max-width: none !important;
                padding: .8rem .82rem calc(9.5rem + env(safe-area-inset-bottom)) !important;
                box-shadow: none;
            }
            .hero h1 { font-size: 1.5rem; }
            .st-key-planner { padding: .88rem !important; }
            .route-meta b { font-size: .66rem; }
            .stop-layout { grid-template-columns: minmax(0, 1fr) 84px; gap: .7rem; }
            .stop-thumb { width: 84px; height: 78px; }
        }


        .restaurant-anchor {
            padding: .9rem;
            border: 1px solid #cfe4dc;
            border-radius: 1rem;
            background: linear-gradient(145deg, #f2faf7, #fff);
        }
        .restaurant-anchor .anchor-label {
            color: var(--g700); font-size: .6rem; font-weight: 950;
        }
        .restaurant-anchor .anchor-name {
            margin-top: .28rem; color: var(--ink); font-size: .9rem;
            font-weight: 950; overflow-wrap: anywhere;
        }
        .restaurant-anchor .anchor-address {
            margin-top: .28rem; color: var(--muted); font-size: .62rem;
            line-height: 1.45; overflow-wrap: anywhere;
        }
        .restaurant-helper {
            margin: -.15rem 0 .45rem; color: var(--muted); font-size: .62rem;
        }

        .mobility-card {
            box-sizing: border-box; width: 100%; max-width: 100%;
            padding: .9rem; margin: .2rem 0 .8rem;
            border: 1px solid #d7e3ef; border-radius: 1.05rem;
            background: linear-gradient(145deg, #f7fbff, #fff);
            box-shadow: 0 8px 20px rgba(36,65,57,.045);
            overflow: hidden;
        }
        .mobility-route {
            color: var(--ink); font-size: .72rem; font-weight: 950;
            overflow-wrap: anywhere;
        }
        .mobility-source {
            margin-top: .4rem; color: #6d7b86; font-size: .6rem;
        }
        .mobility-grid {
            display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: .62rem; margin-top: .7rem;
        }
        .mobility-mode {
            box-sizing: border-box; min-width: 0; padding: .72rem;
            border: 1px solid #dfe8f1; border-radius: .9rem; background: #fff;
            overflow: hidden;
        }
        .mobility-mode.car { border-color: #c7ddf7; background: #f2f7ff; }
        .mobility-mode.taxi { border-color: #efddc7; background: #fff9f1; }
        .mobility-mode.walk {
            margin-top: .62rem; border-color: #cfe7dc; background: #f3faf7;
        }
        .mobility-label { color: var(--ink); font-size: .65rem; font-weight: 900; }
        .mobility-value {
            margin-top: .34rem; color: #182a35;
            font-size: clamp(1.35rem, 5.8vw, 1.85rem); line-height: 1.08;
            font-weight: 500; letter-spacing: -.035em;
            white-space: normal; overflow-wrap: anywhere;
        }
        .mobility-value.taxi-value { font-size: clamp(1.15rem, 4.8vw, 1.6rem); }
        .mobility-value.small { font-size: 1.15rem; }
        .mobility-detail {
            margin-top: .4rem; color: var(--g700); font-size: .61rem;
            font-weight: 850; overflow-wrap: anywhere;
        }
        .mobility-arrival {
            margin-top: .18rem; color: var(--muted); font-size: .57rem;
            line-height: 1.4; overflow-wrap: anywhere;
        }
        .mobility-note {
            margin-top: .6rem; color: var(--muted); font-size: .57rem;
            line-height: 1.45;
        }
        .mobility-error {
            margin-top: .65rem; padding: .7rem; border-radius: .8rem;
            background: #fff6ef; color: #9f5138; font-size: .64rem;
            line-height: 1.45;
        }
        .mobility-error span { color: #84675e; font-size: .57rem; }

        .qr-rule {
            padding: .8rem .85rem; margin-bottom: .75rem;
            border: 1px solid #f0d9b0; border-radius: 1rem;
            background: #fff9ef; color: #74531f; font-size: .66rem;
            line-height: 1.5;
        }
        .qr-rule strong { display: block; color: #8f5f0d; font-size: .72rem; }
        .qr-cooldown-list {
            padding: .72rem .82rem; margin-bottom: .7rem;
            border: 1px solid #dbe7e3; border-radius: .95rem; background: #f7faf9;
        }
        .qr-cooldown-item {
            display: flex; justify-content: space-between; gap: .5rem;
            padding: .22rem 0; color: var(--muted); font-size: .62rem;
        }
        .qr-cooldown-item b { color: var(--ink); }
        .checkin-history-row {
            display: flex; justify-content: space-between; gap: .7rem;
            padding: .65rem 0; border-bottom: 1px solid #edf2f0;
        }
        .checkin-history-row:last-child { border-bottom: 0; }
        .checkin-history-row .history-main { min-width: 0; }
        .checkin-history-row .history-title {
            color: var(--ink); font-size: .69rem; font-weight: 850;
            overflow-wrap: anywhere;
        }
        .checkin-history-row .history-meta {
            margin-top: .15rem; color: var(--muted); font-size: .58rem;
        }
        .checkin-history-row .history-point {
            flex: none; color: var(--g700); font-size: .68rem; font-weight: 950;
        }

        @media (max-width: 365px) {
            .mobility-grid { grid-template-columns: 1fr; }
            .mobility-value, .mobility-value.taxi-value { font-size: 1.4rem; }
            .stop-layout { grid-template-columns: 1fr; }
            .stop-thumb { width: 100%; height: 144px; }
        }
        </style>
        """
    )


def render_header() -> None:
    ui(
        f"""
        <div class="app-header">
            <div class="brand">
                <div class="logo">운</div>
                <div>
                    <strong>{h(APP_NAME)}</strong>
                    <span>{h(APP_SUBTITLE)}</span>
                </div>
            </div>
            <div class="point-pill">🪙 {st.session_state.points:,}P</div>
        </div>
        """
    )


def render_section(title: str, eyebrow: str = "", caption: str = "") -> None:
    ui(
        f"""
        <div class="section-head">
            <div>
                <small>{h(eyebrow)}</small>
                <strong>{h(title)}</strong>
            </div>
            <span>{h(caption)}</span>
        </div>
        """
    )


def render_empty(icon: str, title: str, copy: str) -> None:
    ui(
        f"""
        <div class="empty-card">
            <div class="emoji">{icon}</div>
            <b>{h(title)}</b>
            <span>{h(copy)}</span>
        </div>
        """
    )


def render_bottom_nav() -> None:
    with st.container(key="bottom_nav_shell"):
        columns = st.columns(4)
        for column, page in zip(columns, NAV_ITEMS):
            with column:
                st.button(
                    page,
                    key=f"nav_{page}",
                    type="primary" if st.session_state.page == page else "secondary",
                    width="stretch",
                    on_click=set_page,
                    args=(page,),
                )


# -----------------------------------------------------------------------------
# 홈
# -----------------------------------------------------------------------------
def page_home(places: pd.DataFrame) -> None:
    ui(
        """
        <div class="hero">
            <small>전남광주 AI 로컬 미식 코스</small>
            <h1>오늘 갈 식당을 고르면<br>주변 여행 동선이 완성돼요.</h1>
            <p>동행과 취향, 가능한 시간을 반영해 관광지·문화공간·카페를 자연스럽게 연결합니다.</p>
        </div>
        """
    )

    ui(
        f"""
        <div class="stats">
            <div class="stat"><strong>{len(places)}</strong><span>등록 장소</span></div>
            <div class="stat"><strong>{places['지역'].nunique()}</strong><span>서비스 지역</span></div>
            <div class="stat"><strong>{len(st.session_state.checkins)}</strong><span>QR 체크인</span></div>
        </div>
        """
    )

    render_section("나만의 미식 코스 만들기", "PLAN", "지역·동행·시간")

    regions = sorted(
        places.loc[places["지역"] != "", "지역"].dropna().unique().tolist(),
        key=region_sort_key,
    )
    if not regions:
        st.warning("CSV에서 지역을 찾지 못했습니다.")
        return

    if st.session_state.get("home_region") not in regions:
        st.session_state.home_region = regions[0]

    with st.container(key="planner"):
        region = st.selectbox("여행 지역", regions, key="home_region")

        restaurant_rows = places[
            (places["지역"] == region) & (places["카테고리"] == "맛집")
        ].copy()
        restaurant_names = restaurant_rows["이름"].tolist()
        has_restaurant = bool(restaurant_names)
        restaurant_options = restaurant_names if restaurant_names else ["등록된 식당 없음"]

        if st.session_state.get("home_restaurant") not in restaurant_options:
            st.session_state.home_restaurant = restaurant_options[0]

        subregion_by_name = {
            str(row["이름"]): str(row.get("하위지역") or "")
            for _, row in restaurant_rows.iterrows()
        }

        restaurant = st.selectbox(
            "오늘 방문할 식당",
            restaurant_options,
            key="home_restaurant",
            disabled=not has_restaurant,
            format_func=lambda name: (
                f"{name} · {subregion_by_name.get(name)}"
                if subregion_by_name.get(name)
                else name
            ),
        )

        ui(
            '<div class="restaurant-helper">선택한 식당을 기준으로 가까운 장소와 식사 전후 동선을 구성해요.</div>'
        )

        if has_restaurant:
            selected_row = restaurant_rows[restaurant_rows["이름"] == restaurant].iloc[0]
            restaurant_url = build_naver_search_url(restaurant, str(selected_row["주소"]))
            ui(
                f"""
                <div class="restaurant-anchor">
                    <div class="anchor-label">오늘의 식사 장소</div>
                    <div class="anchor-name">🍽️ {h(restaurant)}</div>
                    <div class="anchor-address">📍 {h(short_address(str(selected_row['주소'])))}</div>
                </div>
                """
            )
            st.link_button("Ⓝ 네이버에서 식당 정보·예약 보기", restaurant_url, width="stretch")
        else:
            st.info("이 지역의 맛집 행을 CSV에 추가하면 바로 선택할 수 있어요.")

        companion = st.segmented_control(
            "동행 유형",
            ("혼자", "연인", "가족", "친구"),
            default="혼자",
            key="home_companion",
            width="stretch",
        ) or "혼자"

        meal_time = st.segmented_control(
            "식사 시간",
            ("점심", "저녁", "상관없음"),
            default="점심",
            key="home_meal_time",
            width="stretch",
        ) or "점심"

        course_label = st.select_slider(
            "코스 여유시간",
            options=tuple(COURSE_OPTIONS),
            value="3시간",
            key="home_course_label",
        )

        start_time_value = st.time_input(
            "코스 시작 시간",
            value=time(10, 0),
            step=900,
            key="home_start_time",
        )

        origin_address = st.text_input(
            "출발지 주소 (선택)",
            placeholder="비워두면 첫 추천 장소부터 시작해요",
            key="home_origin",
        )

        interests = st.multiselect(
            "관심 분야",
            tuple(INTEREST_CATEGORY),
            default=("자연·산책", "카페·디저트"),
            key="home_interests",
        )

        clicked = st.button(
            "✨ AI 미식 코스 만들기",
            type="primary",
            width="stretch",
            disabled=not has_restaurant,
            key="create_course",
        )

    if clicked:
        config = COURSE_OPTIONS[course_label]
        preferences = {
            "region": region,
            "companion": companion,
            "meal_time": meal_time,
            "course_label": course_label,
            "course_minutes": config["minutes"],
            "start_time": start_time_value.strftime("%H:%M"),
            "origin_address": origin_address.strip(),
            "interests": list(interests),
        }

        with st.spinner("AI가 취향과 이동 범위를 함께 살펴보고 있어요..."):
            routes, ai_meta, map_meta = create_recommendations(
                places,
                restaurant,
                preferences,
                variation=0,
            )

        if not routes:
            st.warning("이 지역에서 코스를 만들 수 있는 주변 장소가 부족합니다.")
            return

        st.session_state.preferences = preferences
        st.session_state.selected_restaurant = restaurant
        st.session_state.routes = routes
        st.session_state.route_index = 0
        st.session_state.route_variation = 0
        st.session_state.route_generation += 1
        st.session_state.ai_meta = ai_meta
        st.session_state.map_meta = map_meta
        clear_route_runtime()
        st.session_state.naver_live["geocoding"] = map_meta.get("geocoding_status", "미확인")
        set_page(COURSE)
        st.rerun()

    ui(
        """
        <div class="reward-card">
            <small>LOCAL BENEFIT</small>
            <strong>코스 저장 → QR 체크인 → 포인트 적립</strong>
            <span>식당 방문을 주변 관광지와 지역 상권 경험으로 이어가요.</span>
        </div>
        """
    )

# -----------------------------------------------------------------------------
# 코스
# -----------------------------------------------------------------------------
def render_recommendation_status() -> None:
    meta = st.session_state.ai_meta
    if meta.get("source") == "gemini":
        latency = float(meta.get("latency_seconds") or 0.0)
        ui(
            f"""
            <div class="proof">
                <div class="proof-icon">✦</div>
                <div>
                    <b>AI 맞춤 코스가 완성됐어요</b>
                    <span>동행과 관심사를 반영했어요{' · ' + format(latency, '.1f') + '초' if latency else ''}</span>
                </div>
            </div>
            """
        )
    else:
        ui(
            """
            <div class="proof warn">
                <div class="proof-icon">!</div>
                <div>
                    <b>기본 추천 코스를 보여드리고 있어요</b>
                    <span>아래의 한 개 재시도 버튼으로 AI 추천을 다시 요청할 수 있어요.</span>
                </div>
            </div>
            """
        )


def render_route_card(route: dict[str, Any]) -> None:
    ui(
        f"""
        <div class="route-hero">
            <div class="route-badge">{h(route['route_type'])} · {h(route['region'])}</div>
            <h2>{h(route['title'])}</h2>
            <p>{h(route['summary'])}</p>
            <div class="route-meta">
                <div><b>{len(route['stops'])}곳</b><span>전체 방문지</span></div>
                <div><b>약 {h(format_minutes(route['estimated_minutes']))}</b><span>예상 소요</span></div>
                <div><b>{h(route['start_time'])}</b><span>코스 시작</span></div>
            </div>
        </div>
        """
    )


def render_origin(origin: dict[str, Any], start_time_text: str) -> None:
    ui(
        f"""
        <div class="simple-card">
            <div class="kicker">코스 출발 · {h(start_time_text)}</div>
            <div class="card-title">📍 입력한 출발지</div>
            <div class="card-copy">{h(short_address(origin.get('address', '')))}</div>
        </div>
        """
    )



def render_stop(stop: dict[str, Any]) -> None:
    """장소 카드에 필요한 값이 일부 없더라도 화면이 중단되지 않게 표시합니다."""
    category = str(stop.get("category") or stop.get("카테고리") or "장소").strip()
    name = str(stop.get("name") or stop.get("이름") or "이름 없는 장소").strip()
    address = str(stop.get("address") or stop.get("주소") or "").strip()
    phase = str(stop.get("phase") or "추천 장소").strip()
    stay_minutes = int(stop.get("stay_minutes") or 30)
    reason = str(
        stop.get("reason")
        or "선택한 조건과 동선을 고려해 추천한 장소예요."
    ).strip()

    css_class = "food" if category == "맛집" else ""
    image_src = preview_image_data_uri(str(stop.get("_preview_path") or ""))
    if image_src:
        preview_html = (
            f'<img class="stop-thumb-image" src="{image_src}" '
            f'alt="{h(name)} 미리보기">'
        )
    else:
        preview_html = (
            '<div class="stop-thumb-fallback">'
            f'<span>{CATEGORY_ICON.get(category, "📍")}</span>'
            f'<small>{h(category)}</small>'
            '</div>'
        )

    ui(
        f"""
        <div class="stop-card">
            <div class="stop-layout">
                <div class="stop-copy-wrap">
                    <div class="stop-top">
                        <span class="phase {css_class}">{h(phase)} · {h(category)}</span>
                        <span class="stop-time">권장 {stay_minutes}분</span>
                    </div>
                    <h3>{CATEGORY_ICON.get(category, '📍')} {h(name)}</h3>
                    <p>{h(reason)}</p>
                    <div class="address">📍 {h(short_address(address))}</div>
                </div>
                <div class="stop-thumb">{preview_html}</div>
            </div>
        </div>
        """
    )


def regenerate_course(places: pd.DataFrame) -> None:
    st.session_state.route_variation += 1
    with st.spinner("새로운 AI 코스를 만들고 있어요..."):
        routes, ai_meta, map_meta = create_recommendations(
            places,
            st.session_state.selected_restaurant,
            st.session_state.preferences,
            st.session_state.route_variation,
        )

    if routes:
        st.session_state.routes = routes
        st.session_state.route_index = 0
        st.session_state.route_generation += 1
        st.session_state.ai_meta = ai_meta
        st.session_state.map_meta = map_meta
        clear_route_runtime()
        st.session_state.naver_live["geocoding"] = map_meta.get("geocoding_status", "미확인")
        st.rerun()

    st.error("새로운 코스를 만들지 못했어요. 현재 코스는 그대로 유지됩니다.")


def page_course(places: pd.DataFrame) -> None:
    if not st.session_state.routes:
        render_empty("🧭", "아직 만든 코스가 없어요", "홈에서 지역과 오늘 방문할 식당을 선택해 주세요.")
        st.button(
            "홈에서 코스 만들기",
            type="primary",
            width="stretch",
            on_click=set_page,
            args=(HOME,),
        )
        return

    render_recommendation_status()
    render_section("추천 코스", "AI CURATION", "의도가 다른 세 가지")

    route_types = tuple(route["route_type"] for route in st.session_state.routes)
    selected = st.segmented_control(
        "코스 유형",
        route_types,
        default=route_types[min(st.session_state.route_index, len(route_types) - 1)],
        key=f"route_type_{st.session_state.route_generation}",
        label_visibility="collapsed",
        width="stretch",
    ) or route_types[0]
    st.session_state.route_index = route_types.index(selected)

    route = active_route()
    if route is None:
        return

    render_route_card(route)

    previous: dict[str, Any] | None = route.get("origin")
    if previous:
        render_origin(previous, route["start_time"])

    for index, stop in enumerate(route["stops"]):
        render_stop(stop)
        render_stop_actions(route, index, previous, stop)
        previous = stop

    ui(
        """
        <div class="reward-card">
            <small>LOCAL CHECK-IN</small>
            <strong>추천 장소에서 QR 체크인하면 +100P</strong>
            <span>관광 동선을 지역 상권 이용과 연결하는 시연용 리워드예요.</span>
        </div>
        """
    )

    saved = route["id"] in {item["id"] for item in st.session_state.saved}
    left, right = st.columns(2)
    with left:
        if st.button(
            "✓ 저장됨" if saved else "＋ MY에 담기",
            disabled=saved,
            width="stretch",
            key=f"save_route_{route['id']}",
        ):
            save_active_route()
            st.rerun()
    with right:
        st.button(
            "📷 체크인",
            type="primary",
            width="stretch",
            key=f"checkin_route_{route['id']}",
            on_click=set_page,
            args=(CHECKIN,),
        )

    retry_label = (
        "🔄 AI 추천 다시 시도"
        if st.session_state.ai_meta.get("source") == "python"
        else "🔄 다른 AI 코스 만들기"
    )
    if st.button(retry_label, width="stretch", key="regenerate_ai_course"):
        regenerate_course(places)

    if st.session_state.ai_meta.get("source") == "python":
        with st.expander("AI 연결 상태 확인"):
            st.code(st.session_state.ai_meta.get("error", "오류 정보 없음"), language=None)
            attempts = st.session_state.ai_meta.get("attempts", [])
            if attempts:
                st.caption("자동 재시도 기록")
                st.json(attempts)


# -----------------------------------------------------------------------------
# 체크인
# -----------------------------------------------------------------------------

def page_checkin() -> None:
    route = active_route() or (st.session_state.saved[0] if st.session_state.saved else None)
    if route is None:
        render_empty("📷", "체크인할 코스가 없어요", "먼저 홈에서 코스를 만들어 주세요.")
        return

    render_section("QR 포인트 체크인", "CHECK-IN", "+100P · QR별 1분 쿨타임")

    names = [stop["name"] for stop in route["stops"]]
    if st.session_state.get("checkin_place") not in names:
        st.session_state.checkin_place = names[0]

    selected_name = st.selectbox("체크인할 장소", names, key="checkin_place")
    stop = next(item for item in route["stops"] if item["name"] == selected_name)

    ui(
        f"""
        <div class="simple-card">
            <div class="kicker">{CATEGORY_ICON.get(stop['category'], '📍')} {h(stop['category'])}</div>
            <div class="card-title">{h(stop['name'])}</div>
            <div class="card-copy">📍 {h(short_address(stop['address']))}</div>
        </div>
        """
    )

    feedback = st.session_state.get("qr_feedback")
    if isinstance(feedback, dict):
        if feedback.get("kind") == "success":
            st.success(str(feedback.get("message") or "체크인에 성공했어요."))
        elif feedback.get("kind") == "cooldown":
            st.warning(str(feedback.get("message") or "QR 쿨타임이 남아 있어요."))
        else:
            st.error(str(feedback.get("message") or "QR을 확인하지 못했어요."))

    ui(
        f"""
        <div class="qr-rule">
            <strong>QR 체크인 이용 안내</strong>
            같은 QR은 체크인 후 {QR_COOLDOWN_SECONDS}초 동안 다시 사용할 수 없습니다.
            다른 시연용 QR은 바로 사용할 수 있어요.
        </div>
        """
    )

    cooldowns = active_qr_cooldowns()
    if cooldowns:
        rows = "".join(
            f'<div class="qr-cooldown-item"><b>{h(item["code"])}</b><span>{int(item["remaining"])}초 남음</span></div>'
            for item in cooldowns
        )
        ui(f'<div class="qr-cooldown-list">{rows}</div>')

    method = st.segmented_control(
        "QR 등록 방법",
        ("QR 이미지 업로드", "카메라 촬영"),
        default="QR 이미지 업로드",
        key="qr_input_method",
        width="stretch",
    ) or "QR 이미지 업로드"

    nonce = int(st.session_state.qr_widget_nonce)
    image_bytes: bytes | None = None

    if method == "QR 이미지 업로드":
        uploaded = st.file_uploader(
            "시연용 QR 이미지를 선택해 주세요",
            type=("png", "jpg", "jpeg"),
            key=f"qr_upload_{nonce}",
        )
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            current_sig = sha1(image_bytes).hexdigest()
            if st.session_state.qr_last_input_sig != current_sig:
                st.session_state.qr_feedback = None
                st.session_state.qr_last_input_sig = current_sig
            st.image(image_bytes, caption="선택한 QR 이미지", width="stretch")
    else:
        camera_file = st.camera_input(
            "QR이 화면 가운데 크게 보이도록 촬영해 주세요",
            key=f"qr_camera_{nonce}",
            width="stretch",
        )
        if camera_file is not None:
            image_bytes = camera_file.getvalue()
            current_sig = sha1(image_bytes).hexdigest()
            if st.session_state.qr_last_input_sig != current_sig:
                st.session_state.qr_feedback = None
                st.session_state.qr_last_input_sig = current_sig

    if image_bytes is not None:
        if st.button(
            "QR 체크인 확인",
            type="primary",
            width="stretch",
            key=f"confirm_qr_{nonce}_{sha1(selected_name.encode('utf-8')).hexdigest()[:8]}",
        ):
            st.session_state.qr_feedback = attempt_qr_checkin(stop, image_bytes)
            st.session_state.qr_last_input_sig = ""
            st.rerun()

    completed = len(st.session_state.checkins)
    ui(
        f"""
        <div class="simple-card">
            <div class="kicker">포인트 체크인 기록</div>
            <div class="card-title">{completed}회 체크인 · {st.session_state.points:,}P</div>
            <div class="card-copy">시연용 QR 5개를 번갈아 사용하거나, 같은 QR은 1분 뒤 다시 사용할 수 있어요.</div>
        </div>
        """
    )

# -----------------------------------------------------------------------------
# MY
# -----------------------------------------------------------------------------
def open_saved_route(route: dict[str, Any]) -> None:
    st.session_state.routes = [deepcopy(route)]
    st.session_state.route_index = 0
    st.session_state.route_generation += 1
    st.session_state.selected_restaurant = route.get("restaurant", "")
    st.session_state.preferences = {
        "region": route.get("region", ""),
        "companion": route.get("companion", "혼자"),
        "meal_time": route.get("meal_time", "상관없음"),
        "course_label": route.get("course_label", "3시간"),
        "course_minutes": int(route.get("course_minutes", 180)),
        "start_time": route.get("start_time", "10:00"),
        "origin_address": str((route.get("origin") or {}).get("address") or ""),
        "interests": list(route.get("interests", [])),
    }
    st.session_state.ai_meta = {"source": route.get("source", "python")}
    st.session_state.map_meta = {}
    clear_route_runtime()
    set_page(COURSE)


def delete_saved_route(route_id_value: str) -> None:
    st.session_state.saved = [
        item for item in st.session_state.saved if item["id"] != route_id_value
    ]


def page_my(places: pd.DataFrame) -> None:
    render_section("나의 여행 기록", "MY TRIP", "저장 일정과 포인트")

    ui(
        f"""
        <div class="stats">
            <div class="stat"><strong>{st.session_state.points:,}P</strong><span>적립 포인트</span></div>
            <div class="stat"><strong>{len(st.session_state.saved)}</strong><span>저장 코스</span></div>
            <div class="stat"><strong>{len(st.session_state.checkins)}</strong><span>체크인</span></div>
        </div>
        """
    )

    render_section("저장한 코스", "SAVED", f"{len(st.session_state.saved)}개")
    if not st.session_state.saved:
        render_empty("🗂️", "저장한 코스가 없어요", "추천 코스에서 MY에 담기를 눌러보세요.")
    else:
        for index, route in enumerate(list(st.session_state.saved)):
            ui(
                f"""
                <div class="saved-card">
                    <div class="saved-head">
                        <div>
                            <div class="kicker">{h(route['route_type'])} · {h(route['region'])}</div>
                            <div class="card-title">{h(route['title'])}</div>
                        </div>
                        <div class="saved-time">{h(route.get('saved_at'))}</div>
                    </div>
                    <div class="card-copy">{h(route['restaurant'])} 식사 장소 · {len(route['stops'])}개 장소</div>
                </div>
                """
            )
            left, right = st.columns((2, 1))
            with left:
                st.button(
                    "코스 열기",
                    width="stretch",
                    key=f"open_saved_{route['id']}_{index}",
                    on_click=open_saved_route,
                    args=(route,),
                )
            with right:
                st.button(
                    "삭제",
                    width="stretch",
                    key=f"delete_saved_{route['id']}_{index}",
                    on_click=delete_saved_route,
                    args=(route["id"],),
                )

    render_section("체크인 기록", "HISTORY", f"{len(st.session_state.checkins)}건")
    if not st.session_state.checkins:
        render_empty("📍", "체크인 기록이 없어요", "추천 장소에서 QR 체크인을 해보세요.")
    else:
        history = "".join(
            (
                '<div class="checkin-history-row">'
                '<div class="history-main">'
                f'<div class="history-title">{CATEGORY_ICON.get(item["category"], "📍")} {h(item["name"])}</div>'
                f'<div class="history-meta">{h(item.get("qr_code", "QR"))} · {h(item["checked_at"])}</div>'
                '</div>'
                f'<div class="history-point">+{item["points"]}P</div>'
                '</div>'
            )
            for item in st.session_state.checkins
        )
        ui(f'<div class="simple-card">{history}</div>')

    render_section("연결 상태", "APP STATUS", "시연 준비 확인")
    gemini = get_gemini_status()
    naver = get_naver_status()
    live = st.session_state.naver_live

    with st.expander("AI·지도·데이터 상태"):
        st.write("**Gemini 키:** " + ("설정됨" if gemini["configured"] else "키 없음"))
        st.write(f"**Gemini 모델:** `{gemini['model']}`")
        st.write("**네이버 지도 키:** " + ("설정됨" if naver["configured"] else "키 없음"))
        st.write(f"**Geocoding 최근 상태:** `{live.get('geocoding', '미확인')}`")
        st.write(f"**자동차 경로 최근 상태:** `{live.get('directions', '미확인')}`")
        if live.get("checked_at"):
            st.write(f"**최근 지도 조회:** `{live['checked_at']}`")
        st.write(f"**CSV 장소:** `{len(places)}개`")
        st.write(
            "**지역:** "
            + ", ".join(sorted(places["지역"].unique(), key=region_sort_key))
        )

        if st.session_state.ai_meta.get("source"):
            source_text = "Gemini API" if st.session_state.ai_meta["source"] == "gemini" else "Python 기본 추천"
            st.write(f"**최근 추천 방식:** `{source_text}`")

        if st.session_state.ai_meta.get("error"):
            st.caption("최근 AI 오류")
            st.code(st.session_state.ai_meta["error"], language=None)

        if st.session_state.map_meta.get("error"):
            st.caption("최근 지도 일부 오류")
            st.code(st.session_state.map_meta["error"], language=None)

        if live.get("error"):
            st.caption("최근 자동차 경로 오류")
            st.code(live["error"], language=None)

        if st.button("지도·CSV 캐시 새로고침", width="stretch", key="clear_app_cache"):
            clear_map_cache()
            st.cache_data.clear()
            clear_route_runtime()
            st.rerun()

    if st.button("시연 기록 전체 초기화", width="stretch", key="reset_demo"):
        reset_demo()


# -----------------------------------------------------------------------------
# 실행
# -----------------------------------------------------------------------------
def main() -> None:
    inject_css()
    init_state()
    places = get_places()
    render_header()

    if st.session_state.page == HOME:
        page_home(places)
    elif st.session_state.page == COURSE:
        page_course(places)
    elif st.session_state.page == CHECKIN:
        page_checkin()
    else:
        page_my(places)

    ui('<div class="safe-space"></div>')
    render_bottom_nav()


if __name__ == "__main__":
    main()
