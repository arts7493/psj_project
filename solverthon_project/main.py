from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from hashlib import sha1, sha256
from html import escape
from pathlib import Path
import re
from textwrap import dedent
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st

from ai_client import (
    generate_gemini_routes,
    get_gemini_status,
)


# -----------------------------------------------------------------------------
# 앱 기본 설정
# -----------------------------------------------------------------------------
APP_NAME = "WAITGO"
APP_SUBTITLE = "광주·전남 대기여행"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "places.csv"

REQUIRED_COLUMNS = (
    "카테고리",
    "이름",
    "주소",
)

HOME = "🏠 홈"
COURSE = "🧭 코스"
CHECKIN = "📷 체크인"
MY = "👤 MY"

NAV_ITEMS = (
    HOME,
    COURSE,
    CHECKIN,
    MY,
)

NAV_KEY = "app_nav"

CATEGORY_ICON = {
    "맛집": "🍽️",
    "관광명소": "🌿",
    "문화공간": "🎨",
    "카페": "☕",
}

CATEGORY_CLASS = {
    "맛집": "food",
    "관광명소": "nature",
    "문화공간": "culture",
    "카페": "cafe",
}

DEFAULT_DURATION = {
    "맛집": 60,
    "관광명소": 30,
    "문화공간": 35,
    "카페": 30,
}

INTEREST_CATEGORY = {
    "자연·산책": "관광명소",
    "문화·전시": "문화공간",
    "카페·디저트": "카페",
    "로컬 감성": "관광명소",
}

FALLBACK_ROUTES = (
    (
        "가볍게 쉬어가는 코스",
        "여유형",
        (
            "대기 전후로 부담 없이 "
            "머물 수 있는 장소를 담았어요."
        ),
    ),
    (
        "취향을 따라가는 코스",
        "취향형",
        (
            "선택한 관심사를 중심으로 "
            "지역의 매력을 골랐어요."
        ),
    ),
    (
        "식사 뒤까지 이어지는 코스",
        "균형형",
        (
            "대기시간부터 식사 후 일정까지 "
            "자연스럽게 이어져요."
        ),
    ),
)

st.set_page_config(
    page_title=(
        f"{APP_NAME} · "
        f"{APP_SUBTITLE}"
    ),
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------------
# 공통 도우미
# -----------------------------------------------------------------------------
def ui(markup: str) -> None:
    """
    순수 HTML/CSS는 st.html로 렌더링한다.

    구버전 Streamlit에서는
    들여쓰기를 제거한 st.markdown으로 대체한다.
    """
    body = dedent(markup).strip()

    if hasattr(st, "html"):
        st.html(body)

    else:
        st.markdown(
            body,
            unsafe_allow_html=True,
        )


def h(value: Any) -> str:
    """
    동적 텍스트를 HTML에서 안전하게 사용한다.
    """
    return escape(
        str(value or "").strip()
    )


def short_address(
    address: str,
) -> str:
    """
    카드에서는 주소를 조금 짧게 표시한다.
    """
    return (
        str(address or "")
        .replace(
            "전라남도",
            "전남",
        )
        .replace(
            "광주광역시",
            "광주",
        )
        .strip()
    )


def optional_text(
    row: dict[str, Any] | pd.Series,
    *columns: str,
) -> str:
    """
    CSV에 선택 컬럼을 추가했을 때
    자동으로 활용한다.
    """
    for column in columns:
        value = row.get(
            column,
            "",
        )

        if (
            value is not None
            and str(value).strip()
        ):
            return str(value).strip()

    return ""


def parse_minutes(
    value: Any,
    default: int,
) -> int:
    """
    '30분', '약 40분', '35' 등을
    분 단위 숫자로 바꾼다.
    """
    match = re.search(
        r"\d+",
        str(value or ""),
    )

    if not match:
        return default

    return max(
        10,
        min(
            180,
            int(match.group()),
        ),
    )


def format_minutes(
    value: int,
) -> str:
    """
    분을 1시간 30분 형태로 표시한다.
    """
    hours, minutes = divmod(
        max(0, int(value)),
        60,
    )

    if hours and minutes:
        return (
            f"{hours}시간 "
            f"{minutes}분"
        )

    if hours:
        return f"{hours}시간"

    return f"{minutes}분"


def stable_number(
    text: str,
    modulo: int,
) -> int:
    """
    같은 문자열에서 항상 같은
    시연용 숫자를 만든다.
    """
    digest = sha256(
        text.encode("utf-8")
    ).hexdigest()

    return (
        int(
            digest[:14],
            16,
        )
        % modulo
    )


def extract_region(
    address: str,
) -> str:
    """
    주소에서 광주 구 또는
    전남 시·군을 추출한다.
    """
    address = re.sub(
        r"\s+",
        " ",
        str(address).strip(),
    )

    tokens = address.split()

    if address.startswith(
        (
            "광주광역시",
            "전남광주",
            "광주 ",
        )
    ):
        district = next(
            (
                token
                for token in tokens
                if token.endswith(
                    (
                        "구",
                        "군",
                    )
                )
            ),
            "",
        )

        if district:
            return f"광주 {district}"

        return "광주"

    if address.startswith(
        (
            "전라남도",
            "전남 ",
        )
    ):
        return next(
            (
                token
                for token in tokens[1:]
                if token.endswith(
                    (
                        "시",
                        "군",
                    )
                )
            ),
            "전남",
        )

    return "지역 확인 필요"


def default_reason(
    category: str,
    companion: str,
) -> str:
    """
    Gemini 실패 시 사용할 기본 추천 이유.
    """
    prefix = {
        "혼자": "혼자서도 편안하게",
        "연인": "함께 분위기를 즐기며",
        "가족": "가족과 여유롭게",
        "친구": "친구와 가볍게",
    }.get(
        companion,
        "여유롭게",
    )

    ending = {
        "관광명소": (
            "둘러보기 좋은 지역 명소예요."
        ),
        "문화공간": (
            "새로운 이야기를 만나기 좋은 공간이에요."
        ),
        "카페": (
            "잠시 쉬어가기 좋은 카페예요."
        ),
    }.get(
        category,
        "방문하기 좋은 장소예요.",
    )

    return f"{prefix} {ending}"


def go(page: str) -> None:
    """
    하단 메뉴를 변경하고 즉시 다시 실행한다.
    """
    st.session_state[NAV_KEY] = page
    st.rerun()


# -----------------------------------------------------------------------------
# CSV 로드
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_places(
    path: str,
    modified_time: int,
) -> pd.DataFrame:
    """
    data/places.csv를 읽는다.

    modified_time을 이용해
    CSV 수정 후 캐시가 자동 갱신되게 한다.
    """
    del modified_time

    last_error: Exception | None = None

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "cp949",
    ):
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                dtype=str,
            )

            break

        except UnicodeDecodeError as exc:
            last_error = exc

    else:
        raise ValueError(
            "CSV를 UTF-8 또는 "
            "CP949 형식으로 저장해 주세요."
        ) from last_error

    df.columns = [
        str(column)
        .replace(
            "\ufeff",
            "",
        )
        .strip()
        for column in df.columns
    ]

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "필수 컬럼이 없습니다: "
            + ", ".join(missing)
        )

    for column in df.columns:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    df = (
        df[
            (df["이름"] != "")
            & (df["주소"] != "")
        ]
        .drop_duplicates()
        .copy()
    )

    derived_region = (
        df["주소"]
        .map(extract_region)
    )

    if "지역" in df.columns:
        df["지역"] = (
            df["지역"]
            .where(
                df["지역"] != "",
                derived_region,
            )
        )

    else:
        df["지역"] = derived_region

    category_order = {
        "맛집": 0,
        "관광명소": 1,
        "문화공간": 2,
        "카페": 3,
    }

    df["_order"] = (
        df["카테고리"]
        .map(category_order)
        .fillna(99)
    )

    return (
        df.sort_values(
            [
                "지역",
                "_order",
                "이름",
            ],
            kind="stable",
        )
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def get_places() -> pd.DataFrame:
    """
    CSV 오류를 Streamlit 화면에 표시한다.
    """
    if not DATA_PATH.exists():
        st.error(
            "`data/places.csv`를 "
            "찾지 못했습니다."
        )

        st.code(
            str(DATA_PATH),
            language=None,
        )

        st.stop()

    try:
        return load_places(
            str(DATA_PATH),
            DATA_PATH.stat().st_mtime_ns,
        )

    except Exception as exc:
        st.error(
            "CSV를 읽는 중 "
            "오류가 발생했습니다."
        )

        st.exception(exc)
        st.stop()


# -----------------------------------------------------------------------------
# 세션 상태
# -----------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        NAV_KEY: HOME,
        "points": 0,
        "preferences": {},
        "selected_restaurant": "",
        "routes": [],
        "route_index": 0,
        "route_generation": 0,
        "route_variation": 0,
        "queue": None,
        "saved": [],
        "checkins": [],
        "ai_meta": {
            "source": "",
            "model": "",
            "latency_seconds": 0.0,
            "interaction_id": "",
            "repair_count": 0,
            "warnings": [],
            "error": "",
        },
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = deepcopy(
                value
            )

    if (
        st.session_state[NAV_KEY]
        not in NAV_ITEMS
    ):
        st.session_state[NAV_KEY] = HOME


def reset_demo() -> None:
    """
    CSV는 유지하고 현재 시연 기록만 초기화한다.
    """
    st.session_state.clear()
    st.rerun()


# -----------------------------------------------------------------------------
# 추천 코스 생성
# -----------------------------------------------------------------------------
def candidate_rows(
    places: pd.DataFrame,
    region: str,
) -> list[dict[str, Any]]:
    """
    선택 지역의 맛집 외 장소를 가져온다.
    """
    return (
        places[
            (places["지역"] == region)
            & (
                places["카테고리"]
                != "맛집"
            )
        ]
        .to_dict("records")
    )


def restaurant_row(
    places: pd.DataFrame,
    region: str,
    restaurant: str,
) -> dict[str, Any] | None:
    """
    선택 맛집의 CSV 행을 가져온다.
    """
    rows = places[
        (places["지역"] == region)
        & (
            places["카테고리"]
            == "맛집"
        )
        & (
            places["이름"]
            == restaurant
        )
    ]

    if rows.empty:
        return None

    return rows.iloc[0].to_dict()


def requested_place_count(
    wait: int,
    candidate_count: int,
) -> int:
    """
    코스에 넣을 지역 장소 수.
    """
    if wait <= 45:
        count = 2

    elif wait <= 75:
        count = 3

    else:
        count = 4

    return max(
        1,
        min(
            count,
            candidate_count,
        ),
    )


def fallback_raw_routes(
    candidates: list[dict[str, Any]],
    preferences: dict[str, Any],
    variation: int,
) -> list[dict[str, Any]]:
    """
    Gemini 호출이 실패한 경우
    사용할 Python 기본 추천.
    """
    target_count = (
        requested_place_count(
            int(
                preferences["wait"]
            ),
            len(candidates),
        )
    )

    preferred_categories = [
        INTEREST_CATEGORY[item]
        for item in preferences.get(
            "interests",
            [],
        )
        if item in INTEREST_CATEGORY
    ]

    results: list[dict[str, Any]] = []

    for (
        route_index,
        route_info,
    ) in enumerate(
        FALLBACK_ROUTES
    ):
        (
            title,
            badge,
            summary,
        ) = route_info

        def sort_key(
            item: dict[str, Any],
        ) -> tuple[int, int]:
            category = str(
                item.get(
                    "카테고리",
                    "",
                )
            )

            if (
                route_index == 1
                and category
                in preferred_categories
            ):
                category_rank = (
                    preferred_categories
                    .index(category)
                )

            else:
                category_rank = 0

            seed = (
                f"{preferences['region']}|"
                f"{variation}|"
                f"{route_index}|"
                f"{item.get('이름', '')}|"
                f"{item.get('주소', '')}"
            )

            return (
                category_rank,
                stable_number(
                    seed,
                    10**9,
                ),
            )

        selected = sorted(
            candidates,
            key=sort_key,
        )[:target_count]

        results.append(
            {
                "title": title,
                "badge": badge,
                "summary": summary,
                "places": [
                    {
                        "name": str(
                            item.get(
                                "이름",
                                "",
                            )
                        ),
                        "reason": (
                            default_reason(
                                str(
                                    item.get(
                                        "카테고리",
                                        "장소",
                                    )
                                ),
                                preferences[
                                    "companion"
                                ],
                            )
                        ),
                    }
                    for item in selected
                ],
            }
        )

    return results


def route_id(
    route: dict[str, Any],
) -> str:
    """
    저장 코스 중복 검사용 ID.
    """
    names = "|".join(
        stop["name"]
        for stop in route["stops"]
    )

    source = (
        f"{route['region']}|"
        f"{route['title']}|"
        f"{route['restaurant']}|"
        f"{names}"
    )

    return sha1(
        source.encode("utf-8")
    ).hexdigest()[:14]


def assemble_routes(
    raw_routes: list[dict[str, Any]],
    places: pd.DataFrame,
    restaurant_name: str,
    preferences: dict[str, Any],
    source: str,
) -> list[dict[str, Any]]:
    """
    Gemini 또는 Python 결과를
    화면에서 사용할 코스 구조로 바꾼다.
    """
    restaurant = restaurant_row(
        places,
        preferences["region"],
        restaurant_name,
    )

    if restaurant is None:
        return []

    candidates = candidate_rows(
        places,
        preferences["region"],
    )

    by_name = {
        str(
            item.get(
                "이름",
                "",
            )
        ): item
        for item in candidates
    }

    routes: list[dict[str, Any]] = []

    for (
        route_index,
        raw_route,
    ) in enumerate(
        raw_routes[:3]
    ):
        selected: list[dict[str, Any]] = []
        used: set[str] = set()

        for suggestion in raw_route.get(
            "places",
            [],
        ):
            name = str(
                suggestion.get(
                    "name",
                    "",
                )
            ).strip()

            original = by_name.get(name)

            if (
                original is None
                or name in used
            ):
                continue

            used.add(name)

            category = str(
                original.get(
                    "카테고리",
                    "장소",
                )
            )

            selected.append(
                {
                    "name": name,
                    "category": category,
                    "address": str(
                        original.get(
                            "주소",
                            "",
                        )
                    ),
                    "reason": (
                        str(
                            suggestion.get(
                                "reason",
                                "",
                            )
                        ).strip()
                        or default_reason(
                            category,
                            preferences[
                                "companion"
                            ],
                        )
                    ),
                    "tags": optional_text(
                        original,
                        "태그",
                        "키워드",
                    ),
                    "duration_text": (
                        optional_text(
                            original,
                            "체류시간",
                            "예상체류시간",
                            "소요시간",
                        )
                    ),
                }
            )

        if not selected:
            continue

        # 마지막 추천 장소는
        # 식사 후 일정으로 사용한다.
        if len(selected) >= 2:
            before_places = (
                selected[:-1]
            )

            after_place = (
                selected[-1]
            )

        else:
            before_places = selected
            after_place = None

        wait = int(
            preferences["wait"]
        )

        suggested_each = max(
            15,
            min(
                30,
                max(
                    20,
                    wait - 10,
                )
                // max(
                    1,
                    len(before_places),
                ),
            ),
        )

        stops: list[dict[str, Any]] = []

        for (
            place_index,
            place,
        ) in enumerate(
            before_places
        ):
            duration = min(
                parse_minutes(
                    place[
                        "duration_text"
                    ],
                    suggested_each,
                ),
                suggested_each,
            )

            stops.append(
                {
                    "phase": "대기 활용",
                    "time": (
                        "지금 출발"
                        if place_index == 0
                        else (
                            "다음 "
                            f"{place_index + 1}번째"
                        )
                    ),
                    "category": (
                        place["category"]
                    ),
                    "name": place["name"],
                    "address": (
                        place["address"]
                    ),
                    "duration": duration,
                    "reason": (
                        place["reason"]
                    ),
                    "tags": place["tags"],
                }
            )

        # 선택 맛집을 코스 중간에 삽입한다.
        stops.append(
            {
                "phase": "식사",
                "time": (
                    f"대기 약 {wait}분 후"
                ),
                "category": "맛집",
                "name": restaurant_name,
                "address": str(
                    restaurant.get(
                        "주소",
                        "",
                    )
                ),
                "duration": parse_minutes(
                    optional_text(
                        restaurant,
                        "체류시간",
                        "예상체류시간",
                    ),
                    60,
                ),
                "reason": (
                    "대기 순서에 맞춰 "
                    "선택한 맛집으로 돌아와 "
                    "식사를 이어가요."
                ),
                "tags": optional_text(
                    restaurant,
                    "대표메뉴",
                    "태그",
                ),
            }
        )

        # 식사 후 장소를 마지막에 붙인다.
        if after_place:
            stops.append(
                {
                    "phase": "식후 추천",
                    "time": "식사 후",
                    "category": (
                        after_place[
                            "category"
                        ]
                    ),
                    "name": (
                        after_place["name"]
                    ),
                    "address": (
                        after_place[
                            "address"
                        ]
                    ),
                    "duration": (
                        parse_minutes(
                            after_place[
                                "duration_text"
                            ],
                            DEFAULT_DURATION.get(
                                after_place[
                                    "category"
                                ],
                                30,
                            ),
                        )
                    ),
                    "reason": (
                        after_place[
                            "reason"
                        ]
                    ),
                    "tags": (
                        after_place["tags"]
                    ),
                }
            )

        (
            fallback_title,
            fallback_badge,
            fallback_summary,
        ) = FALLBACK_ROUTES[
            min(
                route_index,
                2,
            )
        ]

        route = {
            "title": (
                str(
                    raw_route.get(
                        "title",
                        "",
                    )
                ).strip()
                or fallback_title
            ),
            "badge": (
                str(
                    raw_route.get(
                        "badge",
                        "",
                    )
                ).strip()
                or fallback_badge
            ),
            "summary": (
                str(
                    raw_route.get(
                        "summary",
                        "",
                    )
                ).strip()
                or fallback_summary
            ),
            "region": (
                preferences["region"]
            ),
            "restaurant": (
                restaurant_name
            ),
            "wait": wait,
            "companion": (
                preferences[
                    "companion"
                ]
            ),
            "meal": preferences["meal"],
            "interests": list(
                preferences.get(
                    "interests",
                    [],
                )
            ),
            "stops": stops,
            "local_stay": sum(
                stop["duration"]
                for stop in stops
                if (
                    stop["category"]
                    != "맛집"
                )
            ),
            "source": source,
        }

        route["id"] = route_id(route)
        routes.append(route)

    return routes


def create_recommendations(
    places: pd.DataFrame,
    restaurant: str,
    preferences: dict[str, Any],
    variation: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Gemini 추천을 먼저 시도한다.

    실패하면 앱이 멈추지 않고
    Python 기본 추천으로 전환한다.
    """
    candidates = candidate_rows(
        places,
        preferences["region"],
    )

    result = generate_gemini_routes(
        region=preferences["region"],
        restaurant=restaurant,
        wait_minutes=int(
            preferences["wait"]
        ),
        companion=preferences[
            "companion"
        ],
        meal_time=preferences["meal"],
        interests=preferences.get(
            "interests",
            [],
        ),
        candidates=candidates,
        variation=variation,
    )

    if result["success"]:
        badges = (
            "AI 취향형",
            "AI 발견형",
            "AI 균형형",
        )

        raw_routes = [
            {
                **route,
                "badge": badges[index],
            }
            for index, route in enumerate(
                result["routes"][:3]
            )
        ]

        routes = assemble_routes(
            raw_routes,
            places,
            restaurant,
            preferences,
            "gemini",
        )

        if routes:
            return (
                routes,
                {
                    "source": "gemini",
                    "model": result["model"],
                    "latency_seconds": (
                        result[
                            "latency_seconds"
                        ]
                    ),
                    "interaction_id": (
                        result[
                            "interaction_id"
                        ]
                    ),
                    "repair_count": (
                        result[
                            "repair_count"
                        ]
                    ),
                    "warnings": result.get(
                        "warnings",
                        [],
                    ),
                    "error": "",
                },
            )

    fallback = fallback_raw_routes(
        candidates,
        preferences,
        variation,
    )

    routes = assemble_routes(
        fallback,
        places,
        restaurant,
        preferences,
        "python",
    )

    return (
        routes,
        {
            "source": "python",
            "model": result.get(
                "model",
                "",
            ),
            "latency_seconds": (
                result.get(
                    "latency_seconds",
                    0.0,
                )
            ),
            "interaction_id": "",
            "repair_count": 0,
            "warnings": [],
            "error": result.get(
                "error",
                (
                    "Gemini 추천 결과를 "
                    "적용하지 못했습니다."
                ),
            ),
        },
    )


def active_route() -> dict[str, Any] | None:
    """
    현재 선택된 추천 코스를 반환한다.
    """
    if not st.session_state.routes:
        return None

    index = max(
        0,
        min(
            st.session_state.route_index,
            len(
                st.session_state.routes
            )
            - 1,
        ),
    )

    return st.session_state.routes[
        index
    ]


def save_active_route() -> None:
    """
    현재 코스를 MY 일정에 저장한다.
    """
    route = active_route()

    if not route:
        return

    saved_ids = {
        item["id"]
        for item in st.session_state.saved
    }

    if route["id"] in saved_ids:
        st.toast(
            "이미 저장한 코스예요.",
            icon="ℹ️",
        )

        return

    saved = deepcopy(route)

    saved["saved_at"] = (
        datetime.now().strftime(
            "%m.%d %H:%M"
        )
    )

    st.session_state.saved.insert(
        0,
        saved,
    )

    st.toast(
        "MY 일정에 저장했어요.",
        icon="✅",
    )


# -----------------------------------------------------------------------------
# 시연용 대기표
# -----------------------------------------------------------------------------
def create_queue(
    region: str,
    restaurant: str,
    wait: int,
) -> dict[str, Any]:
    """
    시연용 대기표를 만든다.
    """
    now = datetime.now()

    code = re.sub(
        r"[^가-힣A-Za-z0-9]",
        "",
        region,
    )[:2] or "GJ"

    number = (
        100
        + stable_number(
            (
                f"{restaurant}|"
                f"{now.isoformat()}"
            ),
            900,
        )
    )

    return {
        "ticket": (
            f"{code}-{number}"
        ),
        "restaurant": restaurant,
        "wait": int(wait),
        "started_at": (
            now.isoformat(
                timespec="seconds"
            )
        ),
        "demo_elapsed": 0,
    }


def queue_status() -> tuple[
    int,
    int,
    int,
]:
    """
    남은 시간, 진행률, 앞 팀 수를 계산한다.
    """
    queue = st.session_state.queue

    if not queue:
        return 0, 0, 0

    try:
        started = datetime.fromisoformat(
            queue["started_at"]
        )

        actual_elapsed = max(
            0,
            int(
                (
                    datetime.now()
                    - started
                ).total_seconds()
                // 60
            ),
        )

    except (
        ValueError,
        TypeError,
        KeyError,
    ):
        actual_elapsed = 0

    wait = max(
        1,
        int(queue["wait"]),
    )

    elapsed = min(
        wait,
        actual_elapsed
        + int(
            queue.get(
                "demo_elapsed",
                0,
            )
        ),
    )

    remaining = max(
        0,
        wait - elapsed,
    )

    progress = int(
        elapsed
        / wait
        * 100
    )

    teams = (
        0
        if remaining == 0
        else max(
            1,
            (
                remaining + 4
            )
            // 5,
        )
    )

    return (
        remaining,
        progress,
        teams,
    )


# -----------------------------------------------------------------------------
# 모바일 디자인 CSS
# -----------------------------------------------------------------------------
def inject_css() -> None:
    ui(
        """
        <style>
        :root {
            --green-900: #073f38;
            --green-800: #075b50;
            --green-700: #08796a;
            --green-500: #18a88f;
            --green-100: #e8f6f2;
            --cream: #fffaf2;
            --orange: #ff7658;
            --yellow: #ffc85a;
            --ink: #17231f;
            --muted: #66756f;
            --line: #dce9e4;
            --paper: #fbfdfc;
        }

        html,
        body,
        [class*="css"] {
            font-family:
                Pretendard,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                "Noto Sans KR",
                sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(
                    circle at 8% 2%,
                    rgba(24, 168, 143, 0.16),
                    transparent 28%
                ),
                radial-gradient(
                    circle at 96% 18%,
                    rgba(255, 118, 88, 0.12),
                    transparent 24%
                ),
                #e9efec;
        }

        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu,
        footer {
            display: none !important;
        }

        [data-testid="stMainBlockContainer"],
        .block-container {
            width: 100% !important;
            max-width: 470px !important;
            min-height: 100vh;
            padding:
                1rem
                1rem
                7.8rem !important;
            background: var(--paper);
            box-shadow:
                0 0 50px
                rgba(25, 55, 47, 0.13);
        }

        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.7rem;
            margin: 0.05rem 0 1rem;
        }

        .brand-wrap {
            display: flex;
            align-items: center;
            gap: 0.68rem;
        }

        .brand-logo {
            width: 2.72rem;
            height: 2.72rem;
            display: grid;
            place-items: center;
            border-radius: 1rem;
            color: white;
            font-size: 1.05rem;
            font-weight: 950;
            background:
                linear-gradient(
                    145deg,
                    var(--green-900),
                    var(--green-500)
                );
            box-shadow:
                0 10px 22px
                rgba(8, 121, 106, 0.24);
        }

        .brand-name {
            color: var(--ink);
            font-size: 1.02rem;
            font-weight: 950;
            line-height: 1.05;
        }

        .brand-sub {
            margin-top: 0.22rem;
            color: var(--muted);
            font-size: 0.66rem;
            font-weight: 750;
        }

        .point-pill {
            padding: 0.55rem 0.72rem;
            border:
                1px solid
                var(--line);
            border-radius: 999px;
            background: white;
            color: var(--green-800);
            font-size: 0.78rem;
            font-weight: 900;
            box-shadow:
                0 6px 15px
                rgba(30, 61, 53, 0.05);
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding:
                1.5rem
                1.35rem
                1.35rem;
            border-radius: 1.7rem;
            color: white;
            background:
                linear-gradient(
                    145deg,
                    var(--green-900),
                    var(--green-700) 58%,
                    #1ca88f
                );
            box-shadow:
                0 20px 38px
                rgba(7, 91, 80, 0.23);
        }

        .hero::before {
            content: "";
            position: absolute;
            width: 11rem;
            height: 11rem;
            right: -5rem;
            top: -5rem;
            border-radius: 50%;
            background:
                rgba(
                    255,
                    255,
                    255,
                    0.12
                );
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 6.5rem;
            height: 6.5rem;
            right: 1rem;
            bottom: -4.6rem;
            border-radius: 50%;
            background:
                rgba(
                    255,
                    200,
                    90,
                    0.28
                );
        }

        .hero > * {
            position: relative;
            z-index: 1;
        }

        .hero-chip {
            display: inline-flex;
            padding: 0.34rem 0.62rem;
            border-radius: 999px;
            background:
                rgba(
                    255,
                    255,
                    255,
                    0.15
                );
            font-size: 0.68rem;
            font-weight: 850;
        }

        .hero-title {
            margin:
                0.82rem
                0
                0.48rem;
            max-width: 20rem;
            color: white;
            font-size: 1.72rem;
            line-height: 1.2;
            font-weight: 950;
            letter-spacing: -0.045em;
        }

        .hero-copy {
            margin: 0;
            max-width: 20rem;
            color:
                rgba(
                    255,
                    255,
                    255,
                    0.84
                );
            font-size: 0.82rem;
        }

        .hero-orbit {
            position: absolute;
            right: 1.1rem;
            top: 4.5rem;
            width: 4.25rem;
            height: 4.25rem;
            display: grid;
            place-items: center;
            border-radius: 1.35rem;
            background:
                rgba(
                    255,
                    255,
                    255,
                    0.13
                );
            border:
                1px solid
                rgba(
                    255,
                    255,
                    255,
                    0.18
                );
            font-size: 1.85rem;
            transform: rotate(6deg);
        }

        .stats-grid {
            display: grid;
            grid-template-columns:
                repeat(3, 1fr);
            gap: 0.52rem;
            margin:
                0.82rem
                0
                1.25rem;
        }

        .stat-card {
            padding:
                0.78rem
                0.42rem;
            border:
                1px solid
                var(--line);
            border-radius: 1rem;
            background: white;
            text-align: center;
            box-shadow:
                0 7px 17px
                rgba(36, 65, 57, 0.045);
        }

        .stat-card strong {
            display: block;
            color: var(--ink);
            font-size: 1.02rem;
        }

        .stat-card span {
            color: var(--muted);
            font-size: 0.64rem;
        }

        .section-head {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 0.6rem;
            margin:
                1.28rem
                0
                0.7rem;
        }

        .section-eyebrow {
            color: var(--green-700);
            font-size: 0.63rem;
            font-weight: 950;
            letter-spacing: 0.09em;
        }

        .section-title {
            margin-top: 0.14rem;
            color: var(--ink);
            font-size: 1.14rem;
            font-weight: 950;
            letter-spacing: -0.025em;
        }

        .section-caption {
            color: var(--muted);
            font-size: 0.66rem;
            text-align: right;
        }

        .ai-ready,
        .ai-proof {
            display: flex;
            align-items: center;
            gap: 0.72rem;
            padding:
                0.82rem
                0.9rem;
            border-radius: 1.1rem;
            border:
                1px solid
                #cfe9e1;
            background:
                linear-gradient(
                    135deg,
                    #f1fbf8,
                    #ffffff
                );
            box-shadow:
                0 7px 17px
                rgba(36, 65, 57, 0.045);
        }

        .ai-ready {
            margin:
                0.75rem
                0
                1rem;
        }

        .ai-proof {
            margin:
                0.72rem
                0
                0.95rem;
        }

        .ai-proof.fallback {
            border-color: #f0d9ba;
            background:
                linear-gradient(
                    135deg,
                    #fff8ed,
                    #ffffff
                );
        }

        .ai-symbol {
            flex: 0 0 auto;
            width: 2.35rem;
            height: 2.35rem;
            display: grid;
            place-items: center;
            border-radius: 0.85rem;
            background:
                var(--green-700);
            color: white;
            font-size: 1.05rem;
            box-shadow:
                0 7px 15px
                rgba(8, 121, 106, 0.2);
        }

        .ai-proof.fallback
        .ai-symbol {
            background: #d88931;
        }

        .ai-copy {
            min-width: 0;
            flex: 1;
        }

        .ai-copy strong {
            display: block;
            color: var(--ink);
            font-size: 0.79rem;
        }

        .ai-copy span {
            display: block;
            margin-top: 0.16rem;
            color: var(--muted);
            font-size: 0.65rem;
        }

        .ai-check {
            color: var(--green-700);
            font-size: 1rem;
            font-weight: 950;
        }

        .queue-card {
            padding: 1rem;
            border:
                1px solid
                var(--line);
            border-radius: 1.25rem;
            background:
                linear-gradient(
                    145deg,
                    #ffffff,
                    #effaf7
                );
            box-shadow:
                0 9px 21px
                rgba(36, 65, 57, 0.06);
        }

        .queue-top {
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
        }

        .queue-ticket {
            color: var(--green-700);
            font-size: 0.64rem;
            font-weight: 950;
        }

        .queue-name {
            margin-top: 0.22rem;
            color: var(--ink);
            font-size: 0.98rem;
            font-weight: 950;
        }

        .queue-right {
            text-align: right;
        }

        .queue-right strong {
            display: block;
            color: var(--green-800);
            font-size: 1.32rem;
            line-height: 1.05;
        }

        .queue-right span {
            color: var(--muted);
            font-size: 0.64rem;
        }

        .progress-track {
            height: 0.58rem;
            margin-top: 0.85rem;
            overflow: hidden;
            border-radius: 999px;
            background: #dbe9e4;
        }

        .progress-bar {
            height: 100%;
            border-radius: 999px;
            background:
                linear-gradient(
                    90deg,
                    var(--green-700),
                    #28b696
                );
        }

        .queue-bottom {
            display: flex;
            justify-content: space-between;
            margin-top: 0.45rem;
            color: var(--muted);
            font-size: 0.62rem;
        }

        .route-hero {
            position: relative;
            overflow: hidden;
            padding: 1.15rem;
            margin:
                0.78rem
                0
                1rem;
            border-radius: 1.4rem;
            color: white;
            background:
                linear-gradient(
                    140deg,
                    #183a32,
                    var(--green-700)
                );
            box-shadow:
                0 15px 30px
                rgba(20, 64, 54, 0.2);
        }

        .route-hero::after {
            content: "";
            position: absolute;
            width: 7rem;
            height: 7rem;
            right: -2.8rem;
            top: -3rem;
            border-radius: 50%;
            background:
                rgba(
                    255,
                    255,
                    255,
                    0.1
                );
        }

        .route-badge {
            position: relative;
            display: inline-flex;
            padding: 0.28rem 0.55rem;
            border-radius: 999px;
            background:
                rgba(
                    255,
                    255,
                    255,
                    0.15
                );
            font-size: 0.62rem;
            font-weight: 900;
        }

        .route-title {
            position: relative;
            margin:
                0.55rem
                0
                0.28rem;
            color: white;
            font-size: 1.2rem;
            font-weight: 950;
            letter-spacing: -0.02em;
        }

        .route-summary {
            position: relative;
            margin: 0;
            max-width: 20rem;
            color:
                rgba(
                    255,
                    255,
                    255,
                    0.82
                );
            font-size: 0.75rem;
        }

        .route-meta {
            position: relative;
            display: grid;
            grid-template-columns:
                repeat(3, 1fr);
            gap: 0.42rem;
            margin-top: 0.85rem;
        }

        .route-meta-item {
            padding:
                0.5rem
                0.25rem;
            border-radius: 0.78rem;
            background:
                rgba(
                    255,
                    255,
                    255,
                    0.1
                );
            text-align: center;
        }

        .route-meta-item strong {
            display: block;
            color: white;
            font-size: 0.78rem;
        }

        .route-meta-item span {
            color:
                rgba(
                    255,
                    255,
                    255,
                    0.68
                );
            font-size: 0.57rem;
        }

        .timeline-row {
            display: grid;
            grid-template-columns:
                2.55rem
                1fr;
            gap: 0.65rem;
        }

        .timeline-rail {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .step-dot {
            width: 2.3rem;
            height: 2.3rem;
            display: grid;
            place-items: center;
            z-index: 1;
            border-radius: 0.85rem;
            color: white;
            font-size: 0.82rem;
            font-weight: 950;
            background:
                var(--green-700);
            box-shadow:
                0 7px 15px
                rgba(8, 121, 106, 0.18);
        }

        .step-dot.food {
            background: var(--orange);
        }

        .step-dot.culture {
            background: #7c6bd3;
        }

        .step-dot.cafe {
            background: #a96d3b;
        }

        .step-line {
            width: 2px;
            flex: 1;
            min-height: 1rem;
            background: #dbe9e4;
        }

        .place-card {
            margin-bottom: 0.72rem;
            padding:
                0.88rem
                0.9rem;
            border:
                1px solid
                var(--line);
            border-radius: 1.1rem;
            background: white;
            box-shadow:
                0 8px 20px
                rgba(36, 65, 57, 0.055);
        }

        .place-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.45rem;
        }

        .phase-chip {
            display: inline-flex;
            padding: 0.24rem 0.48rem;
            border-radius: 999px;
            background:
                var(--green-100);
            color: var(--green-800);
            font-size: 0.59rem;
            font-weight: 950;
        }

        .phase-chip.food {
            background: #fff0ec;
            color: #c24f36;
        }

        .place-time {
            color: var(--muted);
            font-size: 0.6rem;
            text-align: right;
        }

        .place-name {
            margin-top: 0.48rem;
            color: var(--ink);
            font-size: 0.94rem;
            font-weight: 950;
        }

        .place-reason {
            margin-top: 0.27rem;
            color: #51615b;
            font-size: 0.7rem;
            line-height: 1.5;
        }

        .place-address {
            margin-top: 0.42rem;
            color: #7a8782;
            font-size: 0.62rem;
        }

        .place-footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.55rem;
            margin-top: 0.55rem;
        }

        .tag-text {
            color: var(--orange);
            font-size: 0.6rem;
            font-weight: 850;
        }

        .map-link {
            display: inline-flex;
            align-items: center;
            padding: 0.32rem 0.52rem;
            border-radius: 0.65rem;
            background: #f1f6f4;
            color:
                var(--green-800)
                !important;
            font-size: 0.61rem;
            font-weight: 900;
            text-decoration:
                none
                !important;
        }

        .reward-card {
            margin:
                0.85rem
                0;
            padding: 0.95rem;
            border:
                1px solid
                #f0dca8;
            border-radius: 1.18rem;
            background:
                linear-gradient(
                    145deg,
                    #fff9eb,
                    #ffffff
                );
            box-shadow:
                0 8px 20px
                rgba(82, 65, 25, 0.045);
        }

        .reward-card small {
            color: #a86c00;
            font-size: 0.6rem;
            font-weight: 950;
            letter-spacing: 0.08em;
        }

        .reward-card strong {
            display: block;
            margin:
                0.3rem
                0
                0.2rem;
            color: var(--ink);
            font-size: 0.88rem;
        }

        .reward-card span {
            color: var(--muted);
            font-size: 0.68rem;
        }

        .empty-card,
        .saved-card,
        .simple-card {
            padding: 1rem;
            margin-bottom: 0.72rem;
            border:
                1px solid
                var(--line);
            border-radius: 1.18rem;
            background: white;
            box-shadow:
                0 8px 20px
                rgba(36, 65, 57, 0.055);
        }

        .empty-card {
            padding:
                1.45rem
                1rem;
            text-align: center;
        }

        .empty-icon {
            font-size: 2rem;
        }

        .empty-card strong {
            display: block;
            margin:
                0.4rem
                0
                0.22rem;
            color: var(--ink);
        }

        .empty-card span {
            color: var(--muted);
            font-size: 0.76rem;
        }

        .card-kicker {
            color: var(--green-700);
            font-size: 0.62rem;
            font-weight: 950;
        }

        .card-title {
            margin-top: 0.25rem;
            color: var(--ink);
            font-size: 0.94rem;
            font-weight: 950;
        }

        .card-copy {
            margin-top: 0.3rem;
            color: var(--muted);
            font-size: 0.68rem;
        }

        .done-pill {
            display: inline-flex;
            margin-top: 0.52rem;
            padding: 0.28rem 0.52rem;
            border-radius: 999px;
            background:
                var(--green-100);
            color: var(--green-800);
            font-size: 0.62rem;
            font-weight: 900;
        }

        .saved-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.5rem;
        }

        .saved-time {
            color: var(--muted);
            font-size: 0.6rem;
            white-space: nowrap;
        }

        .history-row {
            display: flex;
            justify-content: space-between;
            gap: 0.6rem;
            padding:
                0.67rem
                0;
            border-bottom:
                1px solid
                #edf2f0;
        }

        .history-row:last-child {
            border-bottom: 0;
        }

        .history-name {
            color: var(--ink);
            font-size: 0.74rem;
            font-weight: 850;
        }

        .history-time {
            margin-top: 0.15rem;
            color: var(--muted);
            font-size: 0.61rem;
        }

        .history-point {
            color: var(--green-700);
            font-size: 0.72rem;
            font-weight: 950;
        }

        div[data-baseweb="select"]
        > div,
        [data-testid="stTextInput"]
        input {
            min-height: 3rem;
            border-radius:
                0.92rem
                !important;
            border-color:
                #d9e6e1
                !important;
            background:
                white
                !important;
        }

        [data-testid="stWidgetLabel"]
        p {
            color: #40504a;
            font-size: 0.74rem;
            font-weight: 850;
        }

        .stButton > button {
            width: 100%;
            min-height: 3rem;
            border-radius: 0.93rem;
            border:
                1px solid
                #d6e4df;
            font-weight: 900;
        }

        .stButton
        > button[kind="primary"] {
            border: 0;
            color: white;
            background:
                linear-gradient(
                    135deg,
                    var(--green-900),
                    #0e927a
                );
            box-shadow:
                0 10px 20px
                rgba(8, 120, 106, 0.18);
        }

        [data-testid="stCameraInput"] {
            overflow: hidden;
            border:
                1px solid
                var(--line);
            border-radius: 1.2rem;
            background: white;
        }

        .st-key-home_companion
        [role="radiogroup"],
        .st-key-home_meal
        [role="radiogroup"] {
            display:
                grid
                !important;
            grid-template-columns:
                repeat(4, 1fr);
            gap: 0.34rem;
        }

        [class*="st-key-route_choice_"]
        [role="radiogroup"] {
            display:
                grid
                !important;
            grid-template-columns:
                repeat(3, 1fr);
            gap: 0.34rem;
        }

        .st-key-home_companion
        label,
        .st-key-home_meal
        label,
        [class*="st-key-route_choice_"]
        label {
            justify-content: center;
            min-height: 2.5rem;
            padding:
                0.3rem
                0.13rem
                !important;
            border:
                1px solid
                #dce7e3;
            border-radius: 0.8rem;
            background: white;
            text-align: center;
        }

        .st-key-home_companion
        label:has(input:checked),
        .st-key-home_meal
        label:has(input:checked),
        [class*="st-key-route_choice_"]
        label:has(input:checked) {
            border-color: #9fd5c8;
            background:
                var(--green-100);
        }

        .st-key-home_companion
        label
        > div:first-child,
        .st-key-home_meal
        label
        > div:first-child,
        [class*="st-key-route_choice_"]
        label
        > div:first-child {
            display: none;
        }

        .st-key-home_companion
        p,
        .st-key-home_meal
        p,
        [class*="st-key-route_choice_"]
        p {
            font-size: 0.67rem;
            font-weight: 850;
        }

        .st-key-app_nav {
            position: fixed;
            left: 50%;
            bottom: 0.62rem;
            transform:
                translateX(-50%);
            z-index: 9999;
            width:
                min(
                    435px,
                    calc(
                        100vw
                        - 1.2rem
                    )
                );
            padding: 0.38rem;
            border:
                1px solid
                rgba(
                    204,
                    220,
                    214,
                    0.96
                );
            border-radius: 1.22rem;
            background:
                rgba(
                    255,
                    255,
                    255,
                    0.95
                );
            box-shadow:
                0 15px 36px
                rgba(29, 57, 49, 0.2);
            backdrop-filter:
                blur(16px);
        }

        .st-key-app_nav
        [data-testid="stWidgetLabel"] {
            display: none;
        }

        .st-key-app_nav
        [role="radiogroup"] {
            display:
                grid
                !important;
            grid-template-columns:
                repeat(4, 1fr);
            gap: 0.22rem;
        }

        .st-key-app_nav
        label {
            justify-content: center;
            min-height: 3rem;
            padding:
                0.3rem
                0.1rem
                !important;
            border-radius: 0.88rem;
            text-align: center;
        }

        .st-key-app_nav
        label:has(input:checked) {
            background:
                var(--green-100);
        }

        .st-key-app_nav
        label
        > div:first-child {
            display: none;
        }

        .st-key-app_nav
        p {
            color: #66736f;
            font-size: 0.66rem;
            font-weight: 850;
        }

        .st-key-app_nav
        label:has(input:checked)
        p {
            color: var(--green-900);
        }

        @media (
            max-width: 640px
        ) {
            [data-testid="stMainBlockContainer"],
            .block-container {
                max-width:
                    none
                    !important;
                padding:
                    0.85rem
                    0.9rem
                    7.5rem
                    !important;
                box-shadow: none;
            }

            .hero-title {
                font-size: 1.55rem;
            }

            .hero-orbit {
                right: 0.7rem;
                opacity: 0.9;
            }

            .st-key-app_nav {
                bottom: 0.38rem;
                width:
                    calc(
                        100vw
                        - 1rem
                    );
            }
        }
        </style>
        """
    )


# -----------------------------------------------------------------------------
# 공통 UI
# -----------------------------------------------------------------------------
def render_header() -> None:
    ui(
        f"""
        <div class="app-header">
            <div class="brand-wrap">
                <div class="brand-logo">
                    W
                </div>

                <div>
                    <div class="brand-name">
                        {APP_NAME}
                    </div>

                    <div class="brand-sub">
                        {APP_SUBTITLE}
                    </div>
                </div>
            </div>

            <div class="point-pill">
                🪙 {st.session_state.points:,}P
            </div>
        </div>
        """
    )


def render_section(
    title: str,
    eyebrow: str = "",
    caption: str = "",
) -> None:
    ui(
        f"""
        <div class="section-head">
            <div>
                <div class="section-eyebrow">
                    {h(eyebrow)}
                </div>

                <div class="section-title">
                    {h(title)}
                </div>
            </div>

            <div class="section-caption">
                {h(caption)}
            </div>
        </div>
        """
    )


def render_empty(
    icon: str,
    title: str,
    copy: str,
) -> None:
    ui(
        f"""
        <div class="empty-card">
            <div class="empty-icon">
                {icon}
            </div>

            <strong>
                {h(title)}
            </strong>

            <span>
                {h(copy)}
            </span>
        </div>
        """
    )


def render_bottom_nav() -> None:
    st.radio(
        "하단 메뉴",
        NAV_ITEMS,
        key=NAV_KEY,
        horizontal=True,
        label_visibility="collapsed",
    )


# -----------------------------------------------------------------------------
# 홈 화면
# -----------------------------------------------------------------------------
def page_home(
    places: pd.DataFrame,
) -> None:
    ui(
        """
        <div class="hero">
            <div class="hero-chip">
                광주·전남 미식 대기 분산 서비스
            </div>

            <div class="hero-title">
                맛집 기다리는 시간,<br>
                작은 여행으로 바꿔보세요.
            </div>

            <p class="hero-copy">
                대기시간과 취향을 고르면
                주변 장소를 엮은 맞춤 코스를 제안해요.
            </p>

            <div class="hero-orbit">
                🧭
            </div>
        </div>
        """
    )

    ui(
        f"""
        <div class="stats-grid">
            <div class="stat-card">
                <strong>
                    {len(places)}
                </strong>

                <span>
                    등록 장소
                </span>
            </div>

            <div class="stat-card">
                <strong>
                    {places["지역"].nunique()}
                </strong>

                <span>
                    서비스 지역
                </span>
            </div>

            <div class="stat-card">
                <strong>
                    {len(st.session_state.checkins)}
                </strong>

                <span>
                    내 체크인
                </span>
            </div>
        </div>
        """
    )

    status = get_gemini_status()

    if status["configured"]:
        ui(
            """
            <div class="ai-ready">
                <div class="ai-symbol">
                    ✦
                </div>

                <div class="ai-copy">
                    <strong>
                        AI 맞춤 추천 준비 완료
                    </strong>

                    <span>
                        선택한 지역의 CSV 장소만 이용해
                        코스를 만들어요.
                    </span>
                </div>

                <div class="ai-check">
                    ✓
                </div>
            </div>
            """
        )

    else:
        ui(
            """
            <div class="ai-proof fallback">
                <div class="ai-symbol">
                    !
                </div>

                <div class="ai-copy">
                    <strong>
                        AI 키를 확인해 주세요
                    </strong>

                    <span>
                        키가 없어도 기본 추천으로
                        앱은 계속 동작해요.
                    </span>
                </div>
            </div>
            """
        )

    render_section(
        "오늘의 대기 코스 만들기",
        "PERSONAL ROUTE",
        "취향 맞춤",
    )

    restaurants = places[
        places["카테고리"]
        == "맛집"
    ]

    regions = sorted(
        restaurants[
            "지역"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if not regions:
        st.warning(
            "CSV에 `맛집` 카테고리가 없습니다."
        )

        return

    if (
        st.session_state.get(
            "home_region"
        )
        not in regions
    ):
        st.session_state.home_region = (
            regions[0]
        )

    region = st.selectbox(
        "지역",
        regions,
        key="home_region",
    )

    restaurant_names = (
        restaurants[
            restaurants["지역"]
            == region
        ]["이름"]
        .tolist()
    )

    if not restaurant_names:
        st.warning(
            "선택한 지역에 맛집이 없습니다."
        )

        return

    if (
        st.session_state.get(
            "home_restaurant"
        )
        not in restaurant_names
    ):
        st.session_state.home_restaurant = (
            restaurant_names[0]
        )

    restaurant = st.selectbox(
        "대기 중인 맛집",
        restaurant_names,
        key="home_restaurant",
    )

    companion = st.radio(
        "동행 유형",
        (
            "혼자",
            "연인",
            "가족",
            "친구",
        ),
        horizontal=True,
        key="home_companion",
    )

    meal = st.radio(
        "식사 시간",
        (
            "점심",
            "오후",
            "저녁",
            "야간",
        ),
        horizontal=True,
        key="home_meal",
    )

    wait = st.select_slider(
        "예상 대기시간",
        options=(
            30,
            45,
            60,
            75,
            90,
            120,
        ),
        value=60,
        format_func=(
            lambda value:
            f"{value}분"
        ),
        key="home_wait",
    )

    interests = st.multiselect(
        "관심 분야",
        tuple(
            INTEREST_CATEGORY
        ),
        default=(
            "자연·산책",
            "카페·디저트",
        ),
        key="home_interests",
    )

    if st.button(
        "✨ AI 맞춤 코스 만들기",
        type="primary",
        width="stretch",
        key="make_route",
    ):
        preferences = {
            "region": region,
            "companion": companion,
            "meal": meal,
            "wait": int(wait),
            "interests": list(
                interests
            ),
        }

        with st.spinner(
            "AI가 취향에 맞는 "
            "코스를 만들고 있어요..."
        ):
            routes, meta = (
                create_recommendations(
                    places,
                    restaurant,
                    preferences,
                    variation=0,
                )
            )

        if not routes:
            st.warning(
                "이 지역에는 코스를 만들 "
                "주변 장소가 부족합니다."
            )

            return

        st.session_state.preferences = (
            preferences
        )

        st.session_state.selected_restaurant = (
            restaurant
        )

        st.session_state.routes = routes
        st.session_state.route_index = 0
        st.session_state.route_variation = 0
        st.session_state.route_generation += 1
        st.session_state.ai_meta = meta

        st.session_state.queue = (
            create_queue(
                region,
                restaurant,
                int(wait),
            )
        )

        if meta["source"] == "gemini":
            st.toast(
                "AI 맞춤 코스가 완성됐어요.",
                icon="✨",
            )

        else:
            st.toast(
                "기본 추천 코스로 먼저 안내할게요.",
                icon="🧭",
            )

        go(COURSE)

    ui(
        """
        <div class="reward-card">
            <small>
                WAITGO EXPERIENCE
            </small>

            <strong>
                코스 선택 → 일정 저장 →
                현장 체크인 → 포인트 적립
            </strong>

            <span>
                맛집 대기시간을 주변 상권과
                관광 경험으로 연결해요.
            </span>
        </div>
        """
    )


# -----------------------------------------------------------------------------
# 코스 화면
# -----------------------------------------------------------------------------
def render_queue_card() -> None:
    if not st.session_state.queue:
        return

    (
        remaining,
        progress,
        teams,
    ) = queue_status()

    queue = st.session_state.queue

    status = (
        "입장 가능"
        if remaining == 0
        else f"{remaining}분 남음"
    )

    team_text = (
        "지금 입장해 주세요"
        if remaining == 0
        else f"앞에 약 {teams}팀"
    )

    ui(
        f"""
        <div class="queue-card">
            <div class="queue-top">
                <div>
                    <div class="queue-ticket">
                        대기번호
                        {h(queue["ticket"])}
                    </div>

                    <div class="queue-name">
                        {h(queue["restaurant"])}
                    </div>
                </div>

                <div class="queue-right">
                    <strong>
                        {h(status)}
                    </strong>

                    <span>
                        {h(team_text)}
                    </span>
                </div>
            </div>

            <div class="progress-track">
                <div
                    class="progress-bar"
                    style="width: {progress}%"
                ></div>
            </div>

            <div class="queue-bottom">
                <span>
                    대기 진행률 {progress}%
                </span>

                <span>
                    시연용 예상 정보
                </span>
            </div>
        </div>
        """
    )

    left, right = st.columns(2)

    with left:
        if st.button(
            "↻ 새로고침",
            width="stretch",
            key="queue_refresh",
        ):
            st.rerun()

    with right:
        if st.button(
            "⏩ 10분 경과",
            width="stretch",
            key="queue_skip",
        ):
            st.session_state.queue[
                "demo_elapsed"
            ] += 10

            st.rerun()


def render_ai_proof() -> None:
    """
    실제 Gemini 호출이 적용됐는지
    코스 화면에서 바로 보여준다.
    """
    meta = st.session_state.ai_meta

    if (
        meta.get("source")
        == "gemini"
    ):
        latency = float(
            meta.get(
                "latency_seconds",
                0.0,
            )
        )

        repair_count = int(
            meta.get(
                "repair_count",
                0,
            )
        )

        detail = (
            f"Gemini 응답 {latency:.1f}초 "
            "· CSV 장소 검증 완료"
        )

        if repair_count:
            detail += (
                f" · {repair_count}건 자동 보정"
            )

        ui(
            f"""
            <div class="ai-proof">
                <div class="ai-symbol">
                    ✦
                </div>

                <div class="ai-copy">
                    <strong>
                        AI 맞춤 추천이 적용됐어요
                    </strong>

                    <span>
                        {h(detail)}
                    </span>
                </div>

                <div class="ai-check">
                    ✓
                </div>
            </div>
            """
        )

    else:
        ui(
            """
            <div class="ai-proof fallback">
                <div class="ai-symbol">
                    ↻
                </div>

                <div class="ai-copy">
                    <strong>
                        기본 추천으로 안내 중이에요
                    </strong>

                    <span>
                        AI 연결이 지연돼도
                        코스 기능은 계속 사용할 수 있어요.
                    </span>
                </div>
            </div>
            """
        )


def render_stop(
    stop: dict[str, Any],
    index: int,
    total: int,
) -> None:
    """
    코스 장소를 타임라인 카드로 표시한다.
    """
    category = stop["category"]

    css_class = CATEGORY_CLASS.get(
        category,
        "nature",
    )

    icon = CATEGORY_ICON.get(
        category,
        "📍",
    )

    query = quote_plus(
        (
            f"{stop['name']} "
            f"{stop['address']}"
        )
    )

    tags = str(
        stop.get(
            "tags",
            "",
        )
    ).strip()

    if tags:
        tag_html = (
            '<span class="tag-text">'
            f'#{h(tags).replace(" ", " #")}'
            "</span>"
        )

    else:
        tag_html = "<span></span>"

    line_html = (
        '<div class="step-line"></div>'
        if index < total
        else ""
    )

    ui(
        f"""
        <div class="timeline-row">
            <div class="timeline-rail">
                <div class="step-dot {css_class}">
                    {icon}
                </div>

                {line_html}
            </div>

            <div class="place-card">
                <div class="place-top">
                    <span class="phase-chip {css_class}">
                        {h(stop["phase"])}
                        ·
                        {h(category)}
                    </span>

                    <span class="place-time">
                        {h(stop["time"])}
                        ·
                        권장 {stop["duration"]}분
                    </span>
                </div>

                <div class="place-name">
                    {h(stop["name"])}
                </div>

                <div class="place-reason">
                    {h(stop["reason"])}
                </div>

                <div class="place-address">
                    📍
                    {h(short_address(stop["address"]))}
                </div>

                <div class="place-footer">
                    {tag_html}

                    <a
                        class="map-link"
                        href="https://www.google.com/maps/search/?api=1&query={query}"
                        target="_blank"
                    >
                        지도 검색 ↗
                    </a>
                </div>
            </div>
        </div>
        """
    )


def page_course(
    places: pd.DataFrame,
) -> None:
    if not st.session_state.routes:
        render_empty(
            "🧭",
            "아직 만든 코스가 없어요",
            (
                "홈에서 맛집과 "
                "대기시간을 선택해 주세요."
            ),
        )

        if st.button(
            "홈에서 코스 만들기",
            type="primary",
            width="stretch",
        ):
            go(HOME)

        return

    render_section(
        "현재 대기 현황",
        "QUEUE",
        "기다림을 여행으로",
    )

    render_queue_card()
    render_ai_proof()

    render_section(
        "추천 코스",
        "CURATED ROUTES",
        "세 가지 조합",
    )

    route_labels = tuple(
        f"코스 {index + 1}"
        for index in range(
            len(
                st.session_state.routes
            )
        )
    )

    radio_key = (
        "route_choice_"
        f"{st.session_state.route_generation}"
    )

    selected_label = st.radio(
        "추천 코스 선택",
        route_labels,
        index=min(
            st.session_state.route_index,
            len(route_labels) - 1,
        ),
        horizontal=True,
        label_visibility="collapsed",
        key=radio_key,
    )

    st.session_state.route_index = (
        route_labels.index(
            selected_label
        )
    )

    route = active_route()

    if route is None:
        return

    ui(
        f"""
        <div class="route-hero">
            <div class="route-badge">
                {h(route["badge"])}
                ·
                {h(route["region"])}
            </div>

            <div class="route-title">
                {h(route["title"])}
            </div>

            <p class="route-summary">
                {h(route["summary"])}
            </p>

            <div class="route-meta">
                <div class="route-meta-item">
                    <strong>
                        {len(route["stops"])}곳
                    </strong>

                    <span>
                        전체 방문지
                    </span>
                </div>

                <div class="route-meta-item">
                    <strong>
                        {h(format_minutes(route["local_stay"]))}
                    </strong>

                    <span>
                        권장 지역 체류
                    </span>
                </div>

                <div class="route-meta-item">
                    <strong>
                        {h(route["companion"])}
                    </strong>

                    <span>
                        동행 유형
                    </span>
                </div>
            </div>
        </div>
        """
    )

    for (
        index,
        stop,
    ) in enumerate(
        route["stops"],
        start=1,
    ):
        render_stop(
            stop,
            index,
            len(route["stops"]),
        )

    ui(
        """
        <div class="reward-card">
            <small>
                CHECK-IN REWARD
            </small>

            <strong>
                추천 장소에서 사진 체크인하면
                +100P
            </strong>

            <span>
                노트북 기본 카메라로
                방문을 기록해 보세요.
            </span>
        </div>
        """
    )

    is_saved = (
        route["id"]
        in {
            item["id"]
            for item
            in st.session_state.saved
        }
    )

    left, right = st.columns(2)

    with left:
        if st.button(
            (
                "✓ 저장됨"
                if is_saved
                else "＋ MY에 담기"
            ),
            disabled=is_saved,
            width="stretch",
            key="save_current_route",
        ):
            save_active_route()
            st.rerun()

    with right:
        if st.button(
            "📷 체크인",
            type="primary",
            width="stretch",
            key="go_checkin",
        ):
            go(CHECKIN)

    if st.button(
        "🔄 같은 조건으로 다시 추천",
        width="stretch",
        key="regenerate_route",
    ):
        st.session_state.route_variation += 1

        with st.spinner(
            "AI가 새로운 장소 조합을 "
            "만들고 있어요..."
        ):
            routes, meta = (
                create_recommendations(
                    places,
                    st.session_state
                    .selected_restaurant,
                    st.session_state
                    .preferences,
                    st.session_state
                    .route_variation,
                )
            )

        if routes:
            st.session_state.routes = routes
            st.session_state.route_index = 0
            st.session_state.route_generation += 1
            st.session_state.ai_meta = meta
            st.rerun()

        else:
            st.warning(
                "새로운 코스를 만들지 못했습니다."
            )

    # Gemini 실패 때만 오류 확인 메뉴가 나타난다.
    if (
        st.session_state
        .ai_meta
        .get("source")
        == "python"
    ):
        with st.expander(
            "AI 연결 오류 확인"
        ):
            st.code(
                st.session_state
                .ai_meta
                .get(
                    "error",
                    "오류 정보 없음",
                ),
                language=None,
            )


# -----------------------------------------------------------------------------
# 체크인 화면
# -----------------------------------------------------------------------------
def page_checkin() -> None:
    route = active_route()

    if (
        route is None
        and st.session_state.saved
    ):
        route = (
            st.session_state.saved[0]
        )

    if route is None:
        render_empty(
            "📷",
            "체크인할 코스가 없어요",
            (
                "먼저 홈에서 "
                "코스를 만들어 주세요."
            ),
        )

        if st.button(
            "홈으로 이동",
            type="primary",
            width="stretch",
        ):
            go(HOME)

        return

    render_section(
        "현장 사진 체크인",
        "CHECK-IN",
        "+100P · 장소별 1회",
    )

    names = [
        stop["name"]
        for stop in route["stops"]
    ]

    if (
        st.session_state.get(
            "checkin_place"
        )
        not in names
    ):
        st.session_state.checkin_place = (
            names[0]
        )

    selected_name = st.selectbox(
        "체크인 장소",
        names,
        key="checkin_place",
    )

    stop = next(
        item
        for item in route["stops"]
        if (
            item["name"]
            == selected_name
        )
    )

    checkin_id = sha1(
        (
            f"{stop['name']}|"
            f"{stop['address']}"
        ).encode("utf-8")
    ).hexdigest()[:14]

    already_checked = any(
        item["id"] == checkin_id
        for item
        in st.session_state.checkins
    )

    done_html = (
        """
        <div class="done-pill">
            ✓ 체크인 완료 · 포인트 지급 완료
        </div>
        """
        if already_checked
        else ""
    )

    ui(
        f"""
        <div class="simple-card">
            <div class="card-kicker">
                {
                    CATEGORY_ICON.get(
                        stop["category"],
                        "📍",
                    )
                }
                {h(stop["category"])}
            </div>

            <div class="card-title">
                {h(stop["name"])}
            </div>

            <div class="card-copy">
                📍
                {h(short_address(stop["address"]))}
            </div>

            {done_html}
        </div>
        """
    )

    st.info(
        "브라우저에서 카메라 권한을 "
        "허용한 뒤 촬영해 주세요."
    )

    picture = st.camera_input(
        "체크인 사진",
        key=f"camera_{checkin_id}",
    )

    if picture is not None:
        if st.button(
            (
                "이미 체크인한 장소예요"
                if already_checked
                else "체크인 확정 +100P"
            ),
            type="primary",
            disabled=already_checked,
            width="stretch",
            key=f"confirm_{checkin_id}",
        ):
            st.session_state.checkins.insert(
                0,
                {
                    "id": checkin_id,
                    "name": stop["name"],
                    "category": (
                        stop["category"]
                    ),
                    "checked_at": (
                        datetime.now().strftime(
                            "%m.%d %H:%M"
                        )
                    ),
                    "points": 100,
                    "image_hash": sha256(
                        picture.getvalue()
                    ).hexdigest()[:12],
                },
            )

            st.session_state.points += 100

            st.toast(
                (
                    "체크인 성공! "
                    "100P를 적립했어요."
                ),
                icon="🎉",
            )

            st.rerun()

    render_section(
        "오늘의 방문 미션",
        "PROGRESS",
    )

    completed = len(
        st.session_state.checkins
    )

    percent = min(
        100,
        int(
            completed
            / 3
            * 100
        ),
    )

    ui(
        f"""
        <div class="simple-card">
            <div class="card-kicker">
                지역 체류 미션
            </div>

            <div class="card-title">
                {completed}/3곳 체크인 완료
            </div>

            <div class="progress-track">
                <div
                    class="progress-bar"
                    style="width: {percent}%"
                ></div>
            </div>

            <div class="card-copy">
                3곳 방문 시 총 300P
            </div>
        </div>
        """
    )


# -----------------------------------------------------------------------------
# MY 화면
# -----------------------------------------------------------------------------
def page_my(
    places: pd.DataFrame,
) -> None:
    render_section(
        "나의 대기여행",
        "MY WAITGO",
        "일정과 포인트",
    )

    total_stay = sum(
        route.get(
            "local_stay",
            0,
        )
        for route
        in st.session_state.saved
    )

    ui(
        f"""
        <div class="stats-grid">
            <div class="stat-card">
                <strong>
                    {st.session_state.points:,}P
                </strong>

                <span>
                    적립 포인트
                </span>
            </div>

            <div class="stat-card">
                <strong>
                    {len(st.session_state.saved)}
                </strong>

                <span>
                    저장 코스
                </span>
            </div>

            <div class="stat-card">
                <strong>
                    {h(format_minutes(total_stay))}
                </strong>

                <span>
                    권장 체류
                </span>
            </div>
        </div>
        """
    )

    render_section(
        "저장한 코스",
        "SAVED",
        f"{len(st.session_state.saved)}개",
    )

    if not st.session_state.saved:
        render_empty(
            "🗂️",
            "저장한 코스가 없어요",
            (
                "추천 코스에서 "
                "MY에 담기를 눌러보세요."
            ),
        )

    else:
        for (
            index,
            route,
        ) in enumerate(
            list(
                st.session_state.saved
            )
        ):
            ui(
                f"""
                <div class="saved-card">
                    <div class="saved-head">
                        <div>
                            <div class="card-kicker">
                                {h(route["region"])}
                                ·
                                {h(route["badge"])}
                            </div>

                            <div class="card-title">
                                {h(route["title"])}
                            </div>
                        </div>

                        <div class="saved-time">
                            {h(route.get("saved_at"))}
                        </div>
                    </div>

                    <div class="card-copy">
                        {h(route["restaurant"])}
                        중심 ·
                        {len(route["stops"])}개 장소 ·
                        {h(format_minutes(route["local_stay"]))}
                    </div>
                </div>
                """
            )

            left, right = st.columns(
                [
                    2,
                    1,
                ]
            )

            with left:
                if st.button(
                    "코스 열기",
                    width="stretch",
                    key=(
                        f"open_"
                        f"{route['id']}_"
                        f"{index}"
                    ),
                ):
                    st.session_state.routes = [
                        deepcopy(route)
                    ]

                    st.session_state.route_index = 0
                    st.session_state.route_generation += 1

                    st.session_state.selected_restaurant = (
                        route["restaurant"]
                    )

                    st.session_state.preferences = {
                        "region": (
                            route["region"]
                        ),
                        "companion": (
                            route["companion"]
                        ),
                        "meal": route.get(
                            "meal",
                            "점심",
                        ),
                        "wait": int(
                            route.get(
                                "wait",
                                60,
                            )
                        ),
                        "interests": list(
                            route.get(
                                "interests",
                                [],
                            )
                        ),
                    }

                    st.session_state.queue = (
                        create_queue(
                            route["region"],
                            route["restaurant"],
                            int(
                                route.get(
                                    "wait",
                                    60,
                                )
                            ),
                        )
                    )

                    st.session_state.ai_meta = {
                        "source": route.get(
                            "source",
                            "python",
                        ),
                        "model": "",
                        "latency_seconds": 0.0,
                        "interaction_id": "",
                        "repair_count": 0,
                        "warnings": [],
                        "error": "",
                    }

                    go(COURSE)

            with right:
                if st.button(
                    "삭제",
                    width="stretch",
                    key=(
                        f"delete_"
                        f"{route['id']}_"
                        f"{index}"
                    ),
                ):
                    st.session_state.saved = [
                        item
                        for item
                        in st.session_state.saved
                        if (
                            item["id"]
                            != route["id"]
                        )
                    ]

                    st.rerun()

    render_section(
        "체크인 기록",
        "HISTORY",
        f"{len(st.session_state.checkins)}건",
    )

    if not st.session_state.checkins:
        render_empty(
            "📍",
            "체크인 기록이 없어요",
            (
                "추천 장소에서 "
                "카메라 체크인을 해보세요."
            ),
        )

    else:
        rows = ""

        for record in (
            st.session_state.checkins
        ):
            rows += f"""
            <div class="history-row">
                <div>
                    <div class="history-name">
                        {
                            CATEGORY_ICON.get(
                                record["category"],
                                "📍",
                            )
                        }
                        {h(record["name"])}
                    </div>

                    <div class="history-time">
                        {h(record["checked_at"])}
                    </div>
                </div>

                <div class="history-point">
                    +{record["points"]}P
                </div>
            </div>
            """

        ui(
            f"""
            <div class="simple-card">
                {rows}
            </div>
            """
        )

    render_section(
        "연결 상태",
        "DEVELOPER CHECK",
        "시연 준비 확인",
    )

    status = get_gemini_status()
    meta = st.session_state.ai_meta

    with st.expander(
        "AI 및 데이터 상태 보기"
    ):
        st.write(
            "**Gemini 키:** "
            + (
                "설정됨"
                if status["configured"]
                else "키 없음"
            )
        )

        st.write(
            "**사용 모델:** "
            f"`{status['model']}`"
        )

        st.write(
            "**CSV 장소 수:** "
            f"`{len(places)}개`"
        )

        if meta.get("source"):
            recent_source = (
                "`Gemini API`"
                if (
                    meta["source"]
                    == "gemini"
                )
                else "`Python 기본 추천`"
            )

            st.write(
                "**최근 추천 방식:** "
                + recent_source
            )

        if meta.get(
            "latency_seconds"
        ):
            st.write(
                "**AI 응답시간:** "
                f"`{meta['latency_seconds']:.2f}초`"
            )

        if meta.get(
            "interaction_id"
        ):
            st.write(
                "**Interaction ID:** "
                f"`{meta['interaction_id']}`"
            )

        if meta.get("error"):
            st.caption(
                "최근 AI 오류"
            )

            st.code(
                meta["error"],
                language=None,
            )

        if st.button(
            "CSV 캐시 비우고 다시 읽기",
            width="stretch",
        ):
            st.cache_data.clear()
            st.rerun()

    if st.button(
        "시연 기록 전체 초기화",
        width="stretch",
        key="reset_demo",
    ):
        reset_demo()


# -----------------------------------------------------------------------------
# 실행
# -----------------------------------------------------------------------------
def main() -> None:
    inject_css()
    init_state()

    places = get_places()

    render_header()

    page = st.session_state[
        NAV_KEY
    ]

    if page == HOME:
        page_home(places)

    elif page == COURSE:
        page_course(places)

    elif page == CHECKIN:
        page_checkin()

    else:
        page_my(places)

    render_bottom_nav()


if __name__ == "__main__":
    main()