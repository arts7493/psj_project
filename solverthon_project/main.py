from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from hashlib import sha1, sha256
from html import escape
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# 앱 설정
# -----------------------------------------------------------------------------
APP_NAME = "WAITGO"
APP_SUBTITLE = "광주·전남 대기여행"
BUILD = "V3.3 MOBILE"

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "places.csv"
REQUIRED_COLUMNS = ("카테고리", "이름", "주소")

HOME = "🏠 홈"
COURSE = "🧭 코스"
CHECKIN = "📷 체크인"
MY = "👤 MY"

NAV_ITEMS = (HOME, COURSE, CHECKIN, MY)
NAV_KEY = "bottom_nav"

CATEGORY_ICON = {
    "맛집": "🍽️",
    "관광명소": "🌿",
    "문화공간": "🎨",
    "카페": "☕",
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

ROUTE_TYPES = (
    (
        "가볍게 한바퀴",
        "산책형",
        "짧고 부담 없이 둘러보는 코스",
        ("관광명소", "카페", "문화공간"),
    ),
    (
        "내 취향 중심",
        "맞춤형",
        "선택한 관심사를 먼저 반영한 코스",
        ("카페", "문화공간", "관광명소"),
    ),
    (
        "실내·휴식 균형",
        "안심형",
        "문화공간과 카페를 중심으로 쉬어가는 코스",
        ("문화공간", "카페", "관광명소"),
    ),
)


# Streamlit 명령 중 가장 먼저 실행
st.set_page_config(
    page_title=f"{APP_NAME} · {BUILD}",
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------------
# 공통 함수
# -----------------------------------------------------------------------------
def h(value: Any) -> str:
    """HTML에 넣을 문자열을 안전하게 변환한다."""
    return escape(str(value or "").strip())


def stable_number(text: str, modulo: int) -> int:
    """같은 입력은 항상 같은 숫자가 나오도록 해시를 만든다."""
    digest = sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:14], 16) % modulo


def optional_text(
    row: dict[str, Any] | pd.Series,
    *columns: str,
) -> str:
    """
    CSV에 선택 컬럼이 추가될 경우 활용한다.

    예:
    체류시간, 설명, 대표메뉴, 태그
    """
    for column in columns:
        value = row.get(column, "")

        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


def parse_minutes(value: Any, default: int) -> int:
    """
    '30분', '약 40분', '35' 같은 값을 분 단위 숫자로 변환한다.
    """
    match = re.search(r"\d+", str(value or ""))

    if not match:
        return default

    return max(10, min(180, int(match.group())))


def format_minutes(minutes: int) -> str:
    """분을 1시간 30분 형식으로 표시한다."""
    hours, remain = divmod(max(0, int(minutes)), 60)

    if hours and remain:
        return f"{hours}시간 {remain}분"

    if hours:
        return f"{hours}시간"

    return f"{remain}분"


def extract_region(address: str) -> str:
    """
    주소에서 광주 구 또는 전남 시·군을 추출한다.

    CSV에 '지역' 컬럼이 추가되면 CSV의 지역값을 우선 사용한다.
    """
    address = re.sub(r"\s+", " ", str(address).strip())
    tokens = address.split(" ")

    if address.startswith("광주광역시"):
        district = next(
            (
                token
                for token in tokens[1:]
                if token.endswith(("구", "군"))
            ),
            "",
        )

        return f"광주 {district}".strip() if district else "광주"

    # 현재 CSV에 들어 있는 광주 주소 표기도 대응
    if address.startswith(("전남광주", "광주 ")):
        district = next(
            (
                token
                for token in tokens
                if token.endswith(("구", "군"))
            ),
            "",
        )

        return f"광주 {district}".strip() if district else "광주"

    if address.startswith(("전라남도", "전남 ")):
        city = next(
            (
                token
                for token in tokens[1:]
                if token.endswith(("시", "군"))
            ),
            "",
        )

        return city or "전남"

    return "지역 확인 필요"


def reason_for(
    category: str,
    companion: str,
    custom: str = "",
) -> str:
    """CSV 설명이 없을 때 기본 추천 이유를 만든다."""
    if custom:
        return custom

    prefix = {
        "혼자": "혼자서도 부담 없이",
        "연인": "함께 분위기와 사진을 즐기며",
        "가족": "가족과 편안하게",
        "친구": "친구와 가볍게",
    }.get(companion, "편안하게")

    reasons = {
        "관광명소": f"{prefix} 둘러보기 좋은 지역 명소",
        "문화공간": f"{prefix} 머물기 좋은 문화 공간",
        "카페": f"{prefix} 쉬어가기 좋은 카페",
        "맛집": "대기 종료 후 이어지는 선택 맛집",
    }

    return reasons.get(
        category,
        f"{prefix} 방문하기 좋은 장소",
    )


def go(page: str) -> None:
    """하단 메뉴를 변경하고 화면을 즉시 다시 실행한다."""
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

    modified_time을 인자로 받아 CSV 수정 후 자동으로 캐시가 갱신되게 한다.
    """
    del modified_time

    last_error: Exception | None = None

    for encoding in ("utf-8-sig", "utf-8", "cp949"):
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
            "CSV를 UTF-8 또는 CP949 형식으로 저장해 주세요."
        ) from last_error

    df.columns = [
        str(column).replace("\ufeff", "").strip()
        for column in df.columns
    ]

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"필수 컬럼이 없습니다: {', '.join(missing)}"
        )

    for column in df.columns:
        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    df = df[
        (df["이름"] != "")
        & (df["주소"] != "")
    ].copy()

    df = df.drop_duplicates().copy()

    derived_region = df["주소"].map(extract_region)

    if "지역" in df.columns:
        df["지역"] = df["지역"].where(
            df["지역"] != "",
            derived_region,
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

    df = (
        df.sort_values(
            ["지역", "_order", "이름"],
            kind="stable",
        )
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    return df


def get_places() -> pd.DataFrame:
    """CSV 파일 존재 여부와 로딩 오류를 화면에 표시한다."""
    if not DATA_PATH.exists():
        st.error("`data/places.csv`를 찾지 못했습니다.")

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
        st.error("CSV를 읽는 중 오류가 발생했습니다.")
        st.exception(exc)
        st.stop()


# -----------------------------------------------------------------------------
# 세션 상태
# -----------------------------------------------------------------------------
def init_state() -> None:
    """페이지 이동, 포인트, 추천 결과 등을 세션에 보관한다."""
    defaults = {
        NAV_KEY: HOME,
        "points": 0,
        "preferences": {},
        "selected_restaurant": "",
        "routes": [],
        "route_index": 0,
        "route_nonce": 0,
        "queue": None,
        "saved": [],
        "checkins": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = deepcopy(value)

    if st.session_state[NAV_KEY] not in NAV_ITEMS:
        st.session_state[NAV_KEY] = HOME


def reset_demo() -> None:
    """CSV는 그대로 두고 시연 기록만 초기화한다."""
    st.session_state.points = 0
    st.session_state.preferences = {}
    st.session_state.selected_restaurant = ""
    st.session_state.routes = []
    st.session_state.route_index = 0
    st.session_state.route_nonce = 0
    st.session_state.queue = None
    st.session_state.saved = []
    st.session_state.checkins = []
    st.session_state[NAV_KEY] = HOME


# -----------------------------------------------------------------------------
# 추천 엔진
# -----------------------------------------------------------------------------
def route_id(route: dict[str, Any]) -> str:
    """저장 코스 중복 확인용 ID를 만든다."""
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


def build_routes(
    places: pd.DataFrame,
    restaurant_name: str,
    preferences: dict[str, Any],
    nonce: int = 0,
) -> list[dict[str, Any]]:
    """
    같은 지역 안에서 세 가지 추천 코스를 만든다.

    현재는 실제 위도·경도가 없으므로 시·군 기준으로 추천한다.
    """
    region = preferences["region"]
    wait = int(preferences["wait"])
    companion = preferences["companion"]
    interests = preferences["interests"]

    local_places = places[
        places["지역"] == region
    ].copy()

    restaurant_rows = local_places[
        (local_places["카테고리"] == "맛집")
        & (local_places["이름"] == restaurant_name)
    ]

    candidates = local_places[
        local_places["카테고리"] != "맛집"
    ].to_dict("records")

    if restaurant_rows.empty or not candidates:
        return []

    restaurant = restaurant_rows.iloc[0].to_dict()

    if wait <= 45:
        pre_count = 1
    elif wait <= 75:
        pre_count = 2
    else:
        pre_count = 3

    interest_priority = [
        INTEREST_CATEGORY[item]
        for item in interests
        if item in INTEREST_CATEGORY
    ]

    results: list[dict[str, Any]] = []

    for variant_index, route_type in enumerate(ROUTE_TYPES):
        title, badge, description, base_priority = route_type

        priority = list(
            dict.fromkeys(
                (
                    interest_priority
                    if variant_index == 1
                    else []
                )
                + list(base_priority)
            )
        )

        seed = (
            f"{region}|"
            f"{restaurant_name}|"
            f"{wait}|"
            f"{nonce}|"
            f"{variant_index}"
        )

        def sort_key(
            item: dict[str, Any],
        ) -> tuple[int, int]:
            category = str(
                item.get("카테고리", "")
            )

            if category in priority:
                rank = priority.index(category)
            else:
                rank = len(priority)

            item_seed = (
                f"{seed}|"
                f"{item.get('이름')}|"
                f"{item.get('주소')}"
            )

            return (
                rank,
                stable_number(item_seed, 10**9),
            )

        ordered = sorted(
            candidates,
            key=sort_key,
        )

        pre_places = ordered[
            : min(pre_count, len(ordered))
        ]

        post_place = next(
            (
                place
                for category in (
                    "카페",
                    "문화공간",
                    "관광명소",
                )
                for place in ordered
                if place not in pre_places
                and place.get("카테고리") == category
            ),
            None,
        )

        usable_wait = max(20, wait - 10)

        per_stop = max(
            15,
            min(
                35,
                usable_wait
                // max(1, len(pre_places)),
            ),
        )

        stops: list[dict[str, Any]] = []
        elapsed = 0

        # 식사 전 대기시간 활용 장소
        for index, place in enumerate(pre_places):
            category = place.get(
                "카테고리",
                "장소",
            )

            duration = min(
                parse_minutes(
                    optional_text(
                        place,
                        "체류시간",
                        "예상체류시간",
                        "소요시간",
                    ),
                    per_stop,
                ),
                per_stop,
            )

            stops.append(
                {
                    "phase": "대기 활용",
                    "time": (
                        "지금 출발"
                        if index == 0
                        else f"+{elapsed}분"
                    ),
                    "category": category,
                    "name": place.get("이름", ""),
                    "address": place.get("주소", ""),
                    "duration": duration,
                    "reason": reason_for(
                        category,
                        companion,
                        optional_text(
                            place,
                            "설명",
                            "한줄소개",
                            "추천이유",
                        ),
                    ),
                    "tags": optional_text(
                        place,
                        "태그",
                        "키워드",
                    ),
                }
            )

            elapsed += duration

        # 선택 맛집
        stops.append(
            {
                "phase": "식사",
                "time": f"약 +{wait}분",
                "category": "맛집",
                "name": restaurant.get("이름", ""),
                "address": restaurant.get("주소", ""),
                "duration": parse_minutes(
                    optional_text(
                        restaurant,
                        "체류시간",
                        "예상체류시간",
                    ),
                    60,
                ),
                "reason": f"예상 대기 {wait}분 후 식사",
                "tags": optional_text(
                    restaurant,
                    "대표메뉴",
                    "태그",
                ),
            }
        )

        # 식사 후 장소
        if post_place:
            category = post_place.get(
                "카테고리",
                "장소",
            )

            stops.append(
                {
                    "phase": "식후",
                    "time": "식사 후",
                    "category": category,
                    "name": post_place.get("이름", ""),
                    "address": post_place.get("주소", ""),
                    "duration": parse_minutes(
                        optional_text(
                            post_place,
                            "체류시간",
                            "예상체류시간",
                        ),
                        DEFAULT_DURATION.get(
                            category,
                            30,
                        ),
                    ),
                    "reason": reason_for(
                        category,
                        companion,
                        optional_text(
                            post_place,
                            "설명",
                            "한줄소개",
                            "추천이유",
                        ),
                    ),
                    "tags": optional_text(
                        post_place,
                        "태그",
                        "키워드",
                    ),
                }
            )

        route = {
            "title": title,
            "badge": badge,
            "description": description,
            "region": region,
            "restaurant": restaurant_name,
            "wait": wait,
            "companion": companion,
            "meal": preferences["meal"],
            "interests": interests,
            "stops": stops,
            "local_stay": sum(
                stop["duration"]
                for stop in stops
                if stop["category"] != "맛집"
            ),
        }

        route["id"] = route_id(route)
        results.append(route)

    return results


# -----------------------------------------------------------------------------
# 대기표
# -----------------------------------------------------------------------------
def create_queue(
    region: str,
    restaurant: str,
    wait: int,
) -> dict[str, Any]:
    """시연용 대기표를 만든다."""
    now = datetime.now()

    region_code = re.sub(
        r"[^가-힣A-Za-z0-9]",
        "",
        region,
    )[:2] or "GJ"

    ticket_number = 100 + stable_number(
        f"{restaurant}|{now.isoformat()}",
        900,
    )

    return {
        "ticket": f"{region_code}-{ticket_number}",
        "restaurant": restaurant,
        "wait": wait,
        "started_at": now.isoformat(
            timespec="seconds"
        ),
        "demo_elapsed": 0,
    }


def queue_status() -> tuple[int, int, int]:
    """
    남은 시간, 진행률, 앞 팀 수를 반환한다.

    실제 경과시간과 시연용 경과시간을 함께 사용한다.
    """
    queue = st.session_state.queue

    if not queue:
        return 0, 0, 0

    try:
        started = datetime.fromisoformat(
            queue["started_at"]
        )

        real_elapsed = max(
            0,
            int(
                (
                    datetime.now() - started
                ).total_seconds()
                // 60
            ),
        )

    except (ValueError, TypeError, KeyError):
        real_elapsed = 0

    wait = max(
        1,
        int(queue["wait"]),
    )

    elapsed = min(
        wait,
        real_elapsed
        + int(queue.get("demo_elapsed", 0)),
    )

    remaining = max(
        0,
        wait - elapsed,
    )

    progress = int(
        elapsed / wait * 100
    )

    teams = (
        0
        if remaining == 0
        else max(1, (remaining + 4) // 5)
    )

    return remaining, progress, teams


def active_route() -> dict[str, Any] | None:
    """현재 선택한 추천 코스를 반환한다."""
    if not st.session_state.routes:
        return None

    index = max(
        0,
        min(
            st.session_state.route_index,
            len(st.session_state.routes) - 1,
        ),
    )

    return st.session_state.routes[index]


def save_active_route() -> None:
    """현재 코스를 MY 일정에 저장한다."""
    route = active_route()

    if not route:
        return

    saved_ids = {
        item["id"]
        for item in st.session_state.saved
    }

    if route["id"] in saved_ids:
        st.toast(
            "이미 저장한 코스입니다.",
            icon="ℹ️",
        )
        return

    saved = deepcopy(route)

    saved["saved_at"] = datetime.now().strftime(
        "%m.%d %H:%M"
    )

    st.session_state.saved.insert(
        0,
        saved,
    )

    st.toast(
        "MY 일정에 저장했습니다.",
        icon="✅",
    )


# -----------------------------------------------------------------------------
# 모바일 디자인 CSS
# -----------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --green: #0b7c6b;
            --green-dark: #07584d;
            --green-soft: #e7f5f1;
            --orange: #ff7757;
            --ink: #17221f;
            --muted: #6b7974;
            --line: #e0eae6;
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
                    circle at 10% 0%,
                    rgba(11, 124, 107, 0.15),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 95% 18%,
                    rgba(255, 119, 87, 0.11),
                    transparent 25%
                ),
                #eaf0ed;
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
            max-width: 460px !important;
            min-height: 100vh;
            padding: 1rem 1rem 8rem !important;
            background: var(--paper);
            box-shadow:
                0 0 48px rgba(31, 62, 54, 0.13);
        }

        h1,
        h2,
        h3,
        p {
            color: var(--ink);
        }

        p {
            line-height: 1.55;
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.7rem;
            margin: 0.05rem 0 1rem;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }

        .logo {
            width: 2.65rem;
            height: 2.65rem;
            display: grid;
            place-items: center;
            border-radius: 0.95rem;
            color: white;
            font-weight: 950;
            background:
                linear-gradient(
                    145deg,
                    var(--green-dark),
                    #12a58b
                );
            box-shadow:
                0 9px 18px rgba(11, 124, 107, 0.24);
        }

        .brand-name {
            font-size: 1rem;
            font-weight: 950;
            line-height: 1.08;
        }

        .build {
            margin-top: 0.2rem;
            color: var(--green);
            font-size: 0.62rem;
            font-weight: 900;
            letter-spacing: 0.06em;
        }

        .points {
            padding: 0.52rem 0.72rem;
            border: 1px solid #d8e6e1;
            border-radius: 999px;
            background: white;
            color: var(--green-dark);
            font-size: 0.8rem;
            font-weight: 900;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 1.45rem 1.3rem 1.35rem;
            margin-bottom: 1rem;
            border-radius: 1.65rem;
            color: white;
            background:
                linear-gradient(
                    145deg,
                    var(--green-dark),
                    var(--green) 58%,
                    #18aa8d
                );
            box-shadow:
                0 18px 36px rgba(7, 88, 77, 0.23);
        }

        .hero::before {
            content: "";
            position: absolute;
            width: 10rem;
            height: 10rem;
            right: -4.6rem;
            top: -4.8rem;
            border-radius: 50%;
            background:
                rgba(255, 255, 255, 0.13);
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 6rem;
            height: 6rem;
            right: 1.1rem;
            bottom: -4.2rem;
            border-radius: 50%;
            background:
                rgba(255, 207, 90, 0.27);
        }

        .hero-chip,
        .hero-title,
        .hero-copy {
            position: relative;
            z-index: 1;
        }

        .hero-chip {
            display: inline-flex;
            padding: 0.34rem 0.62rem;
            border-radius: 999px;
            background:
                rgba(255, 255, 255, 0.15);
            font-size: 0.69rem;
            font-weight: 850;
        }

        .hero-title {
            max-width: 18rem;
            margin: 0.8rem 0 0.48rem;
            color: white;
            font-size: 1.72rem;
            line-height: 1.2;
            font-weight: 950;
            letter-spacing: -0.04em;
        }

        .hero-copy {
            max-width: 20rem;
            margin: 0;
            color:
                rgba(255, 255, 255, 0.86);
            font-size: 0.84rem;
        }

        .stats {
            display: grid;
            grid-template-columns:
                repeat(3, 1fr);
            gap: 0.52rem;
            margin: 0.8rem 0 1.2rem;
        }

        .stat {
            padding: 0.76rem 0.45rem;
            border: 1px solid var(--line);
            border-radius: 1rem;
            background: white;
            text-align: center;
            box-shadow:
                0 7px 17px rgba(36, 65, 57, 0.045);
        }

        .stat strong {
            display: block;
            font-size: 1.01rem;
        }

        .stat span {
            font-size: 0.65rem;
            color: var(--muted);
        }

        .section {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 0.6rem;
            margin: 1.22rem 0 0.7rem;
        }

        .eyebrow {
            color: var(--green);
            font-size: 0.64rem;
            font-weight: 900;
            letter-spacing: 0.08em;
        }

        .title {
            margin: 0.15rem 0 0;
            font-size: 1.13rem;
            font-weight: 950;
            letter-spacing: -0.025em;
        }

        .caption {
            color: var(--muted);
            font-size: 0.68rem;
            text-align: right;
        }

        .card,
        .queue,
        .empty,
        .mission,
        .saved {
            padding: 1rem;
            border: 1px solid var(--line);
            border-radius: 1.22rem;
            background: white;
            box-shadow:
                0 8px 20px rgba(36, 65, 57, 0.055);
        }

        .card,
        .saved {
            margin-bottom: 0.75rem;
        }

        .empty {
            padding: 1.5rem 1rem;
            text-align: center;
        }

        .empty-icon {
            font-size: 2rem;
        }

        .empty strong {
            display: block;
            margin: 0.4rem 0 0.25rem;
        }

        .empty span {
            color: var(--muted);
            font-size: 0.78rem;
        }

        .queue {
            background:
                linear-gradient(
                    145deg,
                    white,
                    #effaf7
                );
        }

        .queue-top {
            display: flex;
            justify-content: space-between;
            gap: 0.7rem;
        }

        .q-ticket {
            color: var(--green);
            font-size: 0.66rem;
            font-weight: 900;
        }

        .q-name {
            margin-top: 0.2rem;
            font-size: 1rem;
            font-weight: 950;
        }

        .q-right {
            text-align: right;
        }

        .q-right strong {
            display: block;
            color: var(--green-dark);
            font-size: 1.4rem;
            line-height: 1;
        }

        .q-right span,
        .q-bottom {
            color: var(--muted);
            font-size: 0.65rem;
        }

        .track {
            height: 0.58rem;
            margin-top: 0.85rem;
            overflow: hidden;
            border-radius: 999px;
            background: #dceae5;
        }

        .bar {
            height: 100%;
            border-radius: 999px;
            background:
                linear-gradient(
                    90deg,
                    var(--green),
                    #25b596
                );
        }

        .q-bottom {
            display: flex;
            justify-content: space-between;
            margin-top: 0.45rem;
        }

        .route {
            padding: 1.05rem;
            margin: 0.75rem 0;
            border-radius: 1.28rem;
            color: white;
            background:
                linear-gradient(
                    140deg,
                    #213a34,
                    var(--green)
                );
            box-shadow:
                0 12px 25px rgba(20, 64, 54, 0.18);
        }

        .badge {
            display: inline-flex;
            padding: 0.26rem 0.5rem;
            border-radius: 999px;
            background:
                rgba(255, 255, 255, 0.16);
            font-size: 0.64rem;
            font-weight: 850;
        }

        .route h3 {
            margin: 0.52rem 0 0.25rem;
            color: white;
            font-size: 1.18rem;
        }

        .route p {
            margin: 0;
            color:
                rgba(255, 255, 255, 0.8);
            font-size: 0.76rem;
        }

        .route-meta {
            display: grid;
            grid-template-columns:
                repeat(3, 1fr);
            gap: 0.4rem;
            margin-top: 0.78rem;
        }

        .route-meta div {
            padding: 0.48rem 0.3rem;
            border-radius: 0.76rem;
            background:
                rgba(255, 255, 255, 0.11);
            text-align: center;
        }

        .route-meta strong {
            display: block;
            color: white;
            font-size: 0.8rem;
        }

        .route-meta span {
            color:
                rgba(255, 255, 255, 0.7);
            font-size: 0.59rem;
        }

        .timeline {
            position: relative;
            margin: 0.15rem 0 0.8rem;
        }

        .timeline::before {
            content: "";
            position: absolute;
            left: 1.13rem;
            top: 1.2rem;
            bottom: 1.2rem;
            width: 2px;
            background: #dcebe6;
        }

        .stop {
            position: relative;
            display: grid;
            grid-template-columns:
                2.3rem 1fr;
            gap: 0.68rem;
            padding: 0.7rem 0;
        }

        .stop-icon {
            z-index: 1;
            width: 2.3rem;
            height: 2.3rem;
            display: grid;
            place-items: center;
            border: 1px solid #d4e7e0;
            border-radius: 0.82rem;
            background: white;
            box-shadow:
                0 5px 12px rgba(22, 73, 61, 0.08);
        }

        .stop-body {
            padding: 0.78rem 0.82rem;
            border: 1px solid var(--line);
            border-radius: 1.02rem;
            background: white;
        }

        .stop-top {
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
        }

        .stop-phase {
            color: var(--green);
            font-size: 0.62rem;
            font-weight: 900;
        }

        .stop-time {
            color: var(--muted);
            font-size: 0.62rem;
        }

        .stop-name {
            margin-top: 0.23rem;
            font-size: 0.91rem;
            font-weight: 950;
        }

        .stop-reason {
            margin-top: 0.25rem;
            color: #53625d;
            font-size: 0.71rem;
        }

        .stop-address {
            margin-top: 0.38rem;
            color: #7a8782;
            font-size: 0.64rem;
        }

        .stop-tags {
            margin-top: 0.36rem;
            color: var(--orange);
            font-size: 0.63rem;
            font-weight: 850;
        }

        .map {
            display: inline-block;
            margin-top: 0.4rem;
            color: var(--green-dark) !important;
            font-size: 0.65rem;
            font-weight: 850;
            text-decoration: none;
        }

        .mission {
            margin: 0.85rem 0;
            border-color: #f0dcaa;
            background:
                linear-gradient(
                    145deg,
                    #fffaf0,
                    white
                );
        }

        .mission small {
            color: #ae7000;
            font-weight: 900;
        }

        .mission strong {
            display: block;
            margin: 0.25rem 0;
            font-size: 0.92rem;
        }

        .mission span {
            color: var(--muted);
            font-size: 0.72rem;
        }

        .place-kind {
            color: var(--green);
            font-size: 0.66rem;
            font-weight: 900;
        }

        .place-name {
            margin-top: 0.25rem;
            font-size: 1rem;
            font-weight: 950;
        }

        .place-address {
            margin-top: 0.32rem;
            color: var(--muted);
            font-size: 0.7rem;
        }

        .done {
            display: inline-flex;
            margin-top: 0.5rem;
            padding: 0.28rem 0.52rem;
            border-radius: 999px;
            background: var(--green-soft);
            color: var(--green-dark);
            font-size: 0.64rem;
            font-weight: 900;
        }

        .saved-top {
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
        }

        .saved-title {
            font-size: 0.9rem;
            font-weight: 950;
        }

        .saved-time {
            color: var(--muted);
            font-size: 0.62rem;
        }

        .saved-copy {
            margin-top: 0.27rem;
            color: var(--muted);
            font-size: 0.7rem;
        }

        .history {
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
            padding: 0.68rem 0;
            border-bottom: 1px solid #edf2f0;
        }

        .history:last-child {
            border-bottom: 0;
        }

        .history-name {
            font-size: 0.76rem;
            font-weight: 850;
        }

        .history-time {
            color: var(--muted);
            font-size: 0.64rem;
        }

        .history-points {
            color: var(--green);
            font-size: 0.73rem;
            font-weight: 950;
        }

        div[data-baseweb="select"] > div,
        [data-testid="stTextInput"] input {
            min-height: 3rem;
            border-radius: 0.9rem !important;
            border-color: #dce7e3 !important;
            background: white !important;
        }

        [data-testid="stWidgetLabel"] p {
            color: #40504a;
            font-size: 0.75rem;
            font-weight: 850;
        }

        .stButton > button {
            width: 100%;
            min-height: 3rem;
            border-radius: 0.93rem;
            border: 1px solid #d6e4df;
            font-weight: 900;
        }

        .stButton > button[kind="primary"] {
            border: 0;
            color: white;
            background:
                linear-gradient(
                    135deg,
                    var(--green-dark),
                    #0e927a
                );
            box-shadow:
                0 10px 20px rgba(11, 124, 107, 0.18);
        }

        [data-testid="stCameraInput"] {
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 1.2rem;
            background: white;
        }

        /* 동행 유형과 식사 시간 */
        .st-key-home_companion [role="radiogroup"],
        .st-key-home_meal [role="radiogroup"] {
            display: grid !important;
            grid-template-columns:
                repeat(4, 1fr);
            gap: 0.34rem;
        }

        /* 추천 코스 3안 */
        .st-key-route_picker [role="radiogroup"] {
            display: grid !important;
            grid-template-columns:
                repeat(3, 1fr);
            gap: 0.34rem;
        }

        .st-key-home_companion label,
        .st-key-home_meal label,
        .st-key-route_picker label {
            justify-content: center;
            min-height: 2.5rem;
            padding: 0.3rem 0.15rem !important;
            border: 1px solid #dce7e3;
            border-radius: 0.8rem;
            background: white;
            text-align: center;
        }

        .st-key-home_companion label:has(input:checked),
        .st-key-home_meal label:has(input:checked),
        .st-key-route_picker label:has(input:checked) {
            border-color: #9fd5c8;
            background: var(--green-soft);
        }

        .st-key-home_companion label > div:first-child,
        .st-key-home_meal label > div:first-child,
        .st-key-route_picker label > div:first-child {
            display: none;
        }

        .st-key-home_companion p,
        .st-key-home_meal p,
        .st-key-route_picker p {
            font-size: 0.68rem;
            font-weight: 850;
        }

        /* 하단 고정 모바일 메뉴 */
        .st-key-bottom_nav {
            position: fixed;
            left: 50%;
            bottom: 0.65rem;
            transform: translateX(-50%);
            z-index: 9999;
            width:
                min(
                    430px,
                    calc(100vw - 1.2rem)
                );
            padding: 0.38rem;
            border:
                1px solid rgba(206, 222, 216, 0.95);
            border-radius: 1.22rem;
            background:
                rgba(255, 255, 255, 0.94);
            box-shadow:
                0 14px 35px rgba(29, 57, 49, 0.2);
            backdrop-filter: blur(16px);
        }

        .st-key-bottom_nav
        [data-testid="stWidgetLabel"] {
            display: none;
        }

        .st-key-bottom_nav [role="radiogroup"] {
            display: grid !important;
            grid-template-columns:
                repeat(4, 1fr);
            gap: 0.23rem;
        }

        .st-key-bottom_nav label {
            justify-content: center;
            min-height: 3rem;
            padding: 0.32rem 0.12rem !important;
            border-radius: 0.88rem;
            text-align: center;
        }

        .st-key-bottom_nav
        label:has(input:checked) {
            background: var(--green-soft);
        }

        .st-key-bottom_nav
        label > div:first-child {
            display: none;
        }

        .st-key-bottom_nav p {
            color: #66736f;
            font-size: 0.67rem;
            font-weight: 850;
        }

        .st-key-bottom_nav
        label:has(input:checked) p {
            color: var(--green-dark);
        }

        .note {
            color: var(--muted);
            font-size: 0.65rem;
            line-height: 1.5;
        }

        .divider {
            height: 1px;
            margin: 1rem 0;
            background: var(--line);
        }

        @media (max-width: 640px) {
            [data-testid="stMainBlockContainer"],
            .block-container {
                max-width: none !important;
                padding:
                    0.85rem 0.9rem 7.6rem !important;
                box-shadow: none;
            }

            .hero-title {
                font-size: 1.56rem;
            }

            .st-key-bottom_nav {
                bottom: 0.4rem;
                width: calc(100vw - 1rem);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 공통 UI
# -----------------------------------------------------------------------------
def render_header() -> None:
    st.markdown(
        f"""
        <div class="topbar">
            <div class="brand">
                <div class="logo">W</div>

                <div>
                    <div class="brand-name">
                        {APP_NAME}
                    </div>

                    <div class="build">
                        {APP_SUBTITLE} · {BUILD}
                    </div>
                </div>
            </div>

            <div class="points">
                🪙 {st.session_state.points:,}P
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(
    title: str,
    eyebrow: str = "",
    caption: str = "",
) -> None:
    st.markdown(
        f"""
        <div class="section">
            <div>
                <div class="eyebrow">
                    {h(eyebrow)}
                </div>

                <h3 class="title">
                    {h(title)}
                </h3>
            </div>

            <div class="caption">
                {h(caption)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty(
    icon: str,
    title: str,
    copy: str,
) -> None:
    st.markdown(
        f"""
        <div class="empty">
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
        """,
        unsafe_allow_html=True,
    )


def render_bottom_nav() -> None:
    st.radio(
        "하단 메뉴",
        NAV_ITEMS,
        key=NAV_KEY,
        horizontal=True,
        label_visibility="collapsed",
        width="stretch",
    )


# -----------------------------------------------------------------------------
# 홈 화면
# -----------------------------------------------------------------------------
def page_home(places: pd.DataFrame) -> None:
    st.markdown(
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
                대기시간과 취향을 고르면 주변 장소를 이어 만든
                코스를 바로 제안합니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="stats">
            <div class="stat">
                <strong>{len(places)}</strong>
                <span>등록 장소</span>
            </div>

            <div class="stat">
                <strong>{places["지역"].nunique()}</strong>
                <span>서비스 지역</span>
            </div>

            <div class="stat">
                <strong>{len(st.session_state.checkins)}</strong>
                <span>내 체크인</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section(
        "오늘의 대기 코스 만들기",
        "START",
        "CSV 기반 추천",
    )

    restaurants_df = places[
        places["카테고리"] == "맛집"
    ]

    regions = sorted(
        restaurants_df[
            "지역"
        ].dropna().unique().tolist()
    )

    if not regions:
        st.warning(
            "CSV에 `맛집` 카테고리가 없습니다."
        )
        return

    if (
        st.session_state.get("home_region")
        not in regions
    ):
        st.session_state.home_region = regions[0]

    region = st.selectbox(
        "지역",
        regions,
        key="home_region",
    )

    restaurant_names = restaurants_df[
        restaurants_df["지역"] == region
    ]["이름"].tolist()

    if not restaurant_names:
        st.warning(
            "선택 지역에 맛집 데이터가 없습니다."
        )
        return

    if (
        st.session_state.get("home_restaurant")
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

    st.radio(
        "동행 유형",
        ("혼자", "연인", "가족", "친구"),
        horizontal=True,
        key="home_companion",
    )

    st.radio(
        "식사 시간",
        ("점심", "오후", "저녁", "야간"),
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
        format_func=lambda value: f"{value}분",
        key="home_wait",
    )

    interests = st.multiselect(
        "관심 분야",
        tuple(INTEREST_CATEGORY),
        default=(
            "자연·산책",
            "카페·디저트",
        ),
        key="home_interests",
    )

    if st.button(
        "✨ 내 대기 코스 만들기",
        type="primary",
        width="stretch",
        key="make_route",
    ):
        preferences = {
            "region": region,
            "companion": (
                st.session_state.home_companion
            ),
            "meal": st.session_state.home_meal,
            "wait": int(wait),
            "interests": list(interests),
        }

        routes = build_routes(
            places,
            restaurant,
            preferences,
        )

        if not routes:
            st.warning(
                "이 지역에는 코스를 만들 주변 장소가 부족합니다."
            )

        else:
            st.session_state.preferences = (
                preferences
            )

            st.session_state.selected_restaurant = (
                restaurant
            )

            st.session_state.routes = routes
            st.session_state.route_index = 0
            st.session_state.route_nonce = 0

            st.session_state.queue = create_queue(
                region,
                restaurant,
                int(wait),
            )

            st.session_state.pop(
                "route_picker",
                None,
            )

            go(COURSE)

    st.markdown(
        """
        <div class="mission">
            <small>DEMO FLOW</small>

            <strong>
                코스 생성 → 일정 저장 → 카메라 체크인 → 100P 적립
            </strong>

            <span>
                지도 거리와 GPT 설명은 다음 단계에서 연결합니다.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.routes:
        route = active_route()

        if route:
            render_section(
                "최근 만든 코스",
                "RECENT",
            )

            st.markdown(
                f"""
                <div class="card">
                    <div class="place-kind">
                        {h(route["region"])}
                        ·
                        {h(route["badge"])}
                    </div>

                    <div class="place-name">
                        {h(route["title"])}
                    </div>

                    <div class="place-address">
                        {h(route["restaurant"])}
                        중심 ·
                        {len(route["stops"])}개 장소
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button(
                "최근 코스 다시 보기",
                width="stretch",
                key="open_recent",
            ):
                go(COURSE)


# -----------------------------------------------------------------------------
# 대기 현황 카드
# -----------------------------------------------------------------------------
def render_queue_card() -> None:
    if not st.session_state.queue:
        return

    remaining, progress, teams = (
        queue_status()
    )

    if remaining == 0:
        status = "입장 가능"
        teams_text = "지금 입장해 주세요"
    else:
        status = f"{remaining}분 남음"
        teams_text = f"앞에 약 {teams}팀"

    queue = st.session_state.queue

    st.markdown(
        f"""
        <div class="queue">
            <div class="queue-top">
                <div>
                    <div class="q-ticket">
                        대기번호 {h(queue["ticket"])}
                    </div>

                    <div class="q-name">
                        {h(queue["restaurant"])}
                    </div>
                </div>

                <div class="q-right">
                    <strong>
                        {h(status)}
                    </strong>

                    <span>
                        {h(teams_text)}
                    </span>
                </div>
            </div>

            <div class="track">
                <div
                    class="bar"
                    style="width: {progress}%"
                ></div>
            </div>

            <div class="q-bottom">
                <span>
                    대기 진행률 {progress}%
                </span>

                <span>
                    시연용 예상 정보
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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


# -----------------------------------------------------------------------------
# 코스 장소 카드
# -----------------------------------------------------------------------------
def render_route_stop(
    stop: dict[str, Any],
) -> None:
    category = stop["category"]

    icon = CATEGORY_ICON.get(
        category,
        "📍",
    )

    map_query = quote_plus(
        f"{stop['name']} {stop['address']}"
    )

    tags = str(
        stop.get("tags", "")
    ).strip()

    if tags:
        tags_html = (
            '<div class="stop-tags">'
            f'#{h(tags).replace(" ", " #")}'
            "</div>"
        )
    else:
        tags_html = ""

    st.markdown(
        f"""
        <div class="stop">
            <div class="stop-icon">
                {icon}
            </div>

            <div class="stop-body">
                <div class="stop-top">
                    <div class="stop-phase">
                        {h(stop["phase"])}
                        ·
                        {h(category)}
                    </div>

                    <div class="stop-time">
                        {h(stop["time"])}
                        ·
                        {stop["duration"]}분
                    </div>
                </div>

                <div class="stop-name">
                    {h(stop["name"])}
                </div>

                <div class="stop-reason">
                    {h(stop["reason"])}
                </div>

                <div class="stop-address">
                    {h(stop["address"])}
                </div>

                {tags_html}

                <a
                    class="map"
                    href="https://www.google.com/maps/search/?api=1&query={map_query}"
                    target="_blank"
                >
                    지도에서 검색 ↗
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 코스 화면
# -----------------------------------------------------------------------------
def page_course(
    places: pd.DataFrame,
) -> None:
    if not st.session_state.routes:
        render_empty(
            "🧭",
            "아직 만든 코스가 없습니다",
            "홈에서 맛집과 대기시간을 선택해 주세요.",
        )

        if st.button(
            "홈에서 코스 만들기",
            type="primary",
            width="stretch",
            key="course_home",
        ):
            go(HOME)

        return

    render_section(
        "현재 대기 현황",
        "QUEUE",
        "시연 조작 가능",
    )

    render_queue_card()

    render_section(
        "추천 코스 3안",
        "ROUTE",
        "한 가지를 선택하세요",
    )

    titles = [
        route["title"]
        for route in st.session_state.routes
    ]

    if (
        st.session_state.get("route_picker")
        not in titles
    ):
        st.session_state.route_picker = (
            titles[0]
        )

    selected = st.radio(
        "추천안",
        titles,
        horizontal=True,
        label_visibility="collapsed",
        key="route_picker",
        width="stretch",
    )

    st.session_state.route_index = (
        titles.index(selected)
    )

    route = active_route()

    if route is None:
        st.warning(
            "추천 코스를 불러오지 못했습니다."
        )
        return

    st.markdown(
        f"""
        <div class="route">
            <div class="badge">
                {h(route["badge"])}
                · 지역 기준 추천
            </div>

            <h3>
                {h(route["title"])}
            </h3>

            <p>
                {h(route["description"])}
            </p>

            <div class="route-meta">
                <div>
                    <strong>
                        {len(route["stops"])}곳
                    </strong>

                    <span>
                        전체 방문지
                    </span>
                </div>

                <div>
                    <strong>
                        {format_minutes(route["local_stay"])}
                    </strong>

                    <span>
                        지역 체류
                    </span>
                </div>

                <div>
                    <strong>
                        {h(route["companion"])}
                    </strong>

                    <span>
                        동행 유형
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="timeline">',
        unsafe_allow_html=True,
    )

    for stop in route["stops"]:
        render_route_stop(stop)

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="mission">
            <small>WAITGO MISSION</small>

            <strong>
                추천 장소에서 사진 체크인하면 +100P
            </strong>

            <span>
                노트북 기본 카메라로 촬영 후 체크인을 확정합니다.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    saved_ids = {
        item["id"]
        for item in st.session_state.saved
    }

    already_saved = (
        route["id"] in saved_ids
    )

    left, right = st.columns(2)

    with left:
        if st.button(
            (
                "✓ 저장됨"
                if already_saved
                else "＋ MY에 담기"
            ),
            disabled=already_saved,
            width="stretch",
            key="save_route",
        ):
            save_active_route()
            st.rerun()

    with right:
        if st.button(
            "📷 체크인",
            type="primary",
            width="stretch",
            key="to_checkin",
        ):
            go(CHECKIN)

    if st.button(
        "🔄 같은 조건으로 다시 추천",
        width="stretch",
        key="regenerate",
    ):
        st.session_state.route_nonce += 1

        st.session_state.routes = (
            build_routes(
                places,
                st.session_state.selected_restaurant,
                st.session_state.preferences,
                st.session_state.route_nonce,
            )
        )

        st.session_state.route_index = 0

        st.session_state.pop(
            "route_picker",
            None,
        )

        st.toast(
            "새로운 조합을 만들었습니다.",
            icon="🔄",
        )

        st.rerun()


# -----------------------------------------------------------------------------
# 체크인 화면
# -----------------------------------------------------------------------------
def page_checkin() -> None:
    route = active_route()

    if route is None and st.session_state.saved:
        route = st.session_state.saved[0]

    if route is None:
        render_empty(
            "📷",
            "체크인할 코스가 없습니다",
            "먼저 홈에서 코스를 만들어 주세요.",
        )

        if st.button(
            "홈으로 이동",
            type="primary",
            width="stretch",
            key="checkin_home",
        ):
            go(HOME)

        return

    render_section(
        "현장 사진 체크인",
        "CHECK-IN",
        "+100P · 장소별 1회",
    )

    stops = route["stops"]

    names = [
        stop["name"]
        for stop in stops
    ]

    if (
        st.session_state.get("checkin_place")
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
        for item in stops
        if item["name"] == selected_name
    )

    checkin_id = sha1(
        (
            f"{stop['name']}|"
            f"{stop['address']}"
        ).encode("utf-8")
    ).hexdigest()[:14]

    already_checked = any(
        item["id"] == checkin_id
        for item in st.session_state.checkins
    )

    if already_checked:
        done_html = """
        <div class="done">
            ✓ 체크인 완료 · 포인트 지급 완료
        </div>
        """
    else:
        done_html = ""

    st.markdown(
        f"""
        <div class="card">
            <div class="place-kind">
                {CATEGORY_ICON.get(stop["category"], "📍")}
                {h(stop["category"])}
            </div>

            <div class="place-name">
                {h(stop["name"])}
            </div>

            <div class="place-address">
                {h(stop["address"])}
            </div>

            {done_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "브라우저에서 카메라 권한을 허용한 뒤 촬영해 주세요."
    )

    picture = st.camera_input(
        "체크인 사진",
        key=f"camera_{checkin_id}",
        resolution="720p",
        width="stretch",
    )

    if picture is not None:
        if st.button(
            (
                "이미 체크인한 장소입니다"
                if already_checked
                else "체크인 확정 +100P"
            ),
            type="primary",
            disabled=already_checked,
            width="stretch",
            key=f"confirm_{checkin_id}",
        ):
            record = {
                "id": checkin_id,
                "name": stop["name"],
                "category": stop["category"],
                "checked_at": (
                    datetime.now().strftime(
                        "%m.%d %H:%M"
                    )
                ),
                "points": 100,
                "image_hash": sha256(
                    picture.getvalue()
                ).hexdigest()[:12],
            }

            st.session_state.checkins.insert(
                0,
                record,
            )

            st.session_state.points += 100

            st.toast(
                "체크인 성공! 100P를 적립했습니다.",
                icon="🎉",
            )

            st.rerun()

    st.markdown(
        """
        <div class="note">
            사진 원본은 파일이나 CSV에 저장하지 않고
            현재 세션의 체크인 확인에만 사용합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section(
        "오늘의 미션",
        "PROGRESS",
    )

    completed = len(
        st.session_state.checkins
    )

    percent = min(
        100,
        int(completed / 3 * 100),
    )

    st.markdown(
        f"""
        <div class="card">
            <div class="place-kind">
                지역 체류 미션
            </div>

            <div class="place-name">
                {completed}/3곳 체크인 완료
            </div>

            <div class="track">
                <div
                    class="bar"
                    style="width: {percent}%"
                ></div>
            </div>

            <div class="place-address">
                3곳 방문 시 총 300P
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
        "현재 세션 기록",
    )

    local_stay = sum(
        route.get("local_stay", 0)
        for route in st.session_state.saved
    )

    st.markdown(
        f"""
        <div class="stats">
            <div class="stat">
                <strong>
                    {st.session_state.points:,}P
                </strong>

                <span>
                    적립 포인트
                </span>
            </div>

            <div class="stat">
                <strong>
                    {len(st.session_state.saved)}
                </strong>

                <span>
                    저장 코스
                </span>
            </div>

            <div class="stat">
                <strong>
                    {format_minutes(local_stay)}
                </strong>

                <span>
                    지역 체류
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section(
        "저장한 코스",
        "SAVED",
        f"{len(st.session_state.saved)}개",
    )

    if not st.session_state.saved:
        render_empty(
            "🗂️",
            "저장한 코스가 없습니다",
            "추천 코스에서 MY에 담기를 눌러보세요.",
        )

    else:
        saved_copy = list(
            st.session_state.saved
        )

        for index, route in enumerate(saved_copy):
            st.markdown(
                f"""
                <div class="saved">
                    <div class="saved-top">
                        <div>
                            <div class="place-kind">
                                {h(route["region"])}
                                ·
                                {h(route["badge"])}
                            </div>

                            <div class="saved-title">
                                {h(route["title"])}
                            </div>
                        </div>

                        <div class="saved-time">
                            {h(route.get("saved_at"))}
                        </div>
                    </div>

                    <div class="saved-copy">
                        {h(route["restaurant"])}
                        중심 ·
                        {len(route["stops"])}개 장소 ·
                        {format_minutes(route["local_stay"])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            left, right = st.columns(
                [2, 1]
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

                    st.session_state.selected_restaurant = (
                        route["restaurant"]
                    )

                    st.session_state.preferences = {
                        "region": route["region"],
                        "companion": route["companion"],
                        "meal": route.get(
                            "meal",
                            "점심",
                        ),
                        "wait": int(
                            route.get("wait", 60)
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

                    st.session_state.pop(
                        "route_picker",
                        None,
                    )

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
                        for item in st.session_state.saved
                        if item["id"] != route["id"]
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
            "체크인 기록이 없습니다",
            "추천 장소에서 카메라 체크인을 해보세요.",
        )

    else:
        history_rows = ""

        for record in st.session_state.checkins:
            history_rows += f"""
            <div class="history">
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

                <div class="history-points">
                    +{record["points"]}P
                </div>
            </div>
            """

        st.markdown(
            f"""
            <div class="card">
                {history_rows}
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_section(
        "앱·데이터 확인",
        "SYSTEM",
    )

    with st.expander(
        "현재 실행 중인 main.py 확인"
    ):
        st.write(
            f"**빌드:** `{BUILD}`"
        )

        st.write(
            f"**실행 파일:** `{Path(__file__).resolve()}`"
        )

        st.write(
            f"**CSV:** `{DATA_PATH.resolve()}`"
        )

        st.write(
            f"**장소 수:** `{len(places)}개`"
        )

        visible_columns = [
            column
            for column in (
                "지역",
                "카테고리",
                "이름",
                "주소",
            )
            if column in places.columns
        ]

        st.dataframe(
            places[visible_columns],
            width="stretch",
            hide_index=True,
            height=230,
        )

        if st.button(
            "CSV 캐시 비우고 다시 읽기",
            width="stretch",
            key="reload_csv",
        ):
            st.cache_data.clear()
            st.rerun()

    st.markdown(
        '<div class="divider"></div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "시연 기록 전체 초기화",
        width="stretch",
        key="reset_demo",
    ):
        reset_demo()
        st.rerun()


# -----------------------------------------------------------------------------
# 앱 실행
# -----------------------------------------------------------------------------
def main() -> None:
    inject_css()
    init_state()

    places = get_places()

    render_header()

    current_page = st.session_state[
        NAV_KEY
    ]

    if current_page == HOME:
        page_home(places)

    elif current_page == COURSE:
        page_course(places)

    elif current_page == CHECKIN:
        page_checkin()

    else:
        page_my(places)

    render_bottom_nav()


if __name__ == "__main__":
    main()