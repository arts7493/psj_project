from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 이하
    import tomli as tomllib  # type: ignore[no-redef]


BASE_DIR = Path(__file__).resolve().parent
SECRETS_PATH = BASE_DIR / ".streamlit" / "secrets.toml"

DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_TIMEOUT_SECONDS = 45

# 실제 거리·이동시간·영업정보가 없는 상태에서
# AI가 만들어 내면 안 되는 표현
UNSUPPORTED_FACT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:km|킬로미터|m|미터)"
    r"|(?:도보|차량|자동차|택시|버스)\s*\d+\s*(?:분|시간)"
    r"|실시간\s*(?:혼잡|대기|영업)"
    r"|현재\s*(?:영업|휴무)"
    r"|정확히\s*\d+\s*분",
    re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# 설정 읽기
# -----------------------------------------------------------------------------
def _read_local_secrets() -> dict[str, Any]:
    """
    로컬의 .streamlit/secrets.toml을 읽는다.
    """
    if not SECRETS_PATH.exists():
        return {}

    try:
        with SECRETS_PATH.open("rb") as file:
            return dict(tomllib.load(file))

    except Exception:
        return {}


def _read_streamlit_secrets() -> dict[str, Any]:
    """
    배포 환경의 Streamlit Secrets를 읽는다.
    """
    try:
        import streamlit as st

        result: dict[str, Any] = {}

        for key in (
            "GEMINI_API_KEY",
            "GEMINI_MODEL",
            "AI_TIMEOUT_SECONDS",
        ):
            try:
                if key in st.secrets:
                    result[key] = st.secrets[key]

            except Exception:
                pass

        return result

    except Exception:
        return {}


def _valid_key(value: Any) -> str:
    """
    실제 Gemini API 키인지 검사한다.
    """
    text = str(value or "").strip()

    placeholders = (
        "PUT_YOUR_",
        "YOUR_API_KEY",
        "REPLACE_ME",
        "실제_GEMINI",
    )

    if not text:
        return ""

    if any(
        word.upper() in text.upper()
        for word in placeholders
    ):
        return ""

    return text


def load_gemini_settings() -> dict[str, Any]:
    """
    Gemini 설정을 읽는다.

    우선순위:
    환경변수
    → 로컬 secrets.toml
    → Streamlit Cloud Secrets
    → 기본값
    """
    values = _read_local_secrets()

    # 로컬 파일에 없는 값만
    # Streamlit Cloud Secrets로 보완한다.
    for key, value in _read_streamlit_secrets().items():
        values.setdefault(key, value)

    # 환경변수가 가장 높은 우선순위다.
    for key in (
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "AI_TIMEOUT_SECONDS",
    ):
        if key in os.environ:
            values[key] = os.environ[key]

    try:
        timeout = int(
            float(
                values.get(
                    "AI_TIMEOUT_SECONDS",
                    DEFAULT_TIMEOUT_SECONDS,
                )
            )
        )

    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_SECONDS

    return {
        "api_key": _valid_key(
            values.get("GEMINI_API_KEY")
        ),
        "model": (
            str(
                values.get(
                    "GEMINI_MODEL",
                    DEFAULT_MODEL,
                )
            ).strip()
            or DEFAULT_MODEL
        ),
        "timeout_seconds": max(
            15,
            min(timeout, 90),
        ),
    }


def get_gemini_status() -> dict[str, Any]:
    """
    키 자체는 노출하지 않고
    설정 여부와 모델만 반환한다.
    """
    settings = load_gemini_settings()

    return {
        "configured": bool(
            settings["api_key"]
        ),
        "model": settings["model"],
        "timeout_seconds": (
            settings["timeout_seconds"]
        ),
    }


# -----------------------------------------------------------------------------
# 후보 데이터와 프롬프트
# -----------------------------------------------------------------------------
def _first_text(
    item: dict[str, Any],
    *keys: str,
) -> str:
    """
    여러 후보 컬럼 중 처음 발견되는 값을 반환한다.
    """
    for key in keys:
        value = item.get(key, "")

        if (
            value is not None
            and str(value).strip()
        ):
            return str(value).strip()

    return ""


def _prepare_candidates(
    candidates: Sequence[dict[str, Any]],
    restaurant: str,
) -> list[dict[str, str]]:
    """
    Gemini에 전달할 지역 후보 장소를 정리한다.
    """
    prepared: list[dict[str, str]] = []
    seen: set[str] = set()

    for item in candidates:
        name = _first_text(
            item,
            "이름",
            "name",
        )

        if (
            not name
            or name == restaurant
            or name in seen
        ):
            continue

        seen.add(name)

        prepared.append(
            {
                "name": name,
                "category": (
                    _first_text(
                        item,
                        "카테고리",
                        "category",
                    )
                    or "장소"
                ),
                "address": _first_text(
                    item,
                    "주소",
                    "address",
                ),
                "tags": _first_text(
                    item,
                    "태그",
                    "키워드",
                    "tags",
                ),
            }
        )

    # 요청 크기가 과도하게 커지지 않도록 제한한다.
    return prepared[:30]


def _requested_place_count(
    wait_minutes: int,
    candidate_count: int,
) -> int:
    """
    대기시간에 따라 코스에 넣을 장소 수를 정한다.

    30~45분:
    식사 전 1곳 + 식사 후 1곳

    60~75분:
    식사 전 2곳 + 식사 후 1곳

    90분 이상:
    식사 전 3곳 + 식사 후 1곳
    """
    if wait_minutes <= 45:
        count = 2

    elif wait_minutes <= 75:
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


def _route_schema(
    place_count: int,
    candidate_names: Sequence[str],
) -> dict[str, Any]:
    """
    Gemini 구조화 출력용 동적 JSON Schema.

    장소 이름을 enum으로 제한하므로
    CSV에 없는 장소명을 반환할 수 없다.
    """
    return {
        "type": "object",
        "properties": {
            "routes": {
                "type": "array",
                "description": (
                    "서로 분위기가 다른 "
                    "추천 코스 3개"
                ),
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": (
                                "15자 안팎의 자연스러운 "
                                "한국어 코스명"
                            ),
                        },
                        "summary": {
                            "type": "string",
                            "description": (
                                "동행과 관심사를 반영한 "
                                "한 문장 설명"
                            ),
                        },
                        "places": {
                            "type": "array",
                            "description": (
                                "후보에서 고른 방문 장소. "
                                "마지막 장소는 식사 후 방문지"
                            ),
                            "minItems": place_count,
                            "maxItems": place_count,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": (
                                            "후보 목록의 name과 "
                                            "완전히 같은 장소명"
                                        ),
                                        "enum": list(
                                            candidate_names
                                        ),
                                    },
                                    "reason": {
                                        "type": "string",
                                        "description": (
                                            "거리나 영업정보를 "
                                            "넣지 않은 짧은 추천 이유"
                                        ),
                                    },
                                },
                                "required": [
                                    "name",
                                    "reason",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": [
                        "title",
                        "summary",
                        "places",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": [
            "routes",
        ],
        "additionalProperties": False,
    }


def _build_prompt(
    *,
    region: str,
    restaurant: str,
    wait_minutes: int,
    companion: str,
    meal_time: str,
    interests: Sequence[str],
    candidates: Sequence[dict[str, str]],
    place_count: int,
    variation: int,
) -> str:
    """
    사용자 조건과 CSV 후보를 프롬프트로 만든다.
    """
    before_count = (
        max(1, place_count - 1)
        if place_count > 1
        else 1
    )

    interest_text = (
        ", ".join(interests)
        if interests
        else "특별히 없음"
    )

    candidate_json = json.dumps(
        list(candidates),
        ensure_ascii=False,
        indent=2,
    )

    return f"""
너는 광주·전남 맛집 대기시간을 지역 경험으로 바꾸는
모바일 서비스 'WAITGO'의 여행 코스 큐레이터다.

아래 사용자 조건과 후보 장소를 이용해
추천 코스 3개를 만들어라.

[사용자 조건]

- 지역: {region}
- 대기 중인 맛집: {restaurant}
- 예상 대기시간: {wait_minutes}분
- 동행 유형: {companion}
- 식사 시간대: {meal_time}
- 관심 분야: {interest_text}
- 재추천 번호: {variation}

[반드시 지킬 규칙]

1. 제공된 후보 장소만 사용한다.
2. 장소명은 후보의 name을 한 글자도 바꾸지 않고 그대로 쓴다.
3. 선택 맛집은 추천 places에 넣지 않는다.
   앱이 식사 단계로 별도 추가한다.
4. 각 코스에는 정확히 {place_count}개의 장소를 넣는다.
5. 각 코스의 앞 {before_count}개는 식사 전 대기시간 활용 장소이고,
   마지막 장소는 식사 후 방문지다.
6. 한 코스 안에서는 같은 장소를 중복하지 않는다.
7. 세 코스는 제목, 분위기, 장소 조합이 가능한 한 다르게 한다.
8. 좌표와 지도 API가 없으므로 거리,
   도보시간, 차량시간을 추측하지 않는다.
9. 영업시간, 휴무일, 실시간 혼잡도,
   실제 대기정보를 추측하지 않는다.
10. 제목, 요약, 추천 이유는 짧고 자연스러운 한국어로 작성한다.
11. 반환 형식은 지정된 JSON Schema를 정확히 따른다.

[사용 가능한 후보 장소]

{candidate_json}
""".strip()


# -----------------------------------------------------------------------------
# Gemini 응답 검증
# -----------------------------------------------------------------------------
def _normalize(text: str) -> str:
    """
    공백과 특수문자를 제거해 장소명을 비교한다.
    """
    return re.sub(
        r"[\W_]+",
        "",
        str(text),
        flags=re.UNICODE,
    ).casefold()


def _safe_text(
    value: Any,
    fallback: str,
    limit: int,
) -> str:
    """
    AI 문장을 정리하고,
    지원하지 않는 사실 표현이 있으면
    안전한 기본 문장으로 교체한다.
    """
    text = re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()

    if (
        not text
        or UNSUPPORTED_FACT.search(text)
    ):
        text = fallback

    return text[:limit].rstrip()


def _default_reason(
    category: str,
    companion: str,
) -> str:
    """
    AI 이유를 사용할 수 없을 때의 기본 문장.
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


def _resolve_candidate(
    requested_name: str,
    exact: dict[str, dict[str, str]],
    normalized: dict[str, dict[str, str]],
    candidates: Sequence[dict[str, str]],
) -> tuple[
    dict[str, str] | None,
    bool,
]:
    """
    정확한 장소명을 우선 사용한다.

    공백이나 괄호 차이만 있는 경우에는
    유일하게 일치하는 후보로 보정한다.
    """
    if requested_name in exact:
        return exact[requested_name], False

    normalized_name = _normalize(
        requested_name
    )

    if normalized_name in normalized:
        return (
            normalized[normalized_name],
            True,
        )

    partial_matches = [
        item
        for item in candidates
        if normalized_name
        and (
            normalized_name
            in _normalize(item["name"])
            or _normalize(item["name"])
            in normalized_name
        )
    ]

    if len(partial_matches) == 1:
        return partial_matches[0], True

    return None, False


def region_friendly_title(
    index: int,
) -> str:
    """
    AI 제목이 없을 때 사용할 코스명.
    """
    return (
        "여유로운 대기 산책",
        "취향을 담은 지역 발견",
        "식사 뒤까지 이어지는 하루",
    )[min(index, 2)]


def _validate_routes(
    payload: dict[str, Any],
    candidates: Sequence[dict[str, str]],
    place_count: int,
    companion: str,
) -> tuple[
    list[dict[str, Any]],
    int,
    list[str],
]:
    """
    Gemini 결과를 CSV 후보와 대조한다.

    후보 밖 장소나 중복 장소는 제거하고,
    부족한 부분만 CSV 후보로 보충한다.
    """
    raw_routes = payload.get("routes")

    if (
        not isinstance(raw_routes, list)
        or not raw_routes
    ):
        raise RuntimeError(
            "Gemini 응답에 routes가 없습니다."
        )

    exact = {
        item["name"]: item
        for item in candidates
    }

    normalized = {
        _normalize(item["name"]): item
        for item in candidates
    }

    validated: list[dict[str, Any]] = []
    repair_count = 0
    warnings: list[str] = []
    used_titles: set[str] = set()

    for route_index in range(3):
        raw_route = (
            raw_routes[route_index]
            if route_index < len(raw_routes)
            else {}
        )

        if not isinstance(
            raw_route,
            dict,
        ):
            raw_route = {}

        raw_places = raw_route.get(
            "places",
            [],
        )

        if not isinstance(
            raw_places,
            list,
        ):
            raw_places = []

        selected: list[dict[str, Any]] = []
        used_names: set[str] = set()

        for choice in raw_places:
            if not isinstance(
                choice,
                dict,
            ):
                continue

            requested_name = str(
                choice.get(
                    "name",
                    "",
                )
            ).strip()

            candidate, repaired_name = (
                _resolve_candidate(
                    requested_name,
                    exact,
                    normalized,
                    candidates,
                )
            )

            if (
                candidate is None
                or candidate["name"]
                in used_names
            ):
                repair_count += 1
                continue

            if repaired_name:
                repair_count += 1

            used_names.add(
                candidate["name"]
            )

            selected.append(
                {
                    **candidate,
                    "reason": _safe_text(
                        choice.get("reason"),
                        _default_reason(
                            candidate["category"],
                            companion,
                        ),
                        100,
                    ),
                }
            )

            if (
                len(selected)
                == place_count
            ):
                break

        # 후보 밖 장소나 중복 때문에 부족하면
        # CSV 후보로만 보충한다.
        if len(selected) < place_count:
            offset = (
                route_index
                * max(1, place_count)
            ) % max(
                1,
                len(candidates),
            )

            rotated = (
                list(candidates[offset:])
                + list(candidates[:offset])
            )

            for candidate in rotated:
                if (
                    candidate["name"]
                    in used_names
                ):
                    continue

                used_names.add(
                    candidate["name"]
                )

                selected.append(
                    {
                        **candidate,
                        "reason": _default_reason(
                            candidate["category"],
                            companion,
                        ),
                    }
                )

                repair_count += 1

                if (
                    len(selected)
                    == place_count
                ):
                    break

        if not selected:
            raise RuntimeError(
                "검증 후 사용할 수 있는 "
                "추천 장소가 없습니다."
            )

        title = _safe_text(
            raw_route.get("title"),
            region_friendly_title(
                route_index
            ),
            32,
        )

        if title in used_titles:
            title = (
                f"{title} "
                f"{route_index + 1}"
            )

        used_titles.add(title)

        validated.append(
            {
                "title": title,
                "summary": _safe_text(
                    raw_route.get("summary"),
                    (
                        "선택한 취향에 맞춰 구성한 "
                        "지역 코스예요."
                    ),
                    100,
                ),
                "places": selected,
            }
        )

    if repair_count:
        warnings.append(
            "AI 결과 중 "
            f"{repair_count}개 항목을 "
            "CSV 기준으로 안전하게 보정했습니다."
        )

    return (
        validated,
        repair_count,
        warnings,
    )


# -----------------------------------------------------------------------------
# 외부에서 호출할 공개 함수
# -----------------------------------------------------------------------------
def _failure(
    model: str,
    error: str,
    latency: float = 0.0,
) -> dict[str, Any]:
    return {
        "success": False,
        "routes": [],
        "model": model,
        "latency_seconds": round(
            latency,
            2,
        ),
        "interaction_id": "",
        "repair_count": 0,
        "warnings": [],
        "error": error,
    }


def generate_gemini_routes(
    *,
    region: str,
    restaurant: str,
    wait_minutes: int,
    companion: str,
    meal_time: str,
    interests: Sequence[str],
    candidates: Sequence[dict[str, Any]],
    variation: int = 0,
) -> dict[str, Any]:
    """
    Gemini Interactions API로 추천 코스 3개를 만든다.

    실패해도 예외를 main.py로 던지지 않고
    success=False를 반환한다.
    """
    settings = load_gemini_settings()

    prepared = _prepare_candidates(
        candidates,
        restaurant,
    )

    if not settings["api_key"]:
        return _failure(
            settings["model"],
            (
                "GEMINI_API_KEY가 "
                "설정되지 않았습니다."
            ),
        )

    if not prepared:
        return _failure(
            settings["model"],
            (
                "추천에 사용할 주변 장소가 "
                "없습니다."
            ),
        )

    place_count = _requested_place_count(
        int(wait_minutes),
        len(prepared),
    )

    prompt = _build_prompt(
        region=region,
        restaurant=restaurant,
        wait_minutes=int(wait_minutes),
        companion=companion,
        meal_time=meal_time,
        interests=interests,
        candidates=prepared,
        place_count=place_count,
        variation=int(variation),
    )

    client = None
    started = time.perf_counter()

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=settings["api_key"],
            http_options=types.HttpOptions(
                timeout=(
                    settings[
                        "timeout_seconds"
                    ]
                    * 1000
                ),
                retry_options=(
                    types.HttpRetryOptions(
                        attempts=1,
                    )
                ),
            ),
        )

        # 별도 연결 테스트에서 성공한
        # Interactions API 흐름을 그대로 사용한다.
        #
        # 불필요한 system_instruction,
        # generation_config를 제거해
        # SDK 호환 문제를 최소화한다.
        interaction = (
            client.interactions.create(
                model=settings["model"],
                input=prompt,
                response_format={
                    "type": "text",
                    "mime_type": (
                        "application/json"
                    ),
                    "schema": _route_schema(
                        place_count,
                        [
                            item["name"]
                            for item in prepared
                        ],
                    ),
                },
                store=False,
            )
        )

        latency = (
            time.perf_counter()
            - started
        )

        output_text = str(
            getattr(
                interaction,
                "output_text",
                "",
            )
            or ""
        ).strip()

        if not output_text:
            raise RuntimeError(
                "Gemini의 JSON 응답이 "
                "비어 있습니다."
            )

        # 구조화 출력이어도 방어적으로
        # JSON 코드블록 표시를 제거한다.
        output_text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            output_text,
            flags=re.IGNORECASE,
        ).strip()

        try:
            payload = json.loads(
                output_text
            )

        except json.JSONDecodeError as exc:
            preview = (
                output_text[:300]
                .replace("\n", " ")
            )

            raise RuntimeError(
                "Gemini JSON 해석 실패: "
                f"{preview}"
            ) from exc

        (
            routes,
            repair_count,
            warnings,
        ) = _validate_routes(
            payload,
            prepared,
            place_count,
            companion,
        )

        return {
            "success": True,
            "routes": routes,
            "model": settings["model"],
            "latency_seconds": round(
                latency,
                2,
            ),
            "interaction_id": str(
                getattr(
                    interaction,
                    "id",
                    "",
                )
                or ""
            ),
            "repair_count": repair_count,
            "warnings": warnings,
            "error": "",
        }

    except Exception as exc:
        latency = (
            time.perf_counter()
            - started
        )

        return _failure(
            settings["model"],
            f"{type(exc).__name__}: {exc}",
            latency,
        )

    finally:
        if client is not None:
            try:
                client.close()

            except Exception:
                pass