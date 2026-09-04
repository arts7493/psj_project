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
DEFAULT_RETRY_TIMEOUT_SECONDS = 30
MAX_CANDIDATES = 12
MAX_AI_PLACES = 4

ROUTE_TYPES = (
    "가까운 동네",
    "취향 집중",
    "식사 전후",
)

UNSUPPORTED_FACT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:km|킬로미터|m|미터)"
    r"|(?:도보|차량|자동차|택시|버스)\s*\d+\s*(?:분|시간)"
    r"|실시간\s*(?:혼잡|대기|영업)"
    r"|현재\s*(?:영업|휴무)",
    re.IGNORECASE,
)

COMPANION_CONFLICTS = {
    "혼자": re.compile(r"데이트|커플|연인과|아이와|자녀와|가족과"),
    "연인": re.compile(r"혼밥|혼자만|아이와|자녀와|가족\s*나들이"),
    "가족": re.compile(r"데이트|커플\s*전용|연인만|혼밥|혼자만"),
    "친구": re.compile(r"데이트\s*전용|커플\s*전용|혼밥|혼자만"),
}


def _read_local_secrets() -> dict[str, Any]:
    if not SECRETS_PATH.exists():
        return {}
    try:
        with SECRETS_PATH.open("rb") as file:
            return dict(tomllib.load(file))
    except Exception:
        return {}


def _read_streamlit_secrets() -> dict[str, Any]:
    try:
        import streamlit as st

        result: dict[str, Any] = {}
        for key in (
            "GEMINI_API_KEY",
            "GEMINI_MODEL",
            "AI_TIMEOUT_SECONDS",
            "AI_RETRY_TIMEOUT_SECONDS",
        ):
            try:
                if key in st.secrets:
                    result[key] = st.secrets[key]
            except Exception:
                pass
        return result
    except Exception:
        return {}


def _real_key(value: Any) -> str:
    text = str(value or "").strip()
    placeholders = (
        "PUT_YOUR_",
        "YOUR_API_KEY",
        "REPLACE_ME",
        "여기에",
        "실제_",
        "기존에_",
    )
    if not text:
        return ""
    if any(token.upper() in text.upper() for token in placeholders):
        return ""
    return text


def _safe_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(number, maximum))


def load_gemini_settings() -> dict[str, Any]:
    values = _read_local_secrets()
    for key, value in _read_streamlit_secrets().items():
        values.setdefault(key, value)
    for key in (
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "AI_TIMEOUT_SECONDS",
        "AI_RETRY_TIMEOUT_SECONDS",
    ):
        if key in os.environ:
            values[key] = os.environ[key]

    return {
        "api_key": _real_key(values.get("GEMINI_API_KEY")),
        "model": str(values.get("GEMINI_MODEL") or DEFAULT_MODEL).strip(),
        "timeout_seconds": _safe_int(values.get("AI_TIMEOUT_SECONDS"), DEFAULT_TIMEOUT_SECONDS, 15, 90),
        "retry_timeout_seconds": _safe_int(values.get("AI_RETRY_TIMEOUT_SECONDS"), DEFAULT_RETRY_TIMEOUT_SECONDS, 15, 60),
    }


def get_gemini_status() -> dict[str, Any]:
    settings = load_gemini_settings()
    return {
        "configured": bool(settings["api_key"]),
        "model": settings["model"],
        "timeout_seconds": settings["timeout_seconds"],
        "retry_timeout_seconds": settings["retry_timeout_seconds"],
    }


def _pick(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key, "")
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _safe_float(value: Any) -> float | None:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _prepare_candidates(candidates: Sequence[dict[str, Any]], restaurant: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    prepared: list[dict[str, Any]] = []
    original_by_id: dict[str, dict[str, Any]] = {}
    seen_names: set[str] = set()

    for item in candidates:
        name = _pick(item, "이름", "name")
        if not name or name == restaurant or name in seen_names:
            continue
        seen_names.add(name)
        candidate_id = f"P{len(prepared) + 1:02d}"
        category = _pick(item, "카테고리", "category") or "장소"
        tags = _pick(item, "태그", "키워드", "tags")
        compact = {
            "id": candidate_id,
            "name": name,
            "category": category,
            "distance_km": _safe_float(item.get("distance_km")),
            "same_area": bool(item.get("same_subregion") or item.get("same_area")),
            "interest_score": int(item.get("interest_score") or 0),
            "companion_score": int(item.get("companion_score") or 0),
            "tags": tags[:80],
        }
        original = {**item, "candidate_id": candidate_id, "name": name, "category": category}
        prepared.append(compact)
        original_by_id[candidate_id] = original
        if len(prepared) >= MAX_CANDIDATES:
            break

    return prepared, original_by_id


def _companion_rule(companion: str) -> str:
    return {
        "혼자": "혼자 이동하고 머물기 자연스러운 표현을 쓴다. 데이트, 커플, 연인, 아이, 가족 동반 표현은 쓰지 않는다.",
        "연인": "두 사람이 산책·사진·분위기를 함께 즐기는 방향으로 쓴다. 혼밥, 혼자만의 시간, 아이 동반 표현은 쓰지 않는다.",
        "가족": "세대가 함께 편안하게 식사하고 둘러보는 가족 나들이로 쓴다. 데이트, 커플 전용, 혼밥 표현은 쓰지 않는다.",
        "친구": "친구와 함께 둘러보고 대화하거나 체험하기 좋은 방향으로 쓴다. 커플 전용, 혼밥, 혼자만의 시간 표현은 쓰지 않는다.",
    }.get(companion, "선택한 동행 유형에 맞는 자연스러운 표현을 쓴다.")


def _schema(place_count: int, candidate_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "routes": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "route_type": {"type": "string", "enum": list(ROUTE_TYPES)},
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "place_ids": {
                            "type": "array",
                            "minItems": place_count,
                            "maxItems": place_count,
                            "items": {"type": "string", "enum": list(candidate_ids)},
                        },
                        "reasons": {
                            "type": "array",
                            "minItems": place_count,
                            "maxItems": place_count,
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["route_type", "title", "summary", "place_ids", "reasons"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["routes"],
        "additionalProperties": False,
    }


def _build_prompt(
    *,
    region: str,
    restaurant: str,
    course_label: str,
    course_minutes: int,
    companion: str,
    meal_time: str,
    interests: Sequence[str],
    candidates: Sequence[dict[str, Any]],
    place_count: int,
    variation: int,
    compact_retry: bool,
) -> str:
    interest_text = ", ".join(interests) if interests else "특별히 없음"
    candidate_json = json.dumps(
        list(candidates),
        ensure_ascii=False,
        separators=(",", ":") if compact_retry else None,
    )
    lodging_available = any(
        str(item.get("category") or "") == "숙소" for item in candidates
    )
    lodging_rule = (
        "반나절·하루 코스이므로 숙소 후보가 있으면 각 코스에 숙소를 정확히 1곳 넣고 place_ids의 마지막에 둔다. "
        "숙소는 대실·휴식 또는 숙박 연결 용도다."
        if course_label in {"반나절", "하루"} and lodging_available
        else "2시간·3시간 코스에는 숙소를 선택하지 않는다."
    )

    base = f"""
너는 전남광주 로컬 미식 코스 큐레이터다.

사용자 조건:
- 지역: {region}
- 오늘 방문할 식당: {restaurant}
- 코스 길이: {course_label} (약 {course_minutes}분)
- 식사 시간: {meal_time}
- 동행: {companion}
- 관심사: {interest_text}
- 재추천 번호: {variation}

동행 규칙:
{_companion_rule(companion)}

코스 역할:
1. 가까운 동네: same_area=true와 distance_km가 작은 후보를 우선한다.
2. 취향 집중: interest_score와 companion_score가 높은 후보를 우선한다.
3. 식사 전후: 관광명소·문화공간·카페가 한쪽으로 치우치지 않게 고른다.

반드시 지킬 규칙:
- 위 역할 순서대로 정확히 3개 코스를 만든다.
- 각 코스는 서로 다른 분위기와 가능한 한 다른 장소 조합을 사용한다.
- 각 코스의 place_ids는 정확히 {place_count}개이며 중복하지 않는다.
- 후보의 id만 사용한다. 오늘 방문할 식당은 앱이 따로 넣으므로 후보에서 선택하지 않는다.
- 거리·이동시간·영업시간·휴무·실시간 정보는 문장으로 추측하지 않는다.
- 제목, 요약, 이유는 짧고 자연스러운 한국어로 쓴다.
- 2시간·3시간 코스는 카페를 최대 1곳까지만 고른다.
- 반나절·하루 코스는 카페를 최대 2곳까지만 고른다.
- 카페만 반복하지 말고 가능한 경우 관광명소 또는 문화공간을 반드시 1곳 이상 포함한다.
- 같은 카테고리만 연속해서 선택하지 않는다.
- {lodging_rule}

후보:
{candidate_json}
""".strip()

    if compact_retry:
        base += (
            '\nJSON만 출력한다. 형식:'
            '{"routes":['
            '{"route_type":"가까운 동네","title":"제목","summary":"요약","place_ids":["P01"],"reasons":["이유"]},'
            '{"route_type":"취향 집중","title":"제목","summary":"요약","place_ids":["P02"],"reasons":["이유"]},'
            '{"route_type":"식사 전후","title":"제목","summary":"요약","place_ids":["P03"],"reasons":["이유"]}'
            ']}'
        )

    return base



def _extract_output_text(interaction: Any) -> str:
    text = str(getattr(interaction, "output_text", "") or "").strip()
    if text:
        return text
    outputs = getattr(interaction, "outputs", None) or []
    for output in reversed(outputs):
        text = str(getattr(output, "text", "") or "").strip()
        if text:
            return text
    return ""


def _json_payload(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("JSON 객체를 찾지 못했습니다.")
    return json.loads(cleaned[start : end + 1])


def _call_interaction(*, settings: dict[str, Any], prompt: str, timeout_seconds: int, schema: dict[str, Any] | None, thinking_level: str, max_output_tokens: int) -> tuple[dict[str, Any], str, float]:
    from google import genai
    from google.genai import types

    started = time.perf_counter()
    client = genai.Client(
        api_key=settings["api_key"],
        http_options=types.HttpOptions(
            timeout=timeout_seconds * 1000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    try:
        arguments: dict[str, Any] = {
            "model": settings["model"],
            "input": prompt,
            "generation_config": {
                "thinking_level": thinking_level,
                "temperature": 0.25,
                "max_output_tokens": max_output_tokens,
            },
            "store": False,
        }
        if schema is not None:
            arguments["response_format"] = {
                "type": "text",
                "mime_type": "application/json",
                "schema": schema,
            }
        interaction = client.interactions.create(**arguments)
        output_text = _extract_output_text(interaction)
        if not output_text:
            raise RuntimeError("Gemini 응답이 비어 있습니다.")
        payload = _json_payload(output_text)
        interaction_id = str(getattr(interaction, "id", "") or "")
        return payload, interaction_id, time.perf_counter() - started
    finally:
        try:
            client.close()
        except Exception:
            pass


def _fallback_title(route_type: str, companion: str) -> str:
    table = {
        "혼자": {"가까운 동네": "혼자 천천히 즐기는 동네 한 바퀴", "취향 집중": "내 취향대로 고르는 남도 산책", "식사 전후": "혼자서도 여유로운 미식 하루"},
        "연인": {"가까운 동네": "둘이 가볍게 즐기는 동네 데이트", "취향 집중": "취향을 나누는 남도 데이트", "식사 전후": "산책과 식사를 잇는 둘만의 코스"},
        "가족": {"가까운 동네": "가족과 편안한 동네 나들이", "취향 집중": "온 가족 취향을 담은 남도 여행", "식사 전후": "식사와 관광을 잇는 가족 하루"},
        "친구": {"가까운 동네": "친구와 가볍게 도는 동네 코스", "취향 집중": "친구와 취향대로 즐기는 남도 여행", "식사 전후": "맛과 이야기가 이어지는 친구 코스"},
    }
    return table.get(companion, {}).get(route_type, f"{route_type} 추천 코스")


def _fallback_summary(route_type: str, companion: str) -> str:
    return {
        "가까운 동네": "오늘 방문할 식당과 가까운 장소를 우선해 이동 부담을 줄였어요.",
        "취향 집중": f"{companion} 여행과 선택한 관심 분야를 중심으로 골랐어요.",
        "식사 전후": "식사 전 관광과 식사 후 휴식을 균형 있게 연결했어요.",
    }.get(route_type, "선택한 조건에 맞춘 지역 코스예요.")


def _safe_text(value: Any, fallback: str, companion: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    conflict = COMPANION_CONFLICTS.get(companion)
    if not text or UNSUPPORTED_FACT.search(text) or (conflict is not None and conflict.search(text)):
        text = fallback
    return text[:limit].rstrip()


def _candidate_fill_order(
    route_type: str,
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if route_type == "가까운 동네":
        return sorted(
            candidates,
            key=lambda item: (
                str(item.get("category") or "") == "숙소",
                not bool(item.get("same_area") or item.get("same_subregion")),
                item.get("distance_km") is None,
                float(item.get("distance_km") or 9999),
            ),
        )

    if route_type == "취향 집중":
        return sorted(
            candidates,
            key=lambda item: (
                str(item.get("category") or "") == "숙소",
                -int(item.get("interest_score") or 0),
                -int(item.get("companion_score") or 0),
                item.get("distance_km") is None,
                float(item.get("distance_km") or 9999),
            ),
        )

    category_order = {"관광명소": 0, "문화공간": 1, "카페": 2, "숙소": 3}
    return sorted(
        candidates,
        key=lambda item: (
            category_order.get(str(item.get("category")), 9),
            item.get("distance_km") is None,
            float(item.get("distance_km") or 9999),
        ),
    )



def _max_cafe_count(course_label: str) -> int:
    return 1 if course_label in ("2시간", "3시간") else 2


def _category_of(item: dict[str, Any]) -> str:
    return str(item.get("category") or item.get("카테고리") or "장소")


def _balance_selected_ids(
    selected_ids: list[str],
    prepared: Sequence[dict[str, Any]],
    route_type: str,
    course_label: str,
) -> list[str]:
    compact_by_id = {str(item["id"]): item for item in prepared}
    selected_ids = [candidate_id for candidate_id in selected_ids if candidate_id in compact_by_id]
    target_count = len(selected_ids)
    cafe_limit = _max_cafe_count(course_label)
    lodging_candidates = [
        item for item in prepared if _category_of(item) == "숙소"
    ]
    lodging_required = course_label in {"반나절", "하루"} and bool(lodging_candidates)

    def current_items() -> list[dict[str, Any]]:
        return [compact_by_id[candidate_id] for candidate_id in selected_ids]

    def available(predicate: Any) -> list[dict[str, Any]]:
        return [
            item for item in _candidate_fill_order(route_type, prepared)
            if str(item["id"]) not in selected_ids and predicate(item)
        ]

    # 짧은 코스에서는 숙소를 제거합니다.
    if not lodging_required:
        for index, candidate_id in list(enumerate(selected_ids)):
            if _category_of(compact_by_id[candidate_id]) != "숙소":
                continue
            replacements = available(lambda item: _category_of(item) != "숙소")
            if replacements:
                selected_ids[index] = str(replacements[0]["id"])

    # 반나절·하루에는 숙소 후보가 있으면 정확히 1곳을 포함하고 마지막에 둡니다.
    if lodging_required:
        lodging_ids = [
            candidate_id for candidate_id in selected_ids
            if _category_of(compact_by_id[candidate_id]) == "숙소"
        ]
        chosen_lodging = lodging_ids[0] if lodging_ids else str(
            sorted(
                lodging_candidates,
                key=lambda item: (
                    item.get("distance_km") is None,
                    float(item.get("distance_km") or 9999),
                ),
            )[0]["id"]
        )
        selected_ids = [
            candidate_id for candidate_id in selected_ids
            if _category_of(compact_by_id[candidate_id]) != "숙소"
        ]
        selected_ids = selected_ids[: max(0, target_count - 1)] + [chosen_lodging]

    # 카페 과다를 비카페 활동으로 교체합니다. 숙소는 카페 대체 대상으로 사용하지 않습니다.
    cafe_indices = [
        index for index, candidate_id in enumerate(selected_ids)
        if _category_of(compact_by_id[candidate_id]) == "카페"
    ]
    if len(cafe_indices) > cafe_limit:
        replacements = available(
            lambda item: _category_of(item) not in {"카페", "숙소"}
        )
        for index in cafe_indices[cafe_limit:]:
            if replacements:
                selected_ids[index] = str(replacements.pop(0)["id"])
            else:
                selected_ids[index] = ""
        selected_ids = [candidate_id for candidate_id in selected_ids if candidate_id]

    # 관광명소 또는 문화공간이 있다면 최소 한 곳 포함합니다.
    if not any(
        _category_of(item) in {"관광명소", "문화공간"}
        for item in current_items()
    ):
        scenic = available(
            lambda item: _category_of(item) in {"관광명소", "문화공간"}
        )
        if scenic:
            replace_index = next(
                (
                    index for index, candidate_id in enumerate(selected_ids)
                    if _category_of(compact_by_id[candidate_id]) == "카페"
                ),
                next(
                    (
                        index for index in range(len(selected_ids) - 1, -1, -1)
                        if _category_of(compact_by_id[selected_ids[index]]) != "숙소"
                    ),
                    None,
                ),
            )
            if replace_index is not None:
                selected_ids[replace_index] = str(scenic[0]["id"])

    # 중복을 제거하고 부족분을 채웁니다.
    deduped: list[str] = []
    for candidate_id in selected_ids:
        if candidate_id not in deduped:
            deduped.append(candidate_id)
    selected_ids = deduped

    while len(selected_ids) < target_count:
        cafe_count = sum(
            1 for candidate_id in selected_ids
            if _category_of(compact_by_id[candidate_id]) == "카페"
        )
        candidate = next(
            (
                item for item in _candidate_fill_order(route_type, prepared)
                if str(item["id"]) not in selected_ids
                and not (_category_of(item) == "카페" and cafe_count >= cafe_limit)
                and not (
                    _category_of(item) == "숙소"
                    and any(_category_of(compact_by_id[value]) == "숙소" for value in selected_ids)
                )
            ),
            None,
        )
        if candidate is None:
            break
        selected_ids.append(str(candidate["id"]))

    if lodging_required:
        lodging_ids = [
            candidate_id for candidate_id in selected_ids
            if _category_of(compact_by_id[candidate_id]) == "숙소"
        ]
        non_lodging_ids = [
            candidate_id for candidate_id in selected_ids
            if _category_of(compact_by_id[candidate_id]) != "숙소"
        ]
        selected_ids = non_lodging_ids + lodging_ids[:1]

    return selected_ids[:target_count]



def _validate_routes(payload: dict[str, Any], prepared: Sequence[dict[str, Any]], original_by_id: dict[str, dict[str, Any]], place_count: int, companion: str, course_label: str) -> tuple[list[dict[str, Any]], int, list[str]]:
    raw_routes = payload.get("routes")
    if not isinstance(raw_routes, list):
        raise ValueError("routes 배열이 없습니다.")

    raw_by_type: dict[str, dict[str, Any]] = {}
    leftovers: list[dict[str, Any]] = []
    for raw in raw_routes:
        if not isinstance(raw, dict):
            continue
        route_type = str(raw.get("route_type") or "").strip()
        if route_type in ROUTE_TYPES and route_type not in raw_by_type:
            raw_by_type[route_type] = raw
        else:
            leftovers.append(raw)

    compact_by_id = {str(item["id"]): item for item in prepared}
    validated: list[dict[str, Any]] = []
    repair_count = 0

    for route_type in ROUTE_TYPES:
        raw = raw_by_type.get(route_type) or (leftovers.pop(0) if leftovers else {})
        raw_ids = raw.get("place_ids") if isinstance(raw, dict) else []
        raw_reasons = raw.get("reasons") if isinstance(raw, dict) else []
        if not isinstance(raw_ids, list):
            raw_ids = []
        if not isinstance(raw_reasons, list):
            raw_reasons = []

        selected_ids: list[str] = []
        reasons_by_id: dict[str, str] = {}
        for index, value in enumerate(raw_ids):
            candidate_id = str(value or "").strip()
            if candidate_id not in compact_by_id or candidate_id in selected_ids:
                repair_count += 1
                continue
            selected_ids.append(candidate_id)
            reasons_by_id[candidate_id] = str(raw_reasons[index] if index < len(raw_reasons) else "")
            if len(selected_ids) == place_count:
                break

        if len(selected_ids) < place_count:
            for candidate in _candidate_fill_order(route_type, prepared):
                candidate_id = str(candidate["id"])
                if candidate_id in selected_ids:
                    continue
                selected_ids.append(candidate_id)
                repair_count += 1
                if len(selected_ids) == place_count:
                    break

        balanced_ids = _balance_selected_ids(selected_ids, prepared, route_type, course_label)
        if balanced_ids != selected_ids:
            repair_count += 1
        selected_ids = balanced_ids

        if len(selected_ids) < place_count:
            for candidate in _candidate_fill_order(route_type, prepared):
                candidate_id = str(candidate["id"])
                if candidate_id in selected_ids:
                    continue
                selected_ids.append(candidate_id)
                if len(selected_ids) == place_count:
                    break

        places: list[dict[str, Any]] = []
        for candidate_id in selected_ids[:place_count]:
            original = original_by_id[candidate_id]
            reason = _safe_text(reasons_by_id.get(candidate_id), _fallback_summary(route_type, companion), companion, 110)
            places.append({**original, "reason": reason})

        title = _safe_text(raw.get("title") if isinstance(raw, dict) else "", _fallback_title(route_type, companion), companion, 38)
        summary = _safe_text(raw.get("summary") if isinstance(raw, dict) else "", _fallback_summary(route_type, companion), companion, 125)
        validated.append({"route_type": route_type, "title": title, "summary": summary, "places": places})

    warnings: list[str] = []
    if repair_count:
        warnings.append(f"{repair_count}개 항목을 후보 데이터 기준으로 보정했습니다.")
    return validated, repair_count, warnings


def _failure(model: str, error: str, latency: float, timed_out: bool, attempts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "success": False,
        "routes": [],
        "model": model,
        "latency_seconds": round(latency, 2),
        "interaction_id": "",
        "repair_count": 0,
        "warnings": [],
        "error": error,
        "timed_out": timed_out,
        "attempts": list(attempts),
    }


def generate_gemini_routes(*, region: str, restaurant: str, course_label: str, course_minutes: int, companion: str, meal_time: str, interests: Sequence[str], candidates: Sequence[dict[str, Any]], place_count: int, variation: int = 0) -> dict[str, Any]:
    settings = load_gemini_settings()
    prepared, original_by_id = _prepare_candidates(candidates, restaurant)

    if not settings["api_key"]:
        return _failure(settings["model"], "GEMINI_API_KEY가 설정되지 않았습니다.", 0.0, False, [])
    if not prepared:
        return _failure(settings["model"], "추천에 사용할 주변 장소가 없습니다.", 0.0, False, [])

    place_count = max(1, min(int(place_count), len(prepared), MAX_AI_PLACES))
    schema = _schema(place_count, [str(item["id"]) for item in prepared])

    attempts: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    errors: list[str] = []
    interaction_id = ""
    payload: dict[str, Any] | None = None

    call_specs = (
        {"name": "structured", "timeout": settings["timeout_seconds"], "schema": schema, "thinking": "low", "max_tokens": 1050, "compact_retry": False},
        {"name": "compact_retry", "timeout": settings["retry_timeout_seconds"], "schema": None, "thinking": "minimal", "max_tokens": 900, "compact_retry": True},
    )

    for spec in call_specs:
        prompt = _build_prompt(
            region=region,
            restaurant=restaurant,
            course_label=course_label,
            course_minutes=course_minutes,
            companion=companion,
            meal_time=meal_time,
            interests=interests,
            candidates=prepared,
            place_count=place_count,
            variation=variation,
            compact_retry=bool(spec["compact_retry"]),
        )
        started = time.perf_counter()
        try:
            payload, interaction_id, latency = _call_interaction(
                settings=settings,
                prompt=prompt,
                timeout_seconds=int(spec["timeout"]),
                schema=spec["schema"],
                thinking_level=str(spec["thinking"]),
                max_output_tokens=int(spec["max_tokens"]),
            )
            attempts.append({"name": spec["name"], "success": True, "latency_seconds": round(latency, 2), "error": ""})
            break
        except Exception as exc:
            latency = time.perf_counter() - started
            message = f"{type(exc).__name__}: {exc}"
            errors.append(f"{spec['name']}: {message}")
            attempts.append({"name": spec["name"], "success": False, "latency_seconds": round(latency, 2), "error": message})
            lowered = message.lower()
            if "429" in lowered or "quota" in lowered or "rate limit" in lowered or "too_many_requests" in lowered:
                break

    total_latency = time.perf_counter() - total_started
    if payload is None:
        timed_out = any("timeout" in error.lower() or "timed out" in error.lower() for error in errors)
        return _failure(settings["model"], " | ".join(errors), total_latency, timed_out, attempts)

    try:
        routes, repair_count, warnings = _validate_routes(payload, prepared, original_by_id, place_count, companion, course_label)
    except Exception as exc:
        return _failure(settings["model"], f"응답 검증 실패: {type(exc).__name__}: {exc}", total_latency, False, attempts)

    return {
        "success": True,
        "routes": routes,
        "model": settings["model"],
        "latency_seconds": round(total_latency, 2),
        "interaction_id": interaction_id,
        "repair_count": repair_count,
        "warnings": warnings,
        "error": "",
        "timed_out": False,
        "attempts": attempts,
    }
