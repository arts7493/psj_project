from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


BASE_DIR = Path(__file__).resolve().parent
SECRETS_PATH = BASE_DIR / ".streamlit" / "secrets.toml"

# 사용자가 확인한 신규 호스트를 우선 사용하고 공식 문서 호스트를 예비로 둔다.
GEOCODE_URLS = (
    "https://maps.apigw.ntruss.com/map-geocode/v2/geocode",
    "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode",
)
DIRECTIONS_URLS = (
    "https://maps.apigw.ntruss.com/map-direction/v1/driving",
    "https://naveropenapi.apigw.ntruss.com/map-direction/v1/driving",
)

DEFAULT_TIMEOUT_SECONDS = 8
WALK_SPEED_KMH = 4.5
WALK_ROUTE_FACTOR = 1.25


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
            "NAVER_MAP_CLIENT_ID",
            "NAVER_MAP_CLIENT_SECRET",
            "NAVER_MAP_TIMEOUT_SECONDS",
        ):
            try:
                if key in st.secrets:
                    result[key] = st.secrets[key]
            except Exception:
                pass
        return result
    except Exception:
        return {}


def _real_secret(value: Any) -> str:
    text = str(value or "").strip()
    placeholders = (
        "PUT_YOUR_",
        "YOUR_CLIENT_",
        "REPLACE_ME",
        "여기에",
        "실제_",
        "발급받은_",
    )

    if not text:
        return ""

    if any(token.upper() in text.upper() for token in placeholders):
        return ""

    return text


def load_naver_settings() -> dict[str, Any]:
    values = _read_local_secrets()

    for key, value in _read_streamlit_secrets().items():
        values.setdefault(key, value)

    for key in (
        "NAVER_MAP_CLIENT_ID",
        "NAVER_MAP_CLIENT_SECRET",
        "NAVER_MAP_TIMEOUT_SECONDS",
    ):
        if key in os.environ:
            values[key] = os.environ[key]

    try:
        timeout = int(float(values.get("NAVER_MAP_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)))
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_SECONDS

    return {
        "client_id": _real_secret(values.get("NAVER_MAP_CLIENT_ID")),
        "client_secret": _real_secret(values.get("NAVER_MAP_CLIENT_SECRET")),
        "timeout_seconds": max(3, min(timeout, 30)),
    }


def get_naver_status() -> dict[str, Any]:
    settings = load_naver_settings()
    return {
        "configured": bool(settings["client_id"] and settings["client_secret"]),
        "timeout_seconds": settings["timeout_seconds"],
    }


def _headers(client_id: str, client_secret: str) -> dict[str, str]:
    return {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
        "Accept": "application/json",
    }


def _request_json(
    urls: Iterable[str],
    *,
    params: dict[str, Any],
    client_id: str,
    client_secret: str,
    timeout_seconds: int,
) -> tuple[dict[str, Any] | None, str, str]:
    errors: list[str] = []

    for url in urls:
        try:
            response = requests.get(
                url,
                headers=_headers(client_id, client_secret),
                params=params,
                timeout=(3.05, timeout_seconds),
            )
        except requests.RequestException as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue

        if response.status_code == 200:
            try:
                return response.json(), "", url
            except ValueError:
                errors.append("응답이 JSON 형식이 아닙니다.")
                continue

        body = response.text.strip().replace("\n", " ")[:240]
        errors.append(f"HTTP {response.status_code}: {body}")

        # 인증·할당량 오류는 다른 호스트에서도 같은 키로 실패할 가능성이 높다.
        if response.status_code in {400, 401, 403, 429}:
            break

    return None, " | ".join(errors) or "네이버 지도 API 요청에 실패했습니다.", ""


def normalize_address(address: str) -> str:
    text = " ".join(str(address or "").split())
    text = text.replace("전남광주", "광주광역시")
    return text


def _address_part(address: dict[str, Any], part_type: str) -> str:
    for element in address.get("addressElements", []) or []:
        types = element.get("types") or element.get("type") or []
        if part_type in types:
            return str(element.get("longName") or element.get("shortName") or "").strip()
    return ""


@lru_cache(maxsize=512)
def _geocode_cached(
    address: str,
    client_id: str,
    client_secret: str,
    timeout_seconds: int,
) -> tuple[Any, ...]:
    payload, error, endpoint = _request_json(
        GEOCODE_URLS,
        params={"query": address, "count": 1},
        client_id=client_id,
        client_secret=client_secret,
        timeout_seconds=timeout_seconds,
    )

    if payload is None:
        return False, None, None, "", "", "", "", "", error

    addresses = payload.get("addresses") or []
    if not addresses:
        return False, None, None, "", "", "", "", endpoint, "주소 검색 결과가 없습니다."

    item = addresses[0]
    try:
        lng = float(item["x"])
        lat = float(item["y"])
    except (KeyError, TypeError, ValueError):
        return False, None, None, "", "", "", "", endpoint, "좌표 형식이 올바르지 않습니다."

    return (
        True,
        lat,
        lng,
        str(item.get("roadAddress") or "").strip(),
        str(item.get("jibunAddress") or "").strip(),
        _address_part(item, "SIGUGUN"),
        _address_part(item, "DONGMYUN"),
        endpoint,
        "",
    )


def geocode(address: str) -> dict[str, Any]:
    settings = load_naver_settings()
    clean_address = normalize_address(address)

    if not clean_address:
        return {"ok": False, "error": "주소가 비어 있습니다.", "source": ""}

    if not (settings["client_id"] and settings["client_secret"]):
        return {
            "ok": False,
            "error": "네이버 지도 API 키가 설정되지 않았습니다.",
            "source": "",
        }

    (
        ok,
        lat,
        lng,
        road_address,
        jibun_address,
        sigugun,
        dongmyun,
        endpoint,
        error,
    ) = _geocode_cached(
        clean_address,
        settings["client_id"],
        settings["client_secret"],
        settings["timeout_seconds"],
    )

    return {
        "ok": bool(ok),
        "lat": lat,
        "lng": lng,
        "road_address": road_address,
        "jibun_address": jibun_address,
        "sigugun": sigugun,
        "dongmyun": dongmyun,
        "source": "naver" if ok else "",
        "endpoint": endpoint,
        "error": error,
    }


def geocode_many(items: list[dict[str, Any]], max_workers: int = 4) -> list[dict[str, Any]]:
    if not items:
        return []

    results: list[dict[str, Any] | None] = [None] * len(items)

    def task(index: int, item: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        address = str(item.get("주소") or item.get("address") or "").strip()
        return index, {**item, **geocode(address)}

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(items)))) as executor:
        futures = [executor.submit(task, index, item) for index, item in enumerate(items)]
        for future in as_completed(futures):
            try:
                index, result = future.result()
                results[index] = result
            except Exception:
                pass

    return [result or dict(items[index]) for index, result in enumerate(results)]


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)

    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def straight_distance_km(start: dict[str, Any], goal: dict[str, Any]) -> float | None:
    try:
        return haversine_km(
            float(start["lat"]),
            float(start["lng"]),
            float(goal["lat"]),
            float(goal["lng"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def estimate_walk(start: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
    direct_km = straight_distance_km(start, goal)
    if direct_km is None:
        return {"ok": False, "error": "도보 거리를 계산할 좌표가 없습니다."}

    estimated_km = max(0.05, direct_km * WALK_ROUTE_FACTOR)
    duration_min = max(1, math.ceil(estimated_km / WALK_SPEED_KMH * 60))
    return {
        "ok": True,
        "distance_km": round(estimated_km, 2),
        "duration_min": duration_min,
        "estimated": True,
        "error": "",
    }


@lru_cache(maxsize=512)
def _driving_cached(
    start_lng: float,
    start_lat: float,
    goal_lng: float,
    goal_lat: float,
    client_id: str,
    client_secret: str,
    timeout_seconds: int,
) -> tuple[Any, ...]:
    payload, error, endpoint = _request_json(
        DIRECTIONS_URLS,
        params={
            "start": f"{start_lng:.7f},{start_lat:.7f}",
            "goal": f"{goal_lng:.7f},{goal_lat:.7f}",
            "option": "traoptimal",
            "lang": "ko",
        },
        client_id=client_id,
        client_secret=client_secret,
        timeout_seconds=timeout_seconds,
    )

    if payload is None:
        return False, None, None, 0, 0, 0, "", endpoint, error

    if payload.get("code") not in (0, "0", None):
        return (
            False,
            None,
            None,
            0,
            0,
            0,
            "",
            endpoint,
            str(payload.get("message") or "경로를 찾지 못했습니다."),
        )

    routes = ((payload.get("route") or {}).get("traoptimal") or [])
    if not routes:
        return False, None, None, 0, 0, 0, "", endpoint, "자동차 경로가 없습니다."

    summary = routes[0].get("summary") or {}
    try:
        distance_km = round(float(summary.get("distance", 0)) / 1000, 2)
        duration_min = max(1, round(float(summary.get("duration", 0)) / 60000))
    except (TypeError, ValueError):
        return False, None, None, 0, 0, 0, "", endpoint, "경로 응답 형식이 올바르지 않습니다."

    return (
        True,
        distance_km,
        duration_min,
        int(summary.get("taxiFare") or 0),
        int(summary.get("tollFare") or 0),
        int(summary.get("fuelPrice") or 0),
        str(summary.get("departureTime") or payload.get("currentDateTime") or ""),
        endpoint,
        "",
    )


def get_driving_route(start: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
    settings = load_naver_settings()
    if not (settings["client_id"] and settings["client_secret"]):
        return {"ok": False, "error": "네이버 지도 API 키가 설정되지 않았습니다."}

    try:
        start_lng = float(start["lng"])
        start_lat = float(start["lat"])
        goal_lng = float(goal["lng"])
        goal_lat = float(goal["lat"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "error": "자동차 경로를 계산할 좌표가 없습니다."}

    (
        ok,
        distance_km,
        duration_min,
        taxi_fare,
        toll_fare,
        fuel_price,
        departure_time,
        endpoint,
        error,
    ) = _driving_cached(
        start_lng,
        start_lat,
        goal_lng,
        goal_lat,
        settings["client_id"],
        settings["client_secret"],
        settings["timeout_seconds"],
    )

    return {
        "ok": bool(ok),
        "distance_km": distance_km,
        "duration_min": duration_min,
        "taxi_fare": taxi_fare,
        "toll_fare": toll_fare,
        "fuel_price": fuel_price,
        "departure_time": departure_time,
        "source": "naver" if ok else "",
        "endpoint": endpoint,
        "error": error,
    }


def build_naver_search_url(name: str, address: str = "") -> str:
    query = " ".join(part for part in (str(name).strip(), normalize_address(address)) if part)
    return f"https://map.naver.com/p/search/{quote(query, safe='')}"


def clear_map_cache() -> None:
    _geocode_cached.cache_clear()
    _driving_cached.cache_clear()
