from __future__ import annotations

from datetime import datetime
from io import BytesIO
import math
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


QR_COOLDOWN_SECONDS = 60

# 시연용으로 허용되는 QR 페이로드입니다.
# 이 목록에 없는 QR은 포인트 체크인에 사용할 수 없습니다.
DEMO_QR_CODES: dict[str, dict[str, Any]] = {
    "UNNAM-HWAITING|CHECKIN|01": {
        "code": "QR-01",
        "label": "시연 체크인 QR 1",
        "points": 100,
    },
    "UNNAM-HWAITING|CHECKIN|02": {
        "code": "QR-02",
        "label": "시연 체크인 QR 2",
        "points": 100,
    },
    "UNNAM-HWAITING|CHECKIN|03": {
        "code": "QR-03",
        "label": "시연 체크인 QR 3",
        "points": 100,
    },
    "UNNAM-HWAITING|CHECKIN|04": {
        "code": "QR-04",
        "label": "시연 체크인 QR 4",
        "points": 100,
    },
    "UNNAM-HWAITING|CHECKIN|05": {
        "code": "QR-05",
        "label": "시연 체크인 QR 5",
        "points": 100,
    },
}


def _pil_to_bgr(image_bytes: bytes) -> np.ndarray:
    """이미지 바이트를 EXIF 회전까지 보정한 OpenCV BGR 이미지로 변환합니다."""
    if not image_bytes:
        raise ValueError("이미지 데이터가 비어 있습니다.")

    with Image.open(BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.array(image)

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _decode_once(detector: cv2.QRCodeDetector, image: np.ndarray) -> str:
    """단일/다중 QR 디코딩을 순서대로 시도합니다."""
    try:
        decoded, _, _ = detector.detectAndDecode(image)
        if decoded:
            return str(decoded).strip()
    except Exception:
        pass

    try:
        result = detector.detectAndDecodeMulti(image)
        if len(result) >= 2:
            detected = bool(result[0])
            values = result[1] or []
            if detected:
                for value in values:
                    text = str(value or "").strip()
                    if text:
                        return text
    except Exception:
        pass

    return ""


def decode_qr_image(image_bytes: bytes) -> dict[str, Any]:
    """
    카메라 또는 업로드 이미지에서 QR 문자열을 읽습니다.

    원본, 확대, 회전, 그레이스케일, 이진화 버전을 순차적으로 검사해
    사진 품질이 조금 낮아도 시연용 QR을 읽을 수 있게 합니다.
    """
    try:
        bgr = _pil_to_bgr(image_bytes)
    except Exception as exc:
        return {
            "ok": False,
            "payload": "",
            "error": f"이미지를 읽지 못했습니다: {exc}",
        }

    detector = cv2.QRCodeDetector()
    variants: list[np.ndarray] = []

    # 원본과 방향 회전본
    variants.append(bgr)
    variants.append(cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE))
    variants.append(cv2.rotate(bgr, cv2.ROTATE_180))
    variants.append(cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE))

    # 확대본: 작은 QR 사진 대응
    height, width = bgr.shape[:2]
    if max(height, width) < 1800:
        enlarged = cv2.resize(
            bgr,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )
        variants.append(enlarged)

    # 그레이스케일과 대비 보정본
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    variants.append(gray)

    try:
        equalized = cv2.equalizeHist(gray)
        variants.append(equalized)

        _, otsu = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        variants.append(otsu)
    except Exception:
        pass

    for variant in variants:
        payload = _decode_once(detector, variant)
        if payload:
            return {
                "ok": True,
                "payload": payload,
                "error": "",
            }

    return {
        "ok": False,
        "payload": "",
        "error": "QR 코드를 찾지 못했습니다. QR이 사진 안에 크게 보이도록 다시 촬영해 주세요.",
    }


def get_demo_qr_info(payload: str) -> dict[str, Any] | None:
    """허용된 시연용 QR의 정보만 반환합니다."""
    return DEMO_QR_CODES.get(str(payload or "").strip())


def cooldown_remaining(
    last_scanned_at: str | None,
    *,
    now: datetime | None = None,
) -> int:
    """같은 QR을 다시 사용할 수 있을 때까지 남은 초를 반환합니다."""
    if not last_scanned_at:
        return 0

    current = now or datetime.now().astimezone()

    try:
        previous = datetime.fromisoformat(last_scanned_at)
        if previous.tzinfo is None:
            previous = previous.replace(tzinfo=current.tzinfo)
    except (TypeError, ValueError):
        return 0

    elapsed = (current - previous).total_seconds()
    return max(0, math.ceil(QR_COOLDOWN_SECONDS - elapsed))
