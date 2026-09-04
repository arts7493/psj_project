from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha1
import logging
import threading
import time
from typing import Any

import cv2
import numpy as np
import streamlit as st

try:
    import av
except ImportError:  # requirements 설치 전에도 앱 전체가 즉시 종료되지 않도록 처리
    av = None  # type: ignore[assignment]

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
except ImportError:  # 업로드 방식은 계속 사용할 수 있도록 처리
    WebRtcMode = None  # type: ignore[assignment]
    webrtc_streamer = None  # type: ignore[assignment]


logging.getLogger("streamlit_webrtc").setLevel(logging.WARNING)
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)


@dataclass(frozen=True)
class QRDetectionEvent:
    event_id: int
    payload: str
    detected_at: float
    token: str


@dataclass(frozen=True)
class _DetectionVariant:
    image: np.ndarray
    offset_x: float
    offset_y: float
    scale_x: float
    scale_y: float


class LiveQRScanner:
    """WebRTC 프레임에서 QR을 읽고 이벤트 큐로 전달하는 스레드 안전 스캐너입니다."""

    def __init__(
        self,
        *,
        scan_interval_seconds: float = 0.12,
        disappear_seconds: float = 0.75,
    ) -> None:
        self._lock = threading.Lock()
        self._detector_lock = threading.Lock()
        self._detector = cv2.QRCodeDetector()
        try:
            self._detector.setEpsX(0.35)
            self._detector.setEpsY(0.35)
        except Exception:
            pass

        self._scan_interval = max(0.08, float(scan_interval_seconds))
        self._disappear_seconds = max(0.45, float(disappear_seconds))
        self._last_scan_monotonic = 0.0
        self._last_frame_monotonic = 0.0
        self._last_seen_monotonic = 0.0
        self._visible_payload = ""
        self._last_emitted_payload = ""
        self._event_sequence = 0
        self._events: deque[QRDetectionEvent] = deque(maxlen=12)
        self._latest_event: QRDetectionEvent | None = None
        self._last_detection_points: np.ndarray | None = None

    @staticmethod
    def _resize_with_mapping(
        image: np.ndarray,
        *,
        max_width: int = 1280,
        min_width: int = 0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> _DetectionVariant:
        height, width = image.shape[:2]
        target_width = width
        if width > max_width:
            target_width = max_width
        elif min_width and width < min_width:
            target_width = min(min_width, max_width)

        if target_width == width:
            return _DetectionVariant(image, offset_x, offset_y, 1.0, 1.0)

        scale = target_width / float(width)
        resized = cv2.resize(
            image,
            (target_width, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        return _DetectionVariant(resized, offset_x, offset_y, scale, scale)

    @classmethod
    def _build_variants(cls, image: np.ndarray) -> list[_DetectionVariant]:
        """전체 프레임과 중앙 영역의 대비 보정본을 준비합니다."""
        height, width = image.shape[:2]
        variants: list[_DetectionVariant] = []

        full = cls._resize_with_mapping(image, max_width=1280)
        variants.append(full)

        # 사용자가 보는 중앙 가이드보다 약간 넓게 잘라 작은 QR을 확대합니다.
        crop_size = max(120, int(min(width, height) * 0.78))
        left = max(0, (width - crop_size) // 2)
        top = max(0, (height - crop_size) // 2)
        crop = image[top : top + crop_size, left : left + crop_size]
        center = cls._resize_with_mapping(
            crop,
            max_width=1200,
            min_width=900,
            offset_x=float(left),
            offset_y=float(top),
        )
        variants.append(center)

        gray = cv2.cvtColor(center.image, cv2.COLOR_BGR2GRAY)
        variants.append(
            _DetectionVariant(
                gray,
                center.offset_x,
                center.offset_y,
                center.scale_x,
                center.scale_y,
            )
        )

        try:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
            variants.append(
                _DetectionVariant(
                    clahe,
                    center.offset_x,
                    center.offset_y,
                    center.scale_x,
                    center.scale_y,
                )
            )

            blurred = cv2.GaussianBlur(clahe, (0, 0), 1.1)
            sharpened = cv2.addWeighted(clahe, 1.7, blurred, -0.7, 0)
            variants.append(
                _DetectionVariant(
                    sharpened,
                    center.offset_x,
                    center.offset_y,
                    center.scale_x,
                    center.scale_y,
                )
            )

            _, otsu = cv2.threshold(
                clahe,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            variants.append(
                _DetectionVariant(
                    otsu,
                    center.offset_x,
                    center.offset_y,
                    center.scale_x,
                    center.scale_y,
                )
            )
        except Exception:
            pass

        return variants

    @staticmethod
    def _map_points(points: Any, variant: _DetectionVariant) -> np.ndarray | None:
        if points is None:
            return None
        try:
            polygon = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            polygon[:, 0] = polygon[:, 0] / variant.scale_x + variant.offset_x
            polygon[:, 1] = polygon[:, 1] / variant.scale_y + variant.offset_y
            return polygon
        except Exception:
            return None

    def _decode_variant(self, variant: _DetectionVariant) -> tuple[str, np.ndarray | None]:
        image = variant.image
        with self._detector_lock:
            try:
                decoded, points, _ = self._detector.detectAndDecode(image)
                decoded = str(decoded or "").strip()
                if decoded:
                    return decoded, self._map_points(points, variant)
            except Exception:
                pass

            try:
                multi = self._detector.detectAndDecodeMulti(image)
                if len(multi) >= 3 and bool(multi[0]):
                    values = multi[1] or []
                    polygons = multi[2]
                    for index, value in enumerate(values):
                        decoded = str(value or "").strip()
                        if decoded:
                            points = polygons[index] if polygons is not None else None
                            return decoded, self._map_points(points, variant)
            except Exception:
                pass

            try:
                decoded, points, _ = self._detector.detectAndDecodeCurved(image)
                decoded = str(decoded or "").strip()
                if decoded:
                    return decoded, self._map_points(points, variant)
            except Exception:
                pass

        return "", None

    def decode_bgr_image(self, image: np.ndarray) -> tuple[str, np.ndarray | None]:
        """정적 이미지와 실시간 프레임에서 공통으로 사용하는 QR 디코더입니다."""
        if image is None or image.size == 0:
            return "", None
        for variant in self._build_variants(image):
            decoded, points = self._decode_variant(variant)
            if decoded:
                return decoded, points
        return "", None

    @staticmethod
    def _draw_guide(image: np.ndarray) -> None:
        height, width = image.shape[:2]
        size = int(min(width, height) * 0.58)
        left = max(8, (width - size) // 2)
        top = max(8, (height - size) // 2)
        right = min(width - 8, left + size)
        bottom = min(height - 8, top + size)
        length = max(18, int(size * 0.15))
        color = (255, 255, 255)
        thickness = max(2, int(min(width, height) / 260))
        for start, end in (
            ((left, top + length), (left, top)),
            ((left, top), (left + length, top)),
            ((right - length, top), (right, top)),
            ((right, top), (right, top + length)),
            ((left, bottom - length), (left, bottom)),
            ((left, bottom), (left + length, bottom)),
            ((right - length, bottom), (right, bottom)),
            ((right, bottom), (right, bottom - length)),
        ):
            cv2.line(image, start, end, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_detection(image: np.ndarray, points: np.ndarray | None) -> None:
        if points is None:
            return
        try:
            polygon = np.round(points).astype(np.int32).reshape(-1, 2)
            cv2.polylines(
                image,
                [polygon],
                isClosed=True,
                color=(40, 220, 125),
                thickness=4,
                lineType=cv2.LINE_AA,
            )
        except Exception:
            pass

    def _publish_detection(self, decoded: str, points: np.ndarray | None, now: float) -> None:
        with self._lock:
            if decoded:
                payload_changed = decoded != self._last_emitted_payload
                disappeared_long_enough = (
                    self._visible_payload == ""
                    and now - self._last_seen_monotonic >= self._disappear_seconds
                )
                self._visible_payload = decoded
                self._last_seen_monotonic = now
                self._last_detection_points = points

                if payload_changed or disappeared_long_enough:
                    self._event_sequence += 1
                    event = QRDetectionEvent(
                        event_id=self._event_sequence,
                        payload=decoded,
                        detected_at=time.time(),
                        token=sha1(
                            f"{decoded}|{self._event_sequence}|{now}".encode("utf-8")
                        ).hexdigest()[:16],
                    )
                    self._latest_event = event
                    self._events.append(event)
                    self._last_emitted_payload = decoded
            elif now - self._last_seen_monotonic >= self._disappear_seconds:
                self._visible_payload = ""
                self._last_detection_points = None

    def process_frame(self, frame: Any) -> Any:
        if av is None:
            return frame

        image = frame.to_ndarray(format="bgr24")
        display = image.copy()
        now = time.monotonic()

        with self._lock:
            self._last_frame_monotonic = now
            should_scan = now - self._last_scan_monotonic >= self._scan_interval
            if should_scan:
                self._last_scan_monotonic = now

        decoded = ""
        points: np.ndarray | None = None
        if should_scan:
            try:
                decoded, points = self.decode_bgr_image(image)
            except Exception:
                decoded, points = "", None
            self._publish_detection(decoded, points, now)

        with self._lock:
            visible_payload = self._visible_payload
            visible_points = self._last_detection_points

        if visible_payload:
            self._draw_detection(display, points if points is not None else visible_points)
            cv2.rectangle(display, (0, 0), (display.shape[1], 46), (25, 130, 85), -1)
            cv2.putText(
                display,
                "QR DETECTED",
                (16, 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.74,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            self._draw_guide(display)

        return av.VideoFrame.from_ndarray(display, format="bgr24")

    def pop_event(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._events:
                return None
            event = self._events.popleft()
            return {
                "event_id": event.event_id,
                "payload": event.payload,
                "detected_at": event.detected_at,
                "token": event.token,
            }

    def latest_event(self) -> dict[str, Any] | None:
        with self._lock:
            event = self._latest_event
            if event is None:
                return None
            return {
                "event_id": event.event_id,
                "payload": event.payload,
                "detected_at": event.detected_at,
                "token": event.token,
            }

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            return {
                "stream_active": now - self._last_frame_monotonic < 2.5,
                "visible_payload": self._visible_payload,
                "latest_event_id": self._latest_event.event_id if self._latest_event else 0,
                "pending_events": len(self._events),
            }

    def reset_visibility(self) -> None:
        with self._lock:
            self._visible_payload = ""
            self._last_seen_monotonic = 0.0
            self._last_detection_points = None
            self._events.clear()


def live_qr_available() -> bool:
    return av is not None and webrtc_streamer is not None and WebRtcMode is not None


def _rtc_configuration() -> dict[str, Any]:
    ice_servers: list[dict[str, Any]] = [
        {"urls": ["stun:stun.l.google.com:19302"]},
    ]

    try:
        turn_url = str(st.secrets.get("WEBRTC_TURN_URL", "") or "").strip()
        turn_username = str(st.secrets.get("WEBRTC_TURN_USERNAME", "") or "").strip()
        turn_credential = str(st.secrets.get("WEBRTC_TURN_CREDENTIAL", "") or "").strip()
    except Exception:
        turn_url = ""
        turn_username = ""
        turn_credential = ""

    if turn_url:
        turn_server: dict[str, Any] = {"urls": [turn_url]}
        if turn_username:
            turn_server["username"] = turn_username
        if turn_credential:
            turn_server["credential"] = turn_credential
        ice_servers.append(turn_server)

    return {"iceServers": ice_servers}


def render_live_qr_camera(scanner: LiveQRScanner, *, key: str) -> Any | None:
    """기본 카메라를 자동으로 요청하고 프레임마다 QR을 판독합니다."""
    if not live_qr_available():
        st.error(
            "실시간 QR 카메라 패키지가 설치되지 않았어요. "
            "`py -m pip install -U -r requirements.txt` 실행 후 서버를 다시 시작해 주세요."
        )
        return None

    return webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV,
        desired_playing_state=True,
        rtc_configuration=_rtc_configuration(),
        # 별도 장치 선택을 강제하지 않고 브라우저의 기본 카메라를 사용합니다.
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=scanner.process_frame,
        on_video_ended=scanner.reset_visibility,
        async_processing=False,
        # 최신 버전에서는 스트리밍 중 카메라 아이콘에서 다른 장치로 전환할 수 있습니다.
        media_toggle_controls=True,
        sendback_video=True,
        sendback_audio=False,
        video_html_attrs={
            "autoPlay": True,
            "controls": False,
            "muted": True,
            "playsInline": True,
            "style": {
                "width": "100%",
                "borderRadius": "16px",
                "backgroundColor": "#0d1714",
            },
        },
        translations={
            "start": "카메라 연결",
            "stop": "카메라 중지",
            "select_device": "카메라 변경",
            "select_camera": "카메라 변경",
            "device_ask_permission": "카메라 사용 권한을 허용해 주세요.",
            "device_access_denied": "카메라 권한이 차단되어 있어요.",
            "device_not_available": "사용할 수 있는 카메라를 찾지 못했어요.",
            "media_api_not_available": "이 브라우저에서는 카메라를 사용할 수 없어요.",
        },
    )
