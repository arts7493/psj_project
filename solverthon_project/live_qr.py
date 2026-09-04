from __future__ import annotations

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
except ImportError:  # requirements 설치 전에도 앱 전체가 바로 죽지 않도록 처리
    av = None  # type: ignore[assignment]

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
except ImportError:  # requirements 설치 전에도 업로드 방식은 계속 사용할 수 있음
    WebRtcMode = None  # type: ignore[assignment]
    webrtc_streamer = None  # type: ignore[assignment]


# WebRTC 하위 라이브러리의 과도한 INFO 로그를 줄입니다.
logging.getLogger("streamlit_webrtc").setLevel(logging.WARNING)
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)


@dataclass(frozen=True)
class QRDetectionEvent:
    event_id: int
    payload: str
    detected_at: float
    token: str


class LiveQRScanner:
    """WebRTC 영상 프레임에서 QR을 주기적으로 읽는 스레드 안전 스캐너입니다."""

    def __init__(
        self,
        *,
        scan_interval_seconds: float = 0.18,
        disappear_seconds: float = 0.9,
    ) -> None:
        self._lock = threading.Lock()
        self._detector_lock = threading.Lock()
        self._detector = cv2.QRCodeDetector()
        self._scan_interval = max(0.08, float(scan_interval_seconds))
        self._disappear_seconds = max(0.4, float(disappear_seconds))

        self._last_scan_monotonic = 0.0
        self._last_frame_monotonic = 0.0
        self._last_seen_monotonic = 0.0
        self._visible_payload = ""
        self._last_emitted_payload = ""
        self._event_sequence = 0
        self._latest_event: QRDetectionEvent | None = None

    @staticmethod
    def _resize_for_detection(image: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        max_width = 960
        if width <= max_width:
            return image, 1.0

        scale = max_width / float(width)
        resized = cv2.resize(
            image,
            (max_width, max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    @staticmethod
    def _draw_guide(image: np.ndarray) -> None:
        height, width = image.shape[:2]
        size = int(min(width, height) * 0.52)
        left = max(8, (width - size) // 2)
        top = max(8, (height - size) // 2)
        right = min(width - 8, left + size)
        bottom = min(height - 8, top + size)
        length = max(18, int(size * 0.16))
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
    def _draw_detection(
        image: np.ndarray,
        points: Any,
        *,
        scale: float,
    ) -> None:
        if points is None:
            return

        try:
            polygon = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            if scale and scale != 1.0:
                polygon = polygon / scale
            polygon = np.round(polygon).astype(np.int32)
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

    def process_frame(self, frame: Any) -> Any:
        """streamlit-webrtc의 video_frame_callback입니다."""
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
        points: Any = None
        scale = 1.0

        if should_scan:
            detection_image, scale = self._resize_for_detection(image)
            try:
                # async frame processing에서도 OpenCV detector를 동시에 호출하지 않도록 보호합니다.
                with self._detector_lock:
                    decoded, points, _ = self._detector.detectAndDecode(detection_image)
                decoded = str(decoded or "").strip()
            except Exception:
                decoded = ""
                points = None

            with self._lock:
                if decoded:
                    payload_changed = decoded != self._last_emitted_payload
                    disappeared_long_enough = (
                        self._visible_payload == ""
                        and now - self._last_seen_monotonic >= self._disappear_seconds
                    )

                    self._visible_payload = decoded
                    self._last_seen_monotonic = now

                    if payload_changed or disappeared_long_enough:
                        self._event_sequence += 1
                        token = sha1(
                            f"{decoded}|{self._event_sequence}|{now}".encode("utf-8")
                        ).hexdigest()[:16]
                        self._latest_event = QRDetectionEvent(
                            event_id=self._event_sequence,
                            payload=decoded,
                            detected_at=time.time(),
                            token=token,
                        )
                        self._last_emitted_payload = decoded
                elif now - self._last_seen_monotonic >= self._disappear_seconds:
                    self._visible_payload = ""

        if decoded:
            self._draw_detection(display, points, scale=scale)
            cv2.rectangle(display, (0, 0), (display.shape[1], 44), (25, 130, 85), -1)
            cv2.putText(
                display,
                "QR DETECTED",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.74,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            self._draw_guide(display)

        return av.VideoFrame.from_ndarray(display, format="bgr24")

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
                "stream_active": now - self._last_frame_monotonic < 2.0,
                "visible_payload": self._visible_payload,
                "latest_event_id": (
                    self._latest_event.event_id if self._latest_event else 0
                ),
            }

    def reset_visibility(self) -> None:
        """장소나 입력 방법을 바꿀 때 다음 QR을 새 이벤트로 받을 수 있게 합니다."""
        with self._lock:
            self._visible_payload = ""
            self._last_seen_monotonic = 0.0


def live_qr_available() -> bool:
    return av is not None and webrtc_streamer is not None and WebRtcMode is not None


def _rtc_configuration() -> dict[str, Any]:
    """기본 STUN과 선택적인 TURN 서버 설정을 구성합니다."""
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
    """사용자가 카메라를 시작하면 프레임마다 QR을 자동 판독합니다."""
    if not live_qr_available():
        st.error(
            "실시간 QR 카메라 패키지가 설치되지 않았어요. "
            "`py -m pip install -r requirements.txt` 실행 후 서버를 다시 시작해 주세요."
        )
        return None

    return webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=_rtc_configuration(),
        media_stream_constraints={
            "video": {
                "width": {"ideal": 720},
                "height": {"ideal": 540},
                "facingMode": {"ideal": "environment"},
                "frameRate": {"ideal": 24, "max": 30},
            },
            "audio": False,
        },
        video_frame_callback=scanner.process_frame,
        on_video_ended=scanner.reset_visibility,
        async_processing=True,
        media_toggle_controls=False,
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
            "start": "카메라 시작",
            "stop": "카메라 중지",
            "select_device": "카메라 선택",
            "select_camera": "카메라 선택",
            "device_ask_permission": "카메라 사용 권한을 허용해 주세요.",
            "device_access_denied": "카메라 권한이 차단되어 있어요.",
            "device_not_available": "사용할 수 있는 카메라를 찾지 못했어요.",
            "media_api_not_available": "이 브라우저에서는 카메라를 사용할 수 없어요.",
        },
    )
