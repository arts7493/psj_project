from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


# -----------------------------------------------------------------------------
# 기본 설정
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

SECRETS_PATH = (
    BASE_DIR
    / ".streamlit"
    / "secrets.toml"
)

# 사용자 계정의 오류 메시지에서 권장한 모델
TEST_MODEL = "gemini-3.6-flash"

# Google GenAI SDK의 timeout은 밀리초 단위
TIMEOUT_MS = 20_000


# -----------------------------------------------------------------------------
# 시크릿 읽기
# -----------------------------------------------------------------------------
def read_secrets() -> dict[str, Any]:
    """
    .streamlit/secrets.toml을 읽는다.
    """
    if not SECRETS_PATH.exists():
        raise FileNotFoundError(
            "시크릿 파일을 찾지 못했습니다.\n"
            f"확인 위치: {SECRETS_PATH}"
        )

    with SECRETS_PATH.open("rb") as file:
        return dict(tomllib.load(file))


def get_gemini_api_key(
    secrets: dict[str, Any],
) -> str:
    """
    secrets.toml에서 Gemini 키를 가져오고
    기본 안내문구가 그대로인지 검사한다.
    """
    api_key = str(
        secrets.get(
            "GEMINI_API_KEY",
            "",
        )
    ).strip()

    placeholders = (
        "PUT_YOUR_",
        "YOUR_API_KEY",
        "REPLACE_ME",
        "실제_GEMINI",
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY가 비어 있습니다."
        )

    if any(
        text.upper() in api_key.upper()
        for text in placeholders
    ):
        raise ValueError(
            "GEMINI_API_KEY가 실제 키로 "
            "교체되지 않았습니다."
        )

    return api_key


# -----------------------------------------------------------------------------
# 연결 테스트
# -----------------------------------------------------------------------------
def main() -> int:
    print()
    print("===== WAITGO Gemini 연결 테스트 =====")
    print(f"시크릿 파일 : {SECRETS_PATH}")
    print(f"테스트 모델 : {TEST_MODEL}")
    print(f"사용 API    : Interactions API")
    print(f"제한 시간   : {TIMEOUT_MS // 1000}초")
    print()

    try:
        secrets = read_secrets()
        api_key = get_gemini_api_key(secrets)

    except Exception as exc:
        print("[설정 오류]")
        print(exc)
        return 1

    try:
        from google import genai
        from google.genai import errors, types

    except ImportError:
        print("[패키지 오류]")
        print("google-genai 패키지를 찾지 못했습니다.")
        print()
        print("다음 명령을 실행하세요:")
        print("py -m pip install -U google-genai")
        return 1

    client = None
    started_at = time.perf_counter()

    try:
        print("Gemini에 짧은 메시지를 보내는 중...")

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(
                    attempts=1,
                ),
            ),
        )

        interaction = client.interactions.create(
            model=TEST_MODEL,
            input=(
                "API 연결 확인용 요청입니다. "
                "다른 설명 없이 반드시 "
                "'연결 성공'이라고만 답하세요."
            ),
            store=False,
        )

        elapsed = (
            time.perf_counter()
            - started_at
        )

        answer = str(
            interaction.output_text or ""
        ).strip()

        print()
        print("[성공]")
        print(f"응답 시간 : {elapsed:.2f}초")
        print(f"응답 내용 : {answer or '(빈 응답)'}")

        if not answer:
            print()
            print(
                "API 호출은 성공했지만 "
                "텍스트 응답이 비어 있습니다."
            )
            return 1

        print()
        print(
            "Gemini 키, 모델, Interactions API가 "
            "모두 정상입니다."
        )

        return 0

    except errors.APIError as exc:
        elapsed = (
            time.perf_counter()
            - started_at
        )

        print()
        print("[Gemini API 오류]")
        print(f"경과 시간 : {elapsed:.2f}초")

        print(
            "오류 코드 : "
            f"{getattr(exc, 'code', '확인 불가')}"
        )

        print(
            "오류 내용 : "
            f"{getattr(exc, 'message', str(exc))}"
        )

        return 1

    except Exception as exc:
        elapsed = (
            time.perf_counter()
            - started_at
        )

        print()
        print("[연결 실패]")
        print(f"경과 시간 : {elapsed:.2f}초")
        print(f"오류 종류 : {type(exc).__name__}")
        print(f"오류 내용 : {exc}")

        return 1

    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())