from __future__ import annotations

import os
from pathlib import Path

import pytest

from reptrace.models.ModelWrapper import GeminiWrapper, OpenAIWrapper


class _SecretValue:
    """Holds secrets without exposing value in traceback repr."""

    def __init__(self, value: str):
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - defensive
        return "<secret>"


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _get_env_key(names: list[str], env_file_values: dict[str, str]) -> str | None:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    for name in names:
        value = env_file_values.get(name, "").strip()
        if value:
            return value
    return None


def _run_one_question_openai(api_key: _SecretValue) -> str:
    model_candidates = [
        os.getenv("REPTRACE_OPENAI_TEST_MODEL", "").strip() or "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo",
    ]
    prompt = "Reply with one short token: ACCESS_OK"
    last_error: str | None = None

    for model_name in model_candidates:
        try:
            wrapper = OpenAIWrapper(model_name=model_name, api_key=api_key.value)
            response = wrapper.generate(
                prompt,
                max_length=16,
                temperature=0.0,
                do_sample=False,
                top_p=1.0,
            )
            if response and response.strip():
                return response.strip()
            last_error = f"empty response for model={model_name}"
        except Exception as exc:
            last_error = str(exc)

    raise AssertionError(f"OpenAI access failed for all test models. Last error: {last_error}")


def _run_one_question_gemini(api_key: _SecretValue) -> str:
    model_candidates = [
        os.getenv("REPTRACE_GEMINI_TEST_MODEL", "").strip() or "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    prompt = "Reply with one short token: ACCESS_OK"
    last_error: str | None = None

    for model_name in model_candidates:
        try:
            wrapper = GeminiWrapper(model_name=model_name, api_key=api_key.value)
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": wrapper._build_gemini_generation_config(
                    max_length=16,
                    temperature=0.0,
                    do_sample=False,
                    top_p=1.0,
                ),
            }
            response_obj = wrapper._http_json(
                method="POST",
                url=f"{wrapper.api_base}/models/{model_name}:generateContent",
                payload=payload,
            )
            response = wrapper._extract_gemini_text(response_obj)
            if response and response.strip():
                return response.strip()
            last_error = f"empty response for model={model_name}"
        except Exception as exc:
            last_error = str(exc)

    raise AssertionError(f"Gemini access failed for all test models. Last error: {last_error}")


@pytest.mark.slow
def test_real_openai_api_access_for_one_question():
    if os.getenv("REPTRACE_RUN_REAL_API_TESTS", "").strip() != "1":
        pytest.skip("Set REPTRACE_RUN_REAL_API_TESTS=1 to run real provider access tests.")

    env_file_values = _load_env_file(Path(__file__).resolve().parents[1] / ".env")
    openai_key = _get_env_key(["OPENAI_API_KEY", "APIKEY_OPENAI"], env_file_values)
    if not openai_key:
        pytest.skip("No OpenAI API key found in environment or .env.")

    openai_response = _run_one_question_openai(_SecretValue(openai_key))
    assert openai_response


@pytest.mark.slow
def test_real_gemini_api_access_for_one_question():
    if os.getenv("REPTRACE_RUN_REAL_API_TESTS", "").strip() != "1":
        pytest.skip("Set REPTRACE_RUN_REAL_API_TESTS=1 to run real provider access tests.")

    env_file_values = _load_env_file(Path(__file__).resolve().parents[1] / ".env")
    gemini_key = _get_env_key(
        ["GEMINI_API_KEY", "GOOGLE_API_KEY", "APIKEY_GOOGLE"], env_file_values
    )
    if not gemini_key:
        pytest.skip("No Gemini API key found in environment or .env.")

    gemini_response = _run_one_question_gemini(_SecretValue(gemini_key))
    assert gemini_response
