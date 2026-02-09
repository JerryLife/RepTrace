from __future__ import annotations

import io
import importlib
import json
import logging
import sys
from types import SimpleNamespace

import pytest

from reptrace.models.ModelLoader import ModelLoader
from reptrace.models.ModelWrapper import GeminiWrapper, OpenAIWrapper, OpenRouterWrapper


class _FakeEncoding:
    n_vocab = 1000

    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join(chr(token) for token in token_ids)


def _install_fake_tiktoken(monkeypatch):
    module = SimpleNamespace(
        encoding_for_model=lambda _model: _FakeEncoding(),
        get_encoding=lambda _name: _FakeEncoding(),
    )
    monkeypatch.setitem(sys.modules, "tiktoken", module)


class _FakeOpenAIClient:
    def __init__(self, fail_batch_create: bool = False, never_complete: bool = False):
        self.fail_batch_create = fail_batch_create
        self.never_complete = never_complete
        self.uploaded_files: dict[str, bytes] = {}
        self.output_files: dict[str, bytes] = {}
        self.batch_states: dict[str, list[str]] = {}
        self.batch_outputs: dict[str, str] = {}
        self.file_counter = 0
        self.batch_counter = 0
        self.retrieve_calls = 0
        self.batch_create_calls = 0
        self.batch_cancel_calls = 0

        self.files = SimpleNamespace(create=self._files_create, content=self._files_content)
        self.batches = SimpleNamespace(
            create=self._batches_create,
            retrieve=self._batches_retrieve,
            cancel=self._batches_cancel,
        )
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._chat_create),
        )

    def _files_create(self, file, purpose):
        assert purpose == "batch"
        content = file.read()
        if isinstance(content, str):
            content = content.encode("utf-8")
        file_id = f"file-{self.file_counter}"
        self.file_counter += 1
        self.uploaded_files[file_id] = content
        return SimpleNamespace(id=file_id)

    def _files_content(self, file_id):
        payload = self.output_files[file_id]
        return io.BytesIO(payload)

    def _batches_create(self, input_file_id, endpoint, completion_window):
        del completion_window
        assert endpoint == "/v1/chat/completions"
        self.batch_create_calls += 1
        if self.fail_batch_create:
            raise RuntimeError("batch disabled")

        request_lines = self.uploaded_files[input_file_id].decode("utf-8").splitlines()
        records = [json.loads(line) for line in request_lines if line.strip()]
        output_lines = []
        for record in reversed(records):
            prompt = record["body"]["messages"][0]["content"]
            output_lines.append(
                json.dumps(
                    {
                        "custom_id": record["custom_id"],
                        "response": {
                            "body": {
                                "choices": [
                                    {"message": {"content": f"batch:{prompt}"}},
                                ]
                            }
                        },
                    }
                )
            )

        output_id = f"output-{self.batch_counter}"
        self.output_files[output_id] = ("\n".join(output_lines)).encode("utf-8")

        batch_id = f"batch-{self.batch_counter}"
        self.batch_counter += 1
        if self.never_complete:
            self.batch_states[batch_id] = ["validating"] + (["in_progress"] * 1000)
        else:
            self.batch_states[batch_id] = ["validating", "in_progress", "completed"]
        self.batch_outputs[batch_id] = output_id
        return SimpleNamespace(id=batch_id, status="validating")

    def _batches_retrieve(self, batch_id):
        self.retrieve_calls += 1
        states = self.batch_states[batch_id]
        status = states.pop(0) if states else "completed"
        if status in {"completed", "done"}:
            return SimpleNamespace(
                id=batch_id,
                status=status,
                output_file_id=self.batch_outputs[batch_id],
            )
        return SimpleNamespace(id=batch_id, status=status)

    def _batches_cancel(self, batch_id):
        self.batch_cancel_calls += 1
        self.batch_states[batch_id] = ["canceled"]
        return SimpleNamespace(id=batch_id, status="canceled")

    def _chat_create(self, model, messages, max_tokens, temperature, top_p, **kwargs):
        del model, max_tokens, temperature, top_p, kwargs
        prompt = messages[0]["content"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f"seq:{prompt}"),
                )
            ]
        )


def _install_fake_openai(monkeypatch, client: _FakeOpenAIClient):
    module = SimpleNamespace(OpenAI=lambda api_key=None, **kwargs: client)
    monkeypatch.setitem(sys.modules, "openai", module)


def test_openai_wrapper_generate_batch_uses_batch_api(monkeypatch):
    _install_fake_tiktoken(monkeypatch)
    client = _FakeOpenAIClient(fail_batch_create=False)
    _install_fake_openai(monkeypatch, client)

    wrapper = OpenAIWrapper(model_name="gpt-4o", api_key="test-key")
    outputs = wrapper.generate_batch(
        ["prompt-0", "prompt-1", "prompt-2"],
        max_length=64,
        temperature=0.0,
        do_sample=False,
        top_p=1.0,
        batch_poll_interval_seconds=0.01,
    )

    assert outputs == ["batch:prompt-0", "batch:prompt-1", "batch:prompt-2"]
    assert client.retrieve_calls >= 2
    assert client.batch_counter == 1


def test_openai_wrapper_batch_falls_back_to_sequential(monkeypatch):
    _install_fake_tiktoken(monkeypatch)
    client = _FakeOpenAIClient(fail_batch_create=True)
    _install_fake_openai(monkeypatch, client)

    wrapper = OpenAIWrapper(model_name="gpt-4o", api_key="test-key")
    outputs = wrapper.generate_batch(
        ["alpha", "beta"],
        max_length=64,
        temperature=0.0,
        do_sample=False,
        top_p=1.0,
        batch_poll_interval_seconds=0.01,
    )

    assert outputs == ["seq:alpha", "seq:beta"]


def test_openai_wrapper_explicit_non_batch_path(monkeypatch):
    _install_fake_tiktoken(monkeypatch)
    client = _FakeOpenAIClient(fail_batch_create=False)
    _install_fake_openai(monkeypatch, client)

    wrapper = OpenAIWrapper(model_name="gpt-4o", api_key="test-key")
    outputs = wrapper.generate_batch(
        ["nobatch-a", "nobatch-b"],
        max_length=64,
        temperature=0.0,
        do_sample=False,
        top_p=1.0,
        prefer_batch_api=False,
    )

    assert outputs == ["seq:nobatch-a", "seq:nobatch-b"]
    assert client.batch_create_calls == 0
    assert client.retrieve_calls == 0


def test_openai_wrapper_batch_timeout_falls_back_to_sequential(monkeypatch):
    _install_fake_tiktoken(monkeypatch)
    client = _FakeOpenAIClient(fail_batch_create=False, never_complete=True)
    _install_fake_openai(monkeypatch, client)

    wrapper = OpenAIWrapper(model_name="gpt-4o", api_key="test-key")
    outputs = wrapper.generate_batch(
        ["timeout-a", "timeout-b"],
        max_length=64,
        temperature=0.0,
        do_sample=False,
        top_p=1.0,
        batch_poll_interval_seconds=0.01,
        batch_timeout_seconds=0.0,
    )

    assert outputs == ["seq:timeout-a", "seq:timeout-b"]
    assert client.batch_create_calls == 1
    assert client.batch_cancel_calls == 1


def test_openai_wrapper_error_response_parsing():
    wrapper = OpenAIWrapper.__new__(OpenAIWrapper)
    wrapper.logger = logging.getLogger("test-openai-wrapper")
    raw = "\n".join(
        [
            json.dumps(
                {
                    "custom_id": "prompt_0",
                    "response": {
                        "body": json.dumps(
                            {
                                "choices": [
                                    {"message": {"content": "ok-from-json-string-body"}}
                                ]
                            }
                        )
                    },
                }
            ),
            json.dumps(
                {
                    "custom_id": "prompt_1",
                    "response": {"error": {"message": "rate limited"}},
                }
            ),
            "not-json-line",
        ]
    ).encode("utf-8")

    parsed = OpenAIWrapper._parse_openai_batch_output(wrapper, raw)
    assert parsed[0] == "ok-from-json-string-body"
    assert parsed[1] == ""


def test_openrouter_wrapper_generate_batch_uses_batch_api(monkeypatch):
    _install_fake_tiktoken(monkeypatch)
    client = _FakeOpenAIClient(fail_batch_create=False)
    _install_fake_openai(monkeypatch, client)

    wrapper = OpenRouterWrapper(model_name="openai/gpt-3.5-turbo", api_key="test-key")
    outputs = wrapper.generate_batch(
        ["prompt-0", "prompt-1", "prompt-2"],
        max_length=64,
        temperature=0.0,
        do_sample=False,
        top_p=1.0,
        batch_poll_interval_seconds=0.01,
    )

    assert outputs == ["batch:prompt-0", "batch:prompt-1", "batch:prompt-2"]
    assert client.retrieve_calls >= 2
    assert client.batch_counter == 1


def test_openrouter_wrapper_batch_callback_does_not_break_payload(monkeypatch):
    _install_fake_tiktoken(monkeypatch)
    client = _FakeOpenAIClient(fail_batch_create=False)
    _install_fake_openai(monkeypatch, client)

    callback_events = []

    wrapper = OpenRouterWrapper(model_name="openai/gpt-3.5-turbo", api_key="test-key")
    outputs = wrapper.generate_batch(
        ["prompt-0", "prompt-1"],
        max_length=64,
        temperature=0.0,
        do_sample=False,
        top_p=1.0,
        batch_poll_interval_seconds=0.01,
        on_response_callback=lambda idx, prompt, response: callback_events.append((idx, prompt, response)),
        show_progress=False,
    )

    assert outputs == ["batch:prompt-0", "batch:prompt-1"]
    assert callback_events == [
        (0, "prompt-0", "batch:prompt-0"),
        (1, "prompt-1", "batch:prompt-1"),
    ]
    assert client.batch_counter == 1


def test_openrouter_wrapper_batch_falls_back_to_sequential(monkeypatch):
    _install_fake_tiktoken(monkeypatch)
    client = _FakeOpenAIClient(fail_batch_create=True)
    _install_fake_openai(monkeypatch, client)

    wrapper = OpenRouterWrapper(model_name="openai/gpt-3.5-turbo", api_key="test-key")
    outputs = wrapper.generate_batch(
        ["alpha", "beta"],
        max_length=64,
        temperature=0.0,
        do_sample=False,
        top_p=1.0,
        batch_poll_interval_seconds=0.01,
    )

    assert outputs == ["seq:alpha", "seq:beta"]


def test_openrouter_wrapper_generate_uses_reasoning_fallback(monkeypatch):
    _install_fake_tiktoken(monkeypatch)
    client = _FakeOpenAIClient(fail_batch_create=False)
    _install_fake_openai(monkeypatch, client)

    wrapper = OpenRouterWrapper(model_name="openrouter/pony-alpha", api_key="test-key")

    class _ReasoningMessage:
        content = ""

        @staticmethod
        def model_dump():
            return {"reasoning": "ACCESS_OK"}

    def _reasoning_create(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=_ReasoningMessage(),
                    finish_reason="stop",
                )
            ]
        )

    monkeypatch.setattr(wrapper.client.chat.completions, "create", _reasoning_create)
    assert wrapper.generate("test prompt", max_length=16, temperature=0.0, do_sample=False) == "ACCESS_OK"


def test_openrouter_wrapper_generate_error_returns_empty(monkeypatch):
    _install_fake_tiktoken(monkeypatch)
    client = _FakeOpenAIClient(fail_batch_create=False)
    _install_fake_openai(monkeypatch, client)

    wrapper = OpenRouterWrapper(model_name="openai/gpt-3.5-turbo", api_key="test-key")

    def _raise_error(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("connection failed")

    monkeypatch.setattr(wrapper.client.chat.completions, "create", _raise_error)
    assert wrapper.generate("test prompt") == ""


def test_openrouter_wrapper_get_logits_not_supported(monkeypatch):
    _install_fake_tiktoken(monkeypatch)
    client = _FakeOpenAIClient(fail_batch_create=False)
    _install_fake_openai(monkeypatch, client)

    wrapper = OpenRouterWrapper(model_name="openai/gpt-3.5-turbo", api_key="test-key")
    with pytest.raises(NotImplementedError, match="OpenRouter API models don't provide logits access"):
        wrapper.get_logits("hello")


def test_openrouter_wrapper_tokenizer_fallback_without_tiktoken(monkeypatch):
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    client = _FakeOpenAIClient(fail_batch_create=False)
    _install_fake_openai(monkeypatch, client)

    wrapper = OpenRouterWrapper(model_name="openai/gpt-3.5-turbo", api_key="test-key")

    assert wrapper.tokenizer is None
    assert wrapper.tokenize("Az") == [65, 122]
    assert wrapper.detokenize([1, 2, 3]) == "1 2 3"
    assert wrapper.get_vocab_size() == 0


def test_gemini_wrapper_generate_batch_chunks_and_merges(monkeypatch):
    _install_fake_tiktoken(monkeypatch)
    wrapper = GeminiWrapper(model_name="gemini-2.0-flash", api_key="test-key")

    submitted: dict[str, list[dict]] = {}

    def fake_submit(requests):
        name = f"batches/{len(submitted)}"
        submitted[name] = requests
        return name

    def fake_wait(batch_name, poll_interval_seconds, timeout_seconds):
        del poll_interval_seconds, timeout_seconds
        requests = submitted[batch_name]
        inlined = []
        for item in requests:
            key = item["metadata"]["key"]
            prompt = item["request"]["contents"][0]["parts"][0]["text"]
            inlined.append(
                {
                    "metadata": {"key": key},
                    "response": {
                        "candidates": [
                            {"content": {"parts": [{"text": f"gemini:{prompt}"}]}},
                        ]
                    },
                }
            )
        return {"done": True, "response": {"inlinedResponses": {"inlinedResponses": inlined}}}

    monkeypatch.setattr(wrapper, "_submit_gemini_batch", fake_submit)
    monkeypatch.setattr(wrapper, "_wait_gemini_batch", fake_wait)

    outputs = wrapper.generate_batch(
        ["p0", "p1", "p2", "p3", "p4"],
        max_length=64,
        temperature=0.0,
        do_sample=False,
        top_p=1.0,
        batch_max_requests=2,
        batch_poll_interval_seconds=0.01,
    )

    assert outputs == ["gemini:p0", "gemini:p1", "gemini:p2", "gemini:p3", "gemini:p4"]
    assert len(submitted) == 3


def test_gemini_wrapper_batch_timeout_falls_back_to_sequential(monkeypatch):
    _install_fake_tiktoken(monkeypatch)
    wrapper = GeminiWrapper(model_name="gemini-2.0-flash", api_key="test-key")

    monkeypatch.setattr(wrapper, "_submit_gemini_batch", lambda _reqs: "batches/timeout")
    monkeypatch.setattr(
        wrapper,
        "_wait_gemini_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("timed out")),
    )
    monkeypatch.setattr(
        wrapper,
        "generate",
        lambda prompt, **kwargs: f"seq:{prompt}",
    )

    outputs = wrapper.generate_batch(
        ["ga", "gb"],
        max_length=64,
        temperature=0.0,
        do_sample=False,
        top_p=1.0,
        batch_poll_interval_seconds=0.01,
        batch_timeout_seconds=0.0,
    )

    assert outputs == ["seq:ga", "seq:gb"]


def test_model_loader_auto_detects_and_loads_gemini(monkeypatch):
    class _FakeGeminiWrapper:
        def __init__(self, model_name, api_key=None, **kwargs):
            self.model_name = model_name
            self.api_key = api_key
            self.kwargs = kwargs

    model_loader_module = importlib.import_module("reptrace.models.ModelLoader")
    monkeypatch.setattr(model_loader_module, "GeminiWrapper", _FakeGeminiWrapper)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")

    loader = ModelLoader()
    assert loader._detect_model_type("gemini-2.0-flash") == "gemini"

    model = loader.load_model(
        "gemini-2.0-flash",
        model_type="auto",
        trust_remote_code=True,
        load_in_8bit=True,
        batch_max_requests=128,
    )
    assert isinstance(model, _FakeGeminiWrapper)
    assert model.model_name == "gemini-2.0-flash"
    assert model.api_key == "gemini-test-key"
    assert model.kwargs == {"batch_max_requests": 128}


def test_model_loader_auto_detects_and_loads_openrouter(monkeypatch):
    class _FakeOpenRouterWrapper:
        def __init__(self, model_name, api_key=None, **kwargs):
            self.model_name = model_name
            self.api_key = api_key
            self.kwargs = kwargs

    model_loader_module = importlib.import_module("reptrace.models.ModelLoader")
    monkeypatch.setattr(model_loader_module, "OpenRouterWrapper", _FakeOpenRouterWrapper)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")

    loader = ModelLoader()
    assert loader._detect_model_type("openrouter/pony-alpha") == "openrouter"

    model = loader.load_model(
        "openrouter/pony-alpha",
        model_type="auto",
        batch_max_requests=64,
    )
    assert isinstance(model, _FakeOpenRouterWrapper)
    assert model.model_name == "openrouter/pony-alpha"
    assert model.api_key == "openrouter-test-key"
    assert model.kwargs == {"batch_max_requests": 64}


def test_openai_wrapper_single_generate(monkeypatch):
    """Test single prompt generation without batch API."""
    _install_fake_tiktoken(monkeypatch)
    client = _FakeOpenAIClient(fail_batch_create=False)
    _install_fake_openai(monkeypatch, client)

    wrapper = OpenAIWrapper(model_name="gpt-4o", api_key="test-key")
    output = wrapper.generate("hello world", max_length=64, temperature=0.5)

    assert output == "seq:hello world"
    assert client.batch_counter == 0  # No batch API used


def test_model_loader_detects_provider_model_patterns(monkeypatch):
    """Verify auto-detection covers OpenAI, OpenRouter, and Gemini patterns."""
    from reptrace.models.ModelLoader import ModelLoader

    loader = ModelLoader()
    assert loader._detect_model_type("gpt-4o-mini") == "openai"
    assert loader._detect_model_type("o1-preview") == "openai"
    assert loader._detect_model_type("o3-mini") == "openai"
    assert loader._detect_model_type("GPT-4-turbo-2024") == "openai"  # Case insensitive
    assert loader._detect_model_type("openrouter/pony-alpha") == "openrouter"
    assert loader._detect_model_type("openai/gpt-4o-mini") == "openrouter"
    assert loader._detect_model_type("anthropic/claude-3.5-sonnet") == "openrouter"
    assert loader._detect_model_type("google/gemini-2.0-flash") == "openrouter"
    assert loader._detect_model_type("z-ai/glm-4.7") == "openrouter"
    assert loader._detect_model_type("deepseek/deepseek-v3.2") == "openrouter"
    assert loader._detect_model_type("gemini-2.0-flash") == "gemini"
    assert loader._detect_model_type("Gemini-1.5-pro") == "gemini"  # Case insensitive
    assert loader._detect_model_type("meta-llama/Llama-3-8B") == "huggingface"  # Default



def test_model_loader_supports_legacy_env_key_aliases(monkeypatch):
    class _FakeOpenAIWrapper:
        def __init__(self, model_name, api_key=None, **kwargs):
            self.model_name = model_name
            self.api_key = api_key
            self.kwargs = kwargs

    class _FakeGeminiWrapper:
        def __init__(self, model_name, api_key=None, **kwargs):
            self.model_name = model_name
            self.api_key = api_key
            self.kwargs = kwargs

    class _FakeOpenRouterWrapper:
        def __init__(self, model_name, api_key=None, **kwargs):
            self.model_name = model_name
            self.api_key = api_key
            self.kwargs = kwargs

    model_loader_module = importlib.import_module("reptrace.models.ModelLoader")
    monkeypatch.setattr(model_loader_module, "OpenAIWrapper", _FakeOpenAIWrapper)
    monkeypatch.setattr(model_loader_module, "GeminiWrapper", _FakeGeminiWrapper)
    monkeypatch.setattr(model_loader_module, "OpenRouterWrapper", _FakeOpenRouterWrapper)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)
    monkeypatch.setenv("APIKEY_OPENAI", "openai-legacy-key")
    monkeypatch.setenv("APIKEY_GOOGLE", "gemini-legacy-key")
    monkeypatch.setenv("APIKEY_OPENROUTER", "openrouter-legacy-key")

    loader = ModelLoader()
    openai_model = loader.load_model("gpt-4o", model_type="openai")
    gemini_model = loader.load_model("gemini-2.0-flash", model_type="gemini")
    openrouter_model = loader.load_model("openrouter/pony-alpha", model_type="openrouter")

    assert openai_model.api_key == "openai-legacy-key"
    assert gemini_model.api_key == "gemini-legacy-key"
    assert openrouter_model.api_key == "openrouter-legacy-key"
