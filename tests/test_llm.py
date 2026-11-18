"""LLM client: retries, Retry-After, JSON-mode fallback, rate limiting, cache and the mock provider.

No real network calls and no real sleeping: a fake OpenAI-compatible client and fake clock/sleep are injected.
"""

import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import openai
import pytest

try:  # openai 3.x ships its own HTTP client package; older SDKs use httpx
    import httpx2
except ImportError:  # pragma: no cover - depends on the installed SDK
    import httpx as httpx2

from src import llm
from src.config import MAX_TOKENS
from src.llm import (
    LLMCallError,
    LLMClient,
    MissingAPIKeyError,
    MockChatClient,
    api_key_for,
    retry_after_seconds,
)
from src.parsing import parse_response
from src.prompts import build_messages

REQUEST = httpx2.Request("POST", "https://api.example.test/v1/chat/completions")
MESSAGES = [{"role": "user", "content": "hello"}]


def status_error(cls, status, headers=None):
    return cls(f"HTTP {status}", response=httpx2.Response(status, headers=headers or {}, request=REQUEST), body=None)


def ok_response(text="{}", model="served-model"):
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class FakeClient:
    """Raises the queued exceptions in order, then returns a response; records every call."""

    def __init__(self, failures=()):
        self.failures = list(failures)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        return ok_response()


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(fake, timer=None, **kwargs):
    timer = timer or FakeTime()
    kwargs.setdefault("rpm", 0)
    return LLMClient("groq", "test-model", client=fake, cache_path=None, sleep=timer.sleep, clock=timer.clock,
                     **kwargs), timer


def test_429_then_success_honours_retry_after():
    fake = FakeClient([status_error(openai.RateLimitError, 429, {"retry-after": "7"})])
    client, timer = make_client(fake)
    result = client.complete(MESSAGES, version="V0")
    assert timer.sleeps == [7.0]
    assert result.attempts == 2 and len(fake.calls) == 2
    assert result.returned_model == "served-model" and result.requested_model == "test-model"
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_5xx_and_timeouts_use_exponential_backoff_with_jitter():
    fake = FakeClient([status_error(openai.InternalServerError, 503), openai.APITimeoutError(request=REQUEST)])
    client, timer = make_client(fake)
    client.complete(MESSAGES)
    assert len(timer.sleeps) == 2
    assert 2.0 <= timer.sleeps[0] <= 3.0 and 4.0 <= timer.sleeps[1] <= 5.0


def test_gives_up_after_max_retries():
    fake = FakeClient([status_error(openai.RateLimitError, 429)] * 10)
    client, timer = make_client(fake, max_retries=6)
    with pytest.raises(LLMCallError) as info:
        client.complete(MESSAGES)
    assert info.value.retryable and info.value.status_code == 429
    assert len(fake.calls) == 7 and len(timer.sleeps) == 6


def test_long_retry_after_stops_immediately():
    fake = FakeClient([status_error(openai.RateLimitError, 429, {"retry-after": "86400"})])
    client, timer = make_client(fake)
    with pytest.raises(LLMCallError, match="daily limit"):
        client.complete(MESSAGES)
    assert timer.sleeps == []


def test_non_retryable_error_raises_at_once():
    fake = FakeClient([status_error(openai.AuthenticationError, 401)])
    client, timer = make_client(fake)
    with pytest.raises(LLMCallError) as info:
        client.complete(MESSAGES)
    assert info.value.status_code == 401 and not info.value.retryable
    assert len(fake.calls) == 1 and timer.sleeps == []


def test_json_mode_rejected_retries_once_without_it():
    fake = FakeClient([status_error(openai.BadRequestError, 400)])
    client, _ = make_client(fake)
    result = client.complete(MESSAGES, json_mode=True)
    assert fake.calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in fake.calls[1]
    assert result.json_mode_fallback and result.json_mode_requested and not result.json_mode_used
    assert fake.calls[0]["temperature"] == 0.0 and fake.calls[0]["max_tokens"] == MAX_TOKENS


def test_bad_request_without_json_mode_is_not_retried():
    fake = FakeClient([status_error(openai.BadRequestError, 400)] * 2)
    client, _ = make_client(fake)
    with pytest.raises(LLMCallError):
        client.complete(MESSAGES, json_mode=False)
    assert len(fake.calls) == 1


def test_rate_limiter_spaces_calls():
    fake = FakeClient()
    client, timer = make_client(fake, rpm=30)  # at least 2 s between call starts
    client.complete(MESSAGES)
    client.complete([{"role": "user", "content": "second"}])
    assert timer.sleeps == [pytest.approx(2.0)]


def test_latency_excludes_rate_limit_and_backoff_waits():
    fake = FakeClient([status_error(openai.RateLimitError, 429, {"retry-after": "7"})])
    client, timer = make_client(fake, rpm=30)
    first = client.complete(MESSAGES)  # waits 7 s (Retry-After), which also covers the 2 s rate-limit spacing
    second = client.complete([{"role": "user", "content": "second"}])  # waits 2 s for its slot
    assert first.wait_s == pytest.approx(7.0) and second.wait_s == pytest.approx(2.0)
    assert first.latency_s < 1.0 and second.latency_s < 1.0  # the fake API answers instantly
    assert first.request_key == client.cache_key(MESSAGES, None, 0.0, MAX_TOKENS)
    assert {"latency_s", "wait_s", "request_key"} <= set(first.record())


def test_cache_avoids_repeated_calls(tmp_path):
    cache = tmp_path / "cache.jsonl"
    fake = FakeClient()
    client = LLMClient("groq", "m", client=fake, cache_path=cache, rpm=0)
    first = client.complete(MESSAGES, version="V1")
    second = client.complete(MESSAGES, version="V1")
    assert not first.cached and second.cached and len(fake.calls) == 1
    assert second.text == first.text and second.returned_model == "served-model"
    # A new client instance (e.g. a resumed run) reads the cache file.
    again = LLMClient("groq", "m", client=FakeClient(), cache_path=cache, rpm=0).complete(MESSAGES, version="V1")
    assert again.cached
    # Different version or model -> different key.
    assert client.cache_key(MESSAGES, "V1", 0.0, 400) != client.cache_key(MESSAGES, "V2", 0.0, 400)
    other = LLMClient("groq", "other", client=fake, cache_path=cache, rpm=0)
    assert other.cache_key(MESSAGES, "V1", 0.0, 400) != client.cache_key(MESSAGES, "V1", 0.0, 400)
    lines = cache.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["text"] == "{}"


def test_cache_bypass(tmp_path):
    fake = FakeClient()
    client = LLMClient("groq", "m", client=fake, cache_path=tmp_path / "c.jsonl", use_cache=False, rpm=0)
    client.complete(MESSAGES)
    client.complete(MESSAGES)
    assert len(fake.calls) == 2 and not (tmp_path / "c.jsonl").exists()


def test_missing_key_message(monkeypatch):
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="console.groq.com/keys"):
        api_key_for("groq")
    with pytest.raises(MissingAPIKeyError):
        LLMClient("groq")
    assert api_key_for("mock") == ""


def test_retry_after_http_date():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    header = format_datetime(now + timedelta(seconds=30), usegmt=True)
    exc = status_error(openai.RateLimitError, 429, {"retry-after": header})
    assert retry_after_seconds(exc, now=lambda: now) == pytest.approx(30.0)
    assert retry_after_seconds(ValueError("no response")) is None


def test_unknown_provider():
    with pytest.raises(ValueError):
        LLMClient("nope")
    with pytest.raises(llm.LLMError, match="LLM_PROVIDER"):  # CLIs catch LLMError and exit cleanly
        LLMClient("nope")


def test_mock_provider_returns_schema_valid_json():
    reference = {name: {"human_median": 0.5, "human_q1": 0.4, "human_q3": 0.6, "ai_median": 0.3,
                        "ai_q1": 0.2, "ai_q3": 0.4, "scale": 0.2}
                 for name in ("sentence_length_cv", "mattr", "repeated_trigram_rate", "word_entropy")}
    text = "Short text. Only a few words here."
    client = LLMClient("mock")
    assert client.rpm == 0 and not client.use_cache
    v4 = client.complete(build_messages("V2", text, None, reference), json_mode=True)
    parsed = parse_response(v4.text)
    assert parsed.json_valid and parsed.label == "Uncertain"  # fewer than 100 words
    fenced = client.complete(build_messages("V2", text, None, reference), json_mode=False)
    assert fenced.text.startswith("```json") and parse_response(fenced.text).json_valid
    assert client.list_models() == ["mock-evidence-v1"]


def test_mock_can_emit_malformed_json():
    client = LLMClient("mock", client=MockChatClient(malformed_rate=1.0), rpm=0)
    result = client.complete(build_messages("V0", "Some text. More text here."))
    assert not parse_response(result.text).json_valid


# ---- reasoning effort (gpt-oss reasoning models) --------------------------------------------------------
def test_reasoning_effort_sent_for_gpt_oss_only(monkeypatch):
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    timer = FakeTime()
    fake = FakeClient()
    gpt_oss = LLMClient("groq", "openai/gpt-oss-120b", client=fake, cache_path=None, rpm=0,
                        sleep=timer.sleep, clock=timer.clock)
    result = gpt_oss.complete([{"role": "user", "content": "hi"}], version="V0")
    assert fake.calls[-1]["reasoning_effort"] == "low" and result.reasoning_effort == "low"

    other = LLMClient("groq", "qwen/qwen3.8-27b", client=fake, cache_path=None, rpm=0,
                      sleep=timer.sleep, clock=timer.clock)
    other.complete([{"role": "user", "content": "hi"}], version="V0")
    assert "reasoning_effort" not in fake.calls[-1]


def test_reasoning_effort_override_and_cache_key(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "medium")
    fake = FakeClient()
    medium = LLMClient("groq", "openai/gpt-oss-120b", client=fake, cache_path=None, rpm=0)
    assert medium.reasoning_effort == "medium"
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    off = LLMClient("groq", "openai/gpt-oss-120b", client=fake, cache_path=None, rpm=0)
    assert off.reasoning_effort is None
    explicit = LLMClient("groq", "openai/gpt-oss-120b", client=fake, cache_path=None, rpm=0, reasoning_effort="high")
    assert explicit.reasoning_effort == "high"
    messages = [{"role": "user", "content": "hi"}]
    assert medium.cache_key(messages, "V0", 0.0, 400) != off.cache_key(messages, "V0", 0.0, 400)
