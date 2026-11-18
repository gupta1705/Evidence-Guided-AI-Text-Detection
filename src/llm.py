"""LLM provider client: Groq, Gemini and OpenAI through their OpenAI-compatible endpoints, plus a mock.

* Configuration comes from the environment or ``.env``: ``LLM_PROVIDER`` (default groq), ``LLM_MODEL``,
  ``LLM_RPM`` (default 20), ``LLM_REASONING_EFFORT`` (optional) and the provider key (``GROQ_API_KEY`` /
  ``GEMINI_API_KEY`` / ``OPENAI_API_KEY``).
* Reasoning models (gpt-oss) are sent ``reasoning_effort`` (default "low") because their hidden reasoning
  counts against ``max_tokens`` and the provider's token quota; the value is part of the cache key.
  Keys are never printed or logged.
* Rate limiting: at least ``60 / RPM`` seconds between the starts of two API calls.
* Retries: up to 6 retries with exponential backoff and jitter on HTTP 429, 5xx, timeouts and connection
  errors, honouring ``Retry-After`` when the provider sends it. If JSON mode is rejected (HTTP 400), the
  call is repeated once without it and this is recorded.
* Cache: evaluation responses are stored in ``results/llm_cache.jsonl`` keyed by
  sha256(provider, model, version, messages, temperature, max_tokens), so an interrupted run resumes without
  repeating calls. The app disables the cache (nothing is stored), and the offline mock is never cached.
* The ``mock`` provider is a deterministic offline responder used by the tests and dry runs. It reads the
  feature block of the prompt and returns schema-valid JSON (or malformed JSON on request).

Usage::

    python -m src.llm --list-models [--provider groq]
    python -m src.llm --ping [--provider gemini --model gemini-2.5-flash]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from types import SimpleNamespace

import openai
from dotenv import load_dotenv

from src.config import (
    DEFAULT_RPM,
    ENV_PATH,
    LABEL_AI,
    LABEL_HUMAN,
    LABEL_UNCERTAIN,
    LLM_CACHE_PATH,
    MAX_RETRIES,
    MAX_TOKENS,
    MIN_WORDS_DECISION,
    PROVIDERS,
    SEED,
    TEMPERATURE,
    reasoning_effort_for,
)
from src.reporting import append_jsonl, ensure_utf8_stdout, read_jsonl

BACKOFF_BASE = 2.0
BACKOFF_CAP = 60.0
MAX_RETRY_AFTER = 300.0  # a longer Retry-After (e.g. a daily limit) stops the run instead of waiting


class LLMError(RuntimeError):
    """Base class for LLM client errors."""


class MissingAPIKeyError(LLMError):
    """The provider's API key is not configured."""


class UnknownProviderError(LLMError, ValueError):
    """The provider name (e.g. from ``LLM_PROVIDER``) is not one of ``PROVIDERS``."""


class LLMCallError(LLMError):
    """A call failed permanently (non-retryable error or retries exhausted)."""

    def __init__(self, message: str, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


@dataclass
class LLMResult:
    """One completed call (or cache hit) with everything needed for the run log.

    ``latency_s`` is the duration of the successful API request only; ``wait_s`` is the time this client
    spent waiting before or between attempts (rate limiting, backoff, ``Retry-After``). ``request_key`` is
    the sha256 cache key of (provider, model, version, messages, temperature, max_tokens), so a logged call
    can be tied to the exact prompt that produced it.
    """

    text: str
    provider: str
    requested_model: str
    returned_model: str | None
    version: str | None
    temperature: float
    max_tokens: int
    json_mode_requested: bool
    json_mode_used: bool
    json_mode_fallback: bool
    latency_s: float
    wait_s: float
    usage: dict | None
    attempts: int
    cached: bool
    timestamp: str
    request_key: str
    reasoning_effort: str | None = None

    def record(self) -> dict:
        """Plain-dict view for ``llm_runs.jsonl``."""
        return asdict(self)


# --------------------------------------------------------------------------------------------------
# Settings and real clients
# --------------------------------------------------------------------------------------------------
def load_settings(env_path: Path = ENV_PATH) -> dict:
    """Read ``LLM_PROVIDER``, ``LLM_MODEL``, ``LLM_RPM`` and ``LLM_REASONING_EFFORT`` (``.env`` not overriding)."""
    load_dotenv(env_path, override=False)
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower() or "groq"
    model = os.getenv("LLM_MODEL", "").strip() or None
    try:
        rpm = int(os.getenv("LLM_RPM", str(DEFAULT_RPM)))
    except ValueError:
        rpm = DEFAULT_RPM
    effort = os.getenv("LLM_REASONING_EFFORT")
    return {"provider": provider, "model": model, "rpm": rpm,
            "reasoning_effort": effort if effort is not None and effort.strip() else None}


def check_provider(provider: str) -> str:
    """Return ``provider`` if it is supported, else raise :class:`UnknownProviderError`."""
    if provider not in PROVIDERS:
        raise UnknownProviderError(f"unknown provider {provider!r} (check LLM_PROVIDER / --provider); "
                                   f"choose one of {sorted(PROVIDERS)}")
    return provider


def api_key_for(provider: str) -> str:
    """Return the provider key from the environment or raise :class:`MissingAPIKeyError`."""
    info = PROVIDERS[check_provider(provider)]
    if info.key_env is None:
        return ""
    load_dotenv(ENV_PATH, override=False)
    key = os.getenv(info.key_env, "").strip()
    if not key:
        raise MissingAPIKeyError(f"{info.key_env} is not set: {info.key_help}.")
    return key


def make_openai_client(provider: str, timeout: float = 60.0):
    """OpenAI SDK client for ``provider`` (SDK retries disabled; this module retries itself)."""
    info = PROVIDERS[provider]
    return openai.OpenAI(api_key=api_key_for(provider), base_url=info.base_url, max_retries=0, timeout=timeout)


# --------------------------------------------------------------------------------------------------
# Mock provider
# --------------------------------------------------------------------------------------------------
def mock_assessment(prompt: str) -> dict:
    """Deterministic assessment computed from the prompt's feature block (no model involved).

    Rules: fewer than 100 words or no reference ranges -> Uncertain; otherwise each feature votes for the
    closer reference median and a label is given only when at least three features agree.
    """
    from src.prompts import parse_feature_block

    block = parse_feature_block(prompt)
    values, reference = block["values"], block["reference"]
    word_count = int(values.get("word_count") or 0)
    limitation = "These statistics do not prove authorship; manual review is recommended."
    if reference is None:
        return {
            "label": LABEL_UNCERTAIN,
            "evidence": [f"word_count is {word_count}.", "No human/AI reference ranges were supplied."],
            "limitations": [limitation],
        }
    votes = {"AI": [], "human": []}
    for name, ref in reference.items():
        x = values.get(name)
        if x is None:
            continue
        gap_h, gap_a = abs(x - ref["human_median"]), abs(x - ref["ai_median"])
        if gap_a < gap_h:
            votes["AI"].append(f"{name} is {x:.4f}, closer to the AI reference median {ref['ai_median']:.4f}.")
        elif gap_h < gap_a:
            votes["human"].append(f"{name} is {x:.4f}, closer to the human reference median {ref['human_median']:.4f}.")
    evidence = (votes["AI"] + votes["human"])[:3]
    if word_count < MIN_WORDS_DECISION:
        label = LABEL_UNCERTAIN
        evidence = [f"word_count is {word_count}, below the {MIN_WORDS_DECISION}-word minimum."] + evidence[:2]
    elif len(votes["AI"]) >= 3:
        label = LABEL_AI
    elif len(votes["human"]) >= 3:
        label = LABEL_HUMAN
    else:
        label = LABEL_UNCERTAIN
    if len(evidence) < 2:
        evidence.append(f"word_count is {word_count}.")
    return {"label": label, "evidence": evidence, "limitations": [limitation]}


def _mock_text_reply(prompt: str) -> str:
    """Plain-text reply for non-assessment prompts (synthetic-data generation in tests)."""
    from src.perturb import style_change
    from src.prompts import BEGIN_DELIMITER, END_DELIMITER

    start, end = prompt.find(BEGIN_DELIMITER), prompt.rfind(END_DELIMITER)
    if start >= 0 and end > start:
        return style_change(prompt[start + len(BEGIN_DELIMITER):end].strip(), "expand")
    return (
        "This is an offline mock answer used for testing the pipeline. It does not come from a language model "
        "and it only exists so that the synthetic-data code path can run without an API key or network access."
    )


class _MockCompletions:
    def __init__(self, owner: "MockChatClient"):
        self._owner = owner

    def create(self, *, model: str, messages: list[dict], response_format: dict | None = None, **_: object):
        """Mimic ``client.chat.completions.create`` of the OpenAI SDK."""
        owner = self._owner
        owner.calls += 1
        prompt = messages[-1]["content"] if messages else ""
        if "COMPUTED FEATURES" in prompt:
            digest = zlib.crc32(json.dumps(messages, sort_keys=True).encode("utf-8"))
            if owner.malformed_rate and (digest % 10_000) / 10_000 < owner.malformed_rate:
                content = '{"label": "Likely AI-generated", "evidence": ["unterminated'
            else:
                content = json.dumps(mock_assessment(prompt), ensure_ascii=False)
                if response_format is None:
                    content = f"```json\n{content}\n```"
        else:
            content = _mock_text_reply(prompt)
        n_in = sum(len(m.get("content", "")) for m in messages) // 4
        n_out = len(content) // 4
        return SimpleNamespace(
            model=model,
            choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=n_in, completion_tokens=n_out, total_tokens=n_in + n_out),
        )


class MockChatClient:
    """Offline stand-in for the OpenAI SDK client.

    Args:
        malformed_rate: Share of assessment calls (chosen by a stable hash of the messages) that return
            malformed JSON; 1.0 makes every assessment malformed.
    """

    def __init__(self, malformed_rate: float = 0.0):
        self.malformed_rate = float(malformed_rate)
        self.calls = 0
        self.chat = SimpleNamespace(completions=_MockCompletions(self))
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id=PROVIDERS["mock"].default_model)]))


# --------------------------------------------------------------------------------------------------
# Error classification
# --------------------------------------------------------------------------------------------------
def status_code_of(exc: BaseException) -> int | None:
    """HTTP status code carried by an SDK (or fake) exception, if any."""
    code = getattr(exc, "status_code", None)
    if code is None:
        code = getattr(getattr(exc, "response", None), "status_code", None)
    return int(code) if isinstance(code, int) else None


def retry_after_seconds(exc: BaseException, now: Callable[[], datetime] | None = None) -> float | None:
    """Seconds requested by a ``Retry-After`` header (number or HTTP date), or None."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        pass
    try:
        when = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, IndexError):
        return None
    current = now() if now else datetime.now(timezone.utc)
    return max(0.0, (when - current).total_seconds())


def is_retryable(exc: BaseException, status: int | None) -> bool:
    """429, 5xx, timeouts and connection problems are retried."""
    if status is not None:
        return status == 429 or status >= 500
    return isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError, TimeoutError, ConnectionError))


# --------------------------------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------------------------------
class LLMClient:
    """Rate-limited, retrying, caching chat client for one provider and model.

    Args:
        provider: "groq", "gemini", "openai" or "mock".
        model: Model id (provider default if None).
        rpm: Maximum requests per minute (<= 0 disables the limiter; ignored by the built-in mock).
        cache_path: JSONL cache file (None disables caching).
        use_cache: Set False to bypass the cache entirely (the app does this).
        client: Pre-built OpenAI-compatible client (tests inject fakes here).
        sleep / clock: Injected for tests so that no real time passes.
        max_retries: Retries after the first attempt for retryable errors.
        timeout: Per-request timeout in seconds for real providers.
        mock_malformed_rate: Passed to :class:`MockChatClient` for the mock provider.
        reasoning_effort: "auto" (default) = ``LLM_REASONING_EFFORT`` if set, else "low" for gpt-oss models and
            nothing for other models; or an explicit value ("low"/"medium"/"high"; None/"none" = omit).
    """

    def __init__(
        self,
        provider: str = "groq",
        model: str | None = None,
        *,
        rpm: int = DEFAULT_RPM,
        cache_path: Path | None = LLM_CACHE_PATH,
        use_cache: bool = True,
        client=None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        max_retries: int = MAX_RETRIES,
        timeout: float = 60.0,
        mock_malformed_rate: float = 0.0,
        seed: int = SEED,
        reasoning_effort: str | None = "auto",
    ):
        self.provider = check_provider(provider)
        self.model = model or PROVIDERS[provider].default_model
        if reasoning_effort == "auto":
            override = os.getenv("LLM_REASONING_EFFORT")
            override = override if override is not None and override.strip() else None
            self.reasoning_effort = reasoning_effort_for(self.model, override)
        else:
            self.reasoning_effort = reasoning_effort_for(self.model, reasoning_effort or "none")
        # The offline mock has no quota, so it is never rate-limited unless a fake client is injected.
        self.rpm = 0 if provider == "mock" and client is None else rpm
        self.sleep = sleep
        self.clock = clock
        self.max_retries = max_retries
        self._rng = random.Random(seed)
        self._last_call: float | None = None
        if client is None:
            client = MockChatClient(mock_malformed_rate) if provider == "mock" else make_openai_client(provider, timeout)
        self.client = client
        self.use_cache = bool(use_cache and cache_path is not None and provider != "mock")
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self._cache: dict[str, dict] | None = None

    # ---- cache -----------------------------------------------------------------------------------
    def cache_key(self, messages: list[dict], version: str | None, temperature: float, max_tokens: int) -> str:
        """sha256 over provider, model, version, messages, temperature and max_tokens."""
        payload = {
            "provider": self.provider,
            "model": self.model,
            "version": version,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.reasoning_effort is not None:  # only added when used, so other models keep their keys
            payload["reasoning_effort"] = self.reasoning_effort
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict[str, dict]:
        if self._cache is None:
            self._cache = {r["key"]: r for r in read_jsonl(self.cache_path) if "key" in r} if self.cache_path else {}
        return self._cache

    # ---- rate limiting and retries ---------------------------------------------------------------
    def _wait_for_slot(self) -> float:
        """Sleep so that consecutive call starts are at least ``60 / rpm`` seconds apart; return the wait."""
        wait = 0.0
        if self.rpm and self.rpm > 0 and self._last_call is not None:
            wait = max(0.0, self._last_call + 60.0 / self.rpm - self.clock())
            if wait > 0:
                self.sleep(wait)
        self._last_call = self.clock()
        return wait

    def _backoff(self, attempt: int) -> float:
        return min(BACKOFF_CAP, BACKOFF_BASE * 2 ** attempt) + self._rng.uniform(0, 1)

    def _create_with_retries(self, kwargs: dict) -> tuple[object, int, bool, float, float]:
        """Call the API with rate limiting, retries and the JSON-mode fallback.

        Returns:
            ``(response, attempts, json_mode_fallback, latency_s, wait_s)``: ``latency_s`` times only the
            successful request; ``wait_s`` adds up the rate-limit, backoff and ``Retry-After`` sleeps.
        """
        attempt = 0
        calls = 0
        json_fallback = False
        waited = 0.0
        while True:
            waited += self._wait_for_slot()
            calls += 1
            start = time.perf_counter()
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response, calls, json_fallback, time.perf_counter() - start, waited
            except Exception as exc:  # noqa: BLE001 - classified below, re-raised as LLMCallError
                status = status_code_of(exc)
                if status == 400 and "response_format" in kwargs and not json_fallback:
                    kwargs = {k: v for k, v in kwargs.items() if k != "response_format"}
                    json_fallback = True
                    continue
                if not is_retryable(exc, status):
                    raise LLMCallError(f"{self.provider} call failed ({status or type(exc).__name__}): {exc}",
                                       status, retryable=False) from exc
                retry_after = retry_after_seconds(exc)
                if retry_after is not None and retry_after > MAX_RETRY_AFTER:
                    raise LLMCallError(
                        f"{self.provider} asked to wait {retry_after:.0f}s (probably a daily limit); completed "
                        "calls are cached - rerun later to resume", status, retryable=True) from exc
                if attempt >= self.max_retries:
                    raise LLMCallError(f"{self.provider} call failed after {self.max_retries} retries "
                                       f"({status or type(exc).__name__})", status, retryable=True) from exc
                pause = retry_after if retry_after is not None else self._backoff(attempt)
                self.sleep(pause)
                waited += pause
                attempt += 1

    # ---- public API ------------------------------------------------------------------------------
    def complete(
        self,
        messages: list[dict],
        *,
        version: str | None = None,
        json_mode: bool = False,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ) -> LLMResult:
        """Send ``messages`` and return the response text with call metadata (cache first)."""
        key = self.cache_key(messages, version, temperature, max_tokens)
        if self.use_cache:
            hit = self._load_cache().get(key)
            if hit is not None:
                return LLMResult(
                    text=hit.get("text", ""),
                    provider=self.provider,
                    requested_model=self.model,
                    returned_model=hit.get("returned_model"),
                    version=version,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode_requested=json_mode,
                    json_mode_used=bool(hit.get("json_mode_used", False)),
                    json_mode_fallback=bool(hit.get("json_mode_fallback", False)),
                    latency_s=0.0,
                    wait_s=0.0,
                    usage=hit.get("usage"),
                    attempts=0,
                    cached=True,
                    timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    request_key=key,
                    reasoning_effort=self.reasoning_effort,
                )

        kwargs: dict = {"model": self.model, "messages": messages, "temperature": temperature,
                        "max_tokens": max_tokens}
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response, attempts, fallback, latency, waited = self._create_with_retries(kwargs)
        choice = response.choices[0] if getattr(response, "choices", None) else None
        text = (getattr(getattr(choice, "message", None), "content", None) or "") if choice else ""
        usage_obj = getattr(response, "usage", None)
        usage = (
            {k: getattr(usage_obj, k, None) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
            if usage_obj is not None else None
        )
        result = LLMResult(
            text=text,
            provider=self.provider,
            requested_model=self.model,
            returned_model=getattr(response, "model", None),
            version=version,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode_requested=json_mode,
            json_mode_used=json_mode and not fallback,
            json_mode_fallback=fallback,
            latency_s=round(latency, 3),
            wait_s=round(waited, 3),
            usage=usage,
            attempts=attempts,
            cached=False,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            request_key=key,
            reasoning_effort=self.reasoning_effort,
        )
        if self.use_cache:
            entry = {"key": key, "provider": self.provider, "model": self.model, "version": version,
                     "text": text, "returned_model": result.returned_model, "usage": usage,
                     "json_mode_used": result.json_mode_used, "json_mode_fallback": fallback,
                     "reasoning_effort": self.reasoning_effort, "timestamp": result.timestamp}
            self._load_cache()[key] = entry
            append_jsonl(self.cache_path, [entry])
        return result

    def list_models(self) -> list[str]:
        """Model ids offered by the provider."""
        page = self.client.models.list()
        data = getattr(page, "data", page)
        return sorted(getattr(m, "id", str(m)) for m in data)


def main(argv: list[str] | None = None) -> int:
    """Command-line helpers: list models or send a tiny ping."""
    ensure_utf8_stdout()
    settings = load_settings()
    parser = argparse.ArgumentParser(prog="python -m src.llm", description="Provider helpers (list models / ping).")
    parser.add_argument("--provider", default=settings["provider"], choices=sorted(PROVIDERS),
                        help="provider (default: LLM_PROVIDER or groq)")
    parser.add_argument("--model", default=settings["model"], help="model id (default: LLM_MODEL or provider default)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list-models", action="store_true", help="print the models the provider offers")
    group.add_argument("--ping", action="store_true", help="send a one-line test prompt and print the reply")
    parser.add_argument("--limit", type=int, default=None, help="with --list-models: print at most N models")
    args = parser.parse_args(argv)
    try:
        client = LLMClient(args.provider, args.model, use_cache=False, cache_path=None)
        if args.list_models:
            for model_id in client.list_models()[:args.limit]:
                print(model_id)
        else:
            result = client.complete([{"role": "user", "content": "Reply with the single word: pong"}],
                                     version="ping", max_tokens=5)
            print(f"provider={result.provider} requested={result.requested_model} "
                  f"returned={result.returned_model} latency={result.latency_s}s reply={result.text!r}")
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
