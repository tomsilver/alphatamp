"""Model backends — a thin factory over ``prpl_llm_utils``, plus one cache-key fix.

``prpl_llm_utils`` (prpl-mono) is the lab-standard LLM layer and already provides
everything the baseline needs: the ``PretrainedLargeModel`` abstraction, OpenAI/Gemini
subclasses, PIL image attachment, disk caching keyed by query + model id, and
multi-response support. We depend on it directly rather than writing backend code.

**Local open-weight mode needs no new backend either**: ``OpenAIModel`` constructs a bare
``openai.OpenAI()``, which honours ``OPENAI_BASE_URL``, so pointing it at a local
OpenAI-compatible server (LM Studio, vLLM) is an environment change. Switching between
the local dev arm and a frontier API arm is therefore config only.

The one thing we do add is :class:`_StripNonApiHyperparametersMixin` — see below.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from prpl_llm_utils.cache import (
    FilePretrainedLargeModelCache,
    PretrainedLargeModelCache,
    SQLite3PretrainedLargeModelCache,
)
from prpl_llm_utils.models import (
    GeminiModel,
    OpenAIModel,
    OpenAIResponsesModel,
    PretrainedLargeModel,
)
from prpl_llm_utils.structs import Query, Response

# Hyperparameters that exist only to discriminate the cache key, never to be sent to the
# API. ``prpl_llm_utils`` hashes images *perceptually* (512x512 LANCZOS +
# phash/dhash/whash/colorhash), so two DD2D scenes that differ by a few pixels can share
# a key and silently serve each other's plans. Mixing ``problem_id`` into the key makes
# it exact. ``round`` does the same for the repeat-suppression block, whose growing plan
# list is already in the prompt but which we key explicitly for legibility.
CACHE_ONLY_KEYS = frozenset({"problem_id", "round", "run_tag"})


def _api_safe(query: Query) -> Query:
    """``query`` with cache-only hyperparameters removed.

    ``OpenAIModel`` splats ``query.hyperparameters`` straight into
    ``chat.completions.create``, so an unknown key is a ``TypeError``. Stripping here —
    below the caching layer, which has already keyed on the full dict — keeps the key
    exact while the request stays valid.
    """
    hyperparameters = query.hyperparameters or {}
    kept = {k: v for k, v in hyperparameters.items() if k not in CACHE_ONLY_KEYS}
    if len(kept) == len(hyperparameters):
        return query
    return Query(prompt=query.prompt, imgs=query.imgs, hyperparameters=kept)


class _StripNonApiHyperparametersMixin:
    """Mixin: key the cache on everything, send only what the API understands."""

    def _run_query(self, query: Query) -> Response:
        return super()._run_query(_api_safe(query))  # type: ignore[misc]

    def _run_query_multi_response(
        self, query: Query, num_responses: int
    ) -> list[Response]:
        return super()._run_query_multi_response(  # type: ignore[misc]
            _api_safe(query), num_responses
        )


class CachedOpenAIModel(_StripNonApiHyperparametersMixin, OpenAIModel):
    """``OpenAIModel`` (chat completions) with exact cache keys."""


class CachedOpenAIResponsesModel(
    _StripNonApiHyperparametersMixin, OpenAIResponsesModel
):
    """``OpenAIResponsesModel`` with exact cache keys."""


class CachedGeminiModel(_StripNonApiHyperparametersMixin, GeminiModel):
    """``GeminiModel`` with exact cache keys."""


_BACKENDS = {
    "openai": CachedOpenAIModel,
    "openai_responses": CachedOpenAIResponsesModel,
    "gemini": CachedGeminiModel,
}


@dataclass(frozen=True)
class ModelConfig:
    """Everything that identifies a run's model, recorded into every cache record.

    ``decode`` is passed through to the backend verbatim.

    **Temperature defaults to 1.0, not 0.0.** This loop asks for *diverse* plans across
    several rounds, and at temperature 0 the only thing varying between rounds is the
    repeat-suppression block — so a round that yields nothing accepted leaves the next
    round with a byte-identical prompt and a near-identical completion. KinDER's own runs
    also use temperature 1. `loop.py` additionally passes a per-round ``seed``, so runs
    stay reproducible against a backend that honours it.
    """

    backend: str = "openai"
    model_name: str = "qwen3-vl-8b-instruct"
    cache_path: str = "data/spectre/derived/dd2d_v2/vlmplan_cache.db"
    cache_kind: str = "sqlite"
    decode: Mapping[str, Any] = field(
        default_factory=lambda: {"temperature": 1.0, "max_tokens": 4096}
    )
    base_url: str | None = None

    def describe(self) -> dict[str, Any]:
        """Self-describing provenance for the run's records."""
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "decode": dict(self.decode),
            "base_url": self.base_url or os.environ.get("OPENAI_BASE_URL"),
        }


def make_cache(config: ModelConfig) -> PretrainedLargeModelCache:
    """Build the response cache.

    SQLite by default: the file cache writes one directory per query, which at hundreds
    of problems x several rounds is a lot of inodes for no benefit. Note the SQLite cache
    asserts a *constant* hyperparameter key set for its lifetime, so callers must always
    pass the same keys — see ``loop.py``.
    """
    path = Path(config.cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if config.cache_kind == "file":
        path.mkdir(parents=True, exist_ok=True)
        return FilePretrainedLargeModelCache(path)
    return SQLite3PretrainedLargeModelCache(path)


def make_model(config: ModelConfig) -> PretrainedLargeModel:
    """Construct the backend named by ``config``.

    For a local server, set ``base_url`` (or ``OPENAI_BASE_URL``) and any non-empty
    ``OPENAI_API_KEY`` — the backends assert the key exists but a local server ignores
    its value.

    **An unset endpoint is an error, not a default.** The OpenAI SDK falls back to
    ``api.openai.com``, so forgetting the export does not fail — it silently sends every
    prompt of a 100-problem run to a paid public endpoint under whatever key happens to
    be in the environment. That happened on 2026-08-01 (5 requests, all rejected 401,
    nothing processed) and the only reason it was caught is that no valid key was set.
    A run that *did* have one would have completed and billed. Refuse instead, and name
    the fix.
    """
    if config.backend not in _BACKENDS:
        raise ValueError(
            f"Unknown backend {config.backend!r}; expected one of {sorted(_BACKENDS)}"
        )
    if config.base_url:
        os.environ["OPENAI_BASE_URL"] = config.base_url
    if config.backend.startswith("openai"):
        allow_remote = os.environ.get("SPECTRE_VLMPLAN_ALLOW_REMOTE") == "1"
        if not os.environ.get("OPENAI_BASE_URL") and not allow_remote:
            raise RuntimeError(
                "No OpenAI-compatible endpoint configured, and the SDK would fall back "
                "to api.openai.com — sending every prompt off-box and billing for it. "
                "Set model.base_url in the config, or export OPENAI_BASE_URL "
                "(e.g. http://localhost:1234/v1 for LM Studio). Pass "
                "SPECTRE_VLMPLAN_ALLOW_REMOTE=1 if a hosted endpoint really is intended."
            )
        os.environ.setdefault("OPENAI_API_KEY", "local-server")
    model_cls = _BACKENDS[config.backend]
    return model_cls(config.model_name, make_cache(config))
