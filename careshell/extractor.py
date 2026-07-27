"""Turn one line of timeline text into a typed CareEvent.

This is the only place a language model runs. It classifies; it never decides. Every
safety consequence is handled downstream by careshell.reconciler, in plain Python.

Two implementations behind one interface:

  LLMExtractor      -- the real one. Talks to the vLLM attached to NemoClaw.
  KeywordExtractor  -- a dependency-free fallback so the pipeline, the tests and a
                       demo can run when inference is unavailable. It is deliberately
                       conservative: it never returns CONFIRMED, so it can never cause a
                       dose to be recorded on its own.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Protocol

import requests

from careshell.schemas import CareEvent, TimelineEntry

# NemoClaw exposes its attached inference to the sandbox as `inference.local` over
# **https**. Plain http is refused by the egress proxy with a 403, so the scheme is not
# cosmetic. Verified against nemoclaw v0.0.93 / openshell 0.0.85:
#   nemoclaw <sandbox> exec -- curl -s https://inference.local/v1/models   -> model list
#   nemoclaw <sandbox> exec -- curl -s  http://inference.local/v1/models   -> 403
DEFAULT_ENDPOINT = os.environ.get(
    "CARESHELL_ENDPOINT", "https://inference.local/v1/chat/completions"
)
# Overridden per deployment; deploy/deploy.sh resolves the real id from /v1/models.
DEFAULT_MODEL = os.environ.get("CARESHELL_MODEL", "nvidia/Nemotron-Nano-12B-v2")

# Enough for one CareEvent object. Reasoning models spend hidden tokens against this
# budget before emitting content, so it cannot be as tight as the JSON alone implies.
MAX_OUTPUT_TOKENS = int(os.environ.get("CARESHELL_MAX_TOKENS", "512"))

# Retrying a 503 or a 429 instantly just collects a second 503. A gateway fronting a
# shared upstream will rate-limit a burst -- feeding 17 timeline entries through in
# 34 seconds is exactly such a burst -- and it needs a moment, not a faster retry.
RETRY_BACKOFF_SECONDS = (0.75, 2.0, 4.0)
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

SYSTEM_PROMPT = """You classify a single eldercare observation into one JSON object.

Rules:
- Output JSON only. No prose, no markdown fence, no explanation.
- med_id MUST be one of the listed medication IDs, or null. Never invent an ID.
- certainty=CONFIRMED only if the text states the patient actually ingested the
  medication (swallowed, took, drank it down). Reaching for, holding, opening, being
  handed, or being reminded about a medication is LIKELY, not CONFIRMED.
- If the observation is not about medication or bed state, use kind=OTHER.
- restlessness is 1 (calm) to 5 (severe), or null if the text says nothing about it.

Respond with exactly this shape:
{"kind":"MED_INTAKE|BED_EXIT|BED_RETURN|SLEEP_DISTURBANCE|OTHER",
 "med_id":"<id or null>",
 "certainty":"CONFIRMED|LIKELY|UNCLEAR",
 "restlessness":<1-5 or null>,
 "summary":"<one short clause>"}"""


class Extractor(Protocol):
    def extract(self, entry: TimelineEntry, med_catalog: list[dict]) -> CareEvent: ...


def _review_event(summary: str) -> CareEvent:
    """Fail-safe result: surfaces on the nurse console, never records a dose.

    `needs_review` is what makes the first half of that true. Without it the event is
    just `kind=OTHER`, which the reconciler has no branch for, so the observation would
    be stored and then silently produce no decision at all.
    """
    return CareEvent(
        kind="OTHER", med_id=None, certainty="UNCLEAR", summary=summary, needs_review=True
    )


def _strip_fence(text: str) -> str:
    """Some instruct models wrap JSON in a markdown fence despite being told not to."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class LLMExtractor:
    """OpenAI-compatible chat client. Points at the vLLM NemoClaw already serves."""

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        timeout: int = 30,
        attempts: int = 3,
        sleeper=time.sleep,
    ):
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.attempts = max(1, attempts)
        self.sleeper = sleeper
        self.session = requests.Session()
        # Negotiated on first use; see _swap_token_param.
        self._token_param = os.environ.get("CARESHELL_TOKEN_PARAM", "max_tokens")
        # The sandbox's inference proxy terminates TLS with its own certificate. Trust
        # it only when explicitly told to -- this is a loopback-equivalent hop inside an
        # egress-denied sandbox, but it should still be a deliberate choice.
        if os.environ.get("CARESHELL_INSECURE_TLS") == "1":
            self.session.verify = False

    def _payload(self, entry: TimelineEntry, med_catalog: list[dict]) -> dict:
        catalog = "\n".join(
            f"- {m['id']}: {m['name']} ({m.get('description', m['name'])})"
            for m in med_catalog
        ) or "- (none on file)"
        user = (
            f"Known medications for this patient:\n{catalog}\n\n"
            f"Observation source: {entry.source}\n"
            f"Observation: {entry.text}"
        )
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            # vLLM's OpenAI server supports this. If your build rejects it, swap to
            # {"guided_json": CareEvent.model_json_schema()} -- same effect, vLLM-specific.
            "response_format": {"type": "json_object"},
            "temperature": 0.0,   # the demo must be reproducible
            self._token_param: MAX_OUTPUT_TOKENS,
        }

    def _swap_token_param(self, response: requests.Response) -> bool:
        """Handle servers that disagree on the output-length parameter name.

        vLLM's OpenAI server takes `max_tokens`. Newer OpenAI-hosted models reject it
        with 400 `unsupported_parameter` and require `max_completion_tokens`. NemoClaw
        can front either one behind `inference.local`, so rather than hardcoding a guess
        per deployment, negotiate once on the first rejection and remember the answer.

        Returns True if the parameter was swapped and the request is worth retrying.
        """
        if response.status_code != 400:
            return False
        try:
            error = response.json().get("error", {})
        except ValueError:
            return False
        if error.get("code") != "unsupported_parameter":
            return False
        if error.get("param") != self._token_param:
            return False

        alternative = (
            "max_completion_tokens" if self._token_param == "max_tokens" else "max_tokens"
        )
        print(f"[extractor] server rejected {self._token_param!r}, using {alternative!r}")
        self._token_param = alternative
        return True

    def _backoff(self, tried: int, attempts_left: int) -> None:
        """Pause before the next attempt. No-op once the budget is spent."""
        if attempts_left <= 0:
            return
        base = RETRY_BACKOFF_SECONDS[min(tried, len(RETRY_BACKOFF_SECONDS) - 1)]
        # Jitter, so a burst of entries failing together does not resynchronise into a
        # second burst against an already-strained gateway.
        self.sleeper(base * (0.7 + random.random() * 0.6))

    def extract(self, entry: TimelineEntry, med_catalog: list[dict]) -> CareEvent:
        valid_ids = {m["id"] for m in med_catalog}
        last_err: Exception | None = None

        # A parameter renegotiation is not a failed attempt, so it gets its own budget
        # and never eats the retries meant for transient errors.
        attempts_left = self.attempts
        swaps_left = 1

        while attempts_left > 0:
            payload = self._payload(entry, med_catalog)
            tried = self.attempts - attempts_left

            try:
                r = self.session.post(self.endpoint, json=payload, timeout=self.timeout)
            except Exception as exc:          # noqa: BLE001 - transport failure
                last_err = exc
                attempts_left -= 1
                self._backoff(tried, attempts_left)
                continue

            if swaps_left and self._swap_token_param(r):
                swaps_left -= 1
                continue          # renegotiation, not a failed attempt

            attempts_left -= 1
            try:
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                event = CareEvent(**json.loads(_strip_fence(content)))
            except Exception as exc:          # noqa: BLE001 - any failure degrades the same way
                last_err = exc
                if r.status_code in RETRYABLE_STATUS:
                    self._backoff(tried, attempts_left)
                continue

            # The model is not trusted to stay inside the catalogue.
            if event.med_id is not None and event.med_id not in valid_ids:
                event.med_id = None
                event.certainty = "UNCLEAR"
                event.summary = f"[unknown medication id dropped] {event.summary}"
            # A medication verdict with no medication cannot be acted on.
            if event.kind == "MED_INTAKE" and event.med_id is None:
                event.certainty = "UNCLEAR"
            return event

        return _review_event(
            f"Extraction failed ({type(last_err).__name__}: {last_err}): {entry.text[:120]}"
        )


class KeywordExtractor:
    """Offline fallback for when inference is unavailable.

    By default it never returns CONFIRMED, so it cannot cause a dose to be recorded --
    pattern matching is not evidence. `allow_confirm=True` relaxes that for an
    ingestion verb paired with a recognised medication, which makes an inference-free
    dry run actually exercise the dose path. That is a demo affordance, not a
    production mode.
    """

    _INTAKE = re.compile(
        r"\b(took|takes|taking|swallow\w*|ingest\w*|pill|tablet|capsule|medication|dose|bottle)\b",
        re.I,
    )
    _INGESTED = re.compile(r"\b(swallow\w*|took|ingest\w*|administered)\b", re.I)
    # Anything here forces LIKELY instead of CONFIRMED. Negations matter as much as
    # hedges: "no medication was administered" contains an ingestion verb but is the
    # opposite of an intake, and reading it as a dose would suppress a missed-dose alert.
    _HEDGED = re.compile(
        r"\b(no|not|never|none|without|denied|declined|refus\w*|unable|"
        r"reach\w*|set it back|asked|offered|handed|remind\w*|almost|nearly|"
        r"about to|attempted|tried)\b",
        re.I,
    )
    _EXIT = re.compile(r"\b(bed exit|out of bed|got up|left the bed|standing)\b", re.I)
    _RETURN = re.compile(r"\b(returned to bed|back in bed|lying down|got into bed)\b", re.I)
    _DISTURBANCE = re.compile(r"\b(restless|tossing|turning|agitated|awake|could not sleep)\b", re.I)

    def __init__(self, allow_confirm: bool = False):
        self.allow_confirm = allow_confirm

    def extract(self, entry: TimelineEntry, med_catalog: list[dict]) -> CareEvent:
        text = entry.text
        if self._RETURN.search(text):
            return CareEvent(kind="BED_RETURN", certainty="LIKELY", summary=text[:120])
        if self._EXIT.search(text):
            return CareEvent(kind="BED_EXIT", certainty="LIKELY", summary=text[:120])
        if self._INTAKE.search(text):
            med_id = self._guess_med(text, med_catalog)
            confirmed = (
                self.allow_confirm
                and med_id is not None
                and self._INGESTED.search(text)
                and not self._HEDGED.search(text)
            )
            return CareEvent(
                kind="MED_INTAKE",
                med_id=med_id,
                certainty="CONFIRMED" if confirmed else "LIKELY",
                summary=text[:120],
            )
        if self._DISTURBANCE.search(text):
            return CareEvent(kind="SLEEP_DISTURBANCE", certainty="LIKELY", summary=text[:120])
        return CareEvent(kind="OTHER", certainty="UNCLEAR", summary=text[:120])

    @staticmethod
    def _guess_med(text: str, med_catalog: list[dict]) -> str | None:
        """Match on words that distinguish one medication from the others.

        Naive first-match-wins is wrong here: "round" and "tablet" appear in every
        description, so "the white statin tablet" would match whichever medication is
        listed first. Words shared by more than one medication carry no signal, so they
        are discarded before matching, and a tie resolves to None (human review) rather
        than a coin flip.
        """
        vocab: dict[str, set[str]] = {}
        for med in med_catalog:
            haystack = f"{med['name']} {med.get('description', '')}".lower()
            vocab[med["id"]] = set(re.findall(r"[a-z]{4,}", haystack))

        shared = {
            word
            for word in set().union(*vocab.values(), set())
            if sum(word in words for words in vocab.values()) > 1
        }

        tokens = set(re.findall(r"[a-z]{4,}", text.lower()))
        scores = {
            med_id: len(tokens & (words - shared)) for med_id, words in vocab.items()
        }
        best = max(scores.values(), default=0)
        if best == 0:
            return None
        winners = [med_id for med_id, score in scores.items() if score == best]
        return winners[0] if len(winners) == 1 else None


def build_extractor(
    offline: bool,
    endpoint: str,
    model: str,
    timeout: int = 30,
    trust_keywords: bool = False,
) -> Extractor:
    if offline:
        return KeywordExtractor(allow_confirm=trust_keywords)
    return LLMExtractor(endpoint, model, timeout)
