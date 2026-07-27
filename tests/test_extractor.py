"""Extractor tests. The LLM is stubbed -- these check our handling, not the model."""

from __future__ import annotations

import json

import pytest
import requests

from careshell.extractor import KeywordExtractor, LLMExtractor
from tests.conftest import entry

CATALOG = [
    {"id": "MED_MORNING", "name": "Beta Blocker", "description": "blue round tablet"},
    {"id": "MED_EVENING", "name": "Evening Statin", "description": "white round tablet"},
]


class FakeResponse:
    def __init__(self, content: str, status: int = 200):
        self._content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0

    def post(self, url, json=None, timeout=None):
        self.calls += 1
        # Repeat the last scripted response once the list is exhausted, so a test
        # written for N attempts still behaves sanely if the retry budget grows.
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


def make(*responses, attempts: int = 2) -> tuple[LLMExtractor, FakeSession]:
    # sleeper is a no-op: these tests exercise retry accounting, not wall-clock backoff.
    ex = LLMExtractor(endpoint="http://stub/v1/chat/completions", model="stub",
                      attempts=attempts, sleeper=lambda _s: None)
    session = FakeSession(*responses)
    ex.session = session
    return ex, session


VALID = json.dumps({
    "kind": "MED_INTAKE", "med_id": "MED_MORNING", "certainty": "CONFIRMED",
    "restlessness": None, "summary": "swallowed a tablet",
})


def test_valid_response_parses(store=None):
    ex, _ = make(FakeResponse(VALID))
    event = ex.extract(entry("e1", "2026-03-14T07:58:00"), CATALOG)
    assert event.kind == "MED_INTAKE"
    assert event.med_id == "MED_MORNING"
    assert event.certainty == "CONFIRMED"


def test_markdown_fence_is_stripped():
    ex, _ = make(FakeResponse(f"```json\n{VALID}\n```"))
    event = ex.extract(entry("e1", "2026-03-14T07:58:00"), CATALOG)
    assert event.med_id == "MED_MORNING"


def test_hallucinated_med_id_is_dropped_and_downgraded():
    bad = json.dumps({
        "kind": "MED_INTAKE", "med_id": "MED_INVENTED", "certainty": "CONFIRMED",
        "restlessness": None, "summary": "took something",
    })
    ex, _ = make(FakeResponse(bad))
    event = ex.extract(entry("e1", "2026-03-14T07:58:00"), CATALOG)
    assert event.med_id is None
    assert event.certainty == "UNCLEAR"      # must not be able to record a dose


def test_confirmed_intake_without_med_id_is_downgraded():
    bad = json.dumps({
        "kind": "MED_INTAKE", "med_id": None, "certainty": "CONFIRMED",
        "restlessness": None, "summary": "took something",
    })
    ex, _ = make(FakeResponse(bad))
    event = ex.extract(entry("e1", "2026-03-14T07:58:00"), CATALOG)
    assert event.certainty == "UNCLEAR"


def test_malformed_json_retries_then_degrades():
    ex, session = make(FakeResponse("not json at all"), FakeResponse("still not json"))
    event = ex.extract(entry("e1", "2026-03-14T07:58:00"), CATALOG)
    assert session.calls == 2
    assert event.kind == "OTHER"
    assert event.certainty == "UNCLEAR"


def test_retry_recovers_from_a_single_bad_response():
    ex, session = make(FakeResponse("garbage"), FakeResponse(VALID))
    event = ex.extract(entry("e1", "2026-03-14T07:58:00"), CATALOG)
    assert session.calls == 2
    assert event.med_id == "MED_MORNING"


def test_network_failure_never_raises():
    ex, _ = make(requests.ConnectionError("refused"), requests.ConnectionError("refused"))
    event = ex.extract(entry("e1", "2026-03-14T07:58:00"), CATALOG)
    assert event.kind == "OTHER"
    assert event.certainty == "UNCLEAR"
    assert "Extraction failed" in event.summary


def test_http_error_never_raises():
    ex, _ = make(FakeResponse("", 503), FakeResponse("", 503))
    event = ex.extract(entry("e1", "2026-03-14T07:58:00"), CATALOG)
    assert event.certainty == "UNCLEAR"


def test_prompt_lists_only_catalogue_ids():
    ex, _ = make(FakeResponse(VALID))
    payload = ex._payload(entry("e1", "2026-03-14T07:58:00", "took a pill"), CATALOG)
    user = payload["messages"][1]["content"]
    assert "MED_MORNING" in user and "MED_EVENING" in user
    assert payload["temperature"] == 0.0     # reproducible demos


# ------------------------------------------------------------------- fallback

@pytest.mark.parametrize("text,expected", [
    ("Bed exit detected.", "BED_EXIT"),
    ("Patient returned to bed.", "BED_RETURN"),
    ("He swallowed one tablet with water.", "MED_INTAKE"),
    ("Restless in bed, tossing and turning.", "SLEEP_DISTURBANCE"),
    ("Blood pressure reading recorded.", "OTHER"),
])
def test_keyword_extractor_classifies(text, expected):
    event = KeywordExtractor().extract(entry("e1", "2026-03-14T07:58:00", text), CATALOG)
    assert event.kind == expected


def test_keyword_matches_on_distinctive_words_not_shared_ones():
    """"round" and "tablet" appear in both descriptions and must carry no signal."""
    ex = KeywordExtractor(allow_confirm=True)
    evening = ex.extract(
        entry("e1", "2026-03-14T20:00:00", "Evening round. Swallowed the white statin tablet."),
        CATALOG,
    )
    morning = ex.extract(
        entry("e2", "2026-03-14T08:00:00", "Swallowed the blue beta blocker tablet."),
        CATALOG,
    )
    assert evening.med_id == "MED_EVENING"
    assert morning.med_id == "MED_MORNING"


def test_keyword_ambiguous_medication_resolves_to_none():
    ex = KeywordExtractor(allow_confirm=True)
    event = ex.extract(entry("e1", "2026-03-14T08:00:00", "Swallowed a round tablet."), CATALOG)
    assert event.med_id is None
    assert event.certainty != "CONFIRMED"


@pytest.mark.parametrize("text", [
    "No medication was administered this morning.",
    "He declined to take the blue beta blocker tablet.",
    "John refused his white statin tablet.",
    "She was unable to get him to swallow the blue beta blocker.",
    "He almost took a second blue beta blocker tablet.",
])
def test_negated_intake_is_never_confirmed(text):
    """"no medication was administered" contains an ingestion verb and is not an intake."""
    ex = KeywordExtractor(allow_confirm=True)
    event = ex.extract(entry("e1", "2026-03-15T09:12:00", text), CATALOG)
    assert event.certainty != "CONFIRMED", text


def test_keyword_extractor_never_confirms():
    """It is pattern matching, not evidence -- it must never record a dose by itself."""
    texts = [
        "He swallowed one tablet with water.",
        "Took the blue beta blocker tablet.",
        "Bed exit detected.",
    ]
    for text in texts:
        event = KeywordExtractor().extract(entry("e1", "2026-03-14T07:58:00", text), CATALOG)
        assert event.certainty != "CONFIRMED"
