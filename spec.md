# CareShell: Technical Specification (`spec.md`)

**System Name:** CareShell
**Target Environment:** Dell Pro Max, NVIDIA Grace Blackwell GB10
**Runtime:** NemoClaw (OpenClaw agent) inside NVIDIA OpenShell, vLLM already attached to NemoClaw
**Core Frameworks:** NemoClaw / OpenShell, vLLM (OpenAI-compatible, text-only), Pydantic, SQLite, FastAPI
**Primary Input:** Per-patient event timeline — timestamped text lines (`data/patients/<patient_id>.jsonl`)

This document is the design record. It describes decisions and their reasons; it does not
duplicate the implementation. Code lives in the repository and is the source of truth —
see [`README.md`](README.md) for how to run it.

---

## 0. What Changed From v1

v1 was a computer-vision system: video file → OpenCV motion trigger → Qwen-2.5-VL frame
analysis. That is gone.

**Input is now a per-patient timeline of text.** Something upstream — bedside sensors, a
nurse's tablet, a device bridge — already turned the world into sentences. CareShell reads
those sentences and does the care reasoning.

| Removed | Why |
| --- | --- |
| OpenCV, frame buffer, motion trigger tiers | No pixels enter the system |
| VLM (Qwen-2.5-VL), base64 image payloads | Text model only, served by the vLLM already attached to NemoClaw |
| `sample_video.mp4`, frame-count loop | The timeline file replaces it |
| GB10 VLM feasibility risk | No sm_121 vision kernels, no NIM allowlist problem, no ~35 s/image |
| "Zero **video** leaves the device" | Restated: zero **PHI text** leaves the device |

Dropping the VLM removed the single largest execution risk in the original plan. Any
instruct text model vLLM is already serving works.

What survives, and carries the actual product value: schedule reconciliation, double-dose
interception, **missed-dose detection**, night-time behaviour anomalies, and the local
nurse console.

---

## 1. Architecture

### 1.1 The load-bearing decision: the LLM interprets, Python decides

- **LLM's job** (`careshell/extractor.py`): turn one line of timeline text into a typed
  `CareEvent`.
- **Python's job** (`careshell/reconciler.py`): everything with a safety consequence.
  Lockout windows, dose caps, missed-dose timers, alert escalation. Deterministic,
  unit-testable, no model in the path.

**Why:** a double-dose lockout a language model can talk itself out of is not a safety
control. Constraining the model to classification also makes the demo reproducible — the
same timeline produces the same alerts every run.

Two consequences that fall out of this rule:

1. **Only `certainty=CONFIRMED` records a dose.** `LIKELY` and `UNCLEAR` become
   `NEEDS_HUMAN_REVIEW` and never touch dose state. "Reached for the bottle" is not "took
   the pill" — v1 conflated them behind an inferred `SWALLOWING` stage.
2. **The timeline's own `ts` is the clock.** Nothing calls `datetime.now()` during a
   replay. v1 used wall-clock time, so a demo run at 3pm would never match an 08:00
   medication window and the night-time rule could not fire.

### 1.2 Data flow

```
+------------------------------------------------------------------------------+
| INPUT: data/patients/P104.jsonl                                              |
|   {id, ts, patient_id, source, text} -- one JSON object per line             |
|   careshell/timeline.py  (batch)   careshell/stream.py  (paced / live tail)  |
+------------------------------------------------------------------------------+
                                    |
                                    v
+------------------------------------------------------------------------------+
| EXTRACTOR (careshell/extractor.py)                                           |
|   LLMExtractor -> vLLM at inference.local, OpenAI-compatible /v1             |
|   Prompt carries the patient's med IDs; model picks from that enum or null   |
|   Out: CareEvent (Pydantic-validated, one retry, degrades to human review)   |
+------------------------------------------------------------------------------+
                                    |
                                    v
+------------------------------------------------------------------------------+
| RECONCILER (careshell/reconciler.py)   <-- deterministic, no LLM             |
|   care_plan.yaml: windows, max daily doses, per-medication lockout           |
|   SQLite history, idempotent on entry id                                     |
|   Emits: MED_ON_TIME | MED_OUT_OF_WINDOW | DOUBLE_DOSE_BLOCKED               |
|          MAX_DAILY_DOSES_BLOCKED | MISSED_DOSE | NIGHT_OUT_OF_BED            |
|          NEEDS_HUMAN_REVIEW                                                  |
+------------------------------------------------------------------------------+
                                    |
                +-------------------+-------------------+
                v                   v                   v
        workspace/            workspace/           alerting/
        careshell.db          MEMORY.md            TTS + nurse console :8000
        (authoritative)       (human narrative)
```

```
+------------------------------------------------------------------------------+
| OPENSHELL SANDBOX (managed by NemoClaw)                                      |
|   Landlock: filesystem    seccomp: syscalls    <- kernel-enforced            |
|   Egress: userspace CONNECT proxy + OPA        <- NOT a kernel packet filter |
|   Allowed: inference.local (vLLM), loopback, LAN ingress :8000               |
+------------------------------------------------------------------------------+
```

---

## 2. Repository Layout

```
careshell/
├── README.md                     how to run it
├── spec.md                       this document
├── requirements.txt
├── pytest.ini
├── Dockerfile                    optional --network=none variant
├── data/patients/P104.jsonl      demo timeline
├── data/live/                    live feed target (gitignored)
├── openshell/
│   └── careshell-preset.yaml     POLICY DELTA, never a full replacement
├── deploy/
│   ├── deploy.sh                 policy + deps + console
│   ├── preflight.sh              8 checks, all from inside the sandbox
│   └── demo.sh                   streaming demo
├── tools/
│   └── feed_timeline.py          push a source file into a live feed (stdlib only)
├── careshell/
│   ├── schemas.py                TimelineEntry, CareEvent, Decision
│   ├── care_plan.py              YAML loader + validation
│   ├── timeline.py               JSONL reader, dedupe, sort
│   ├── stream.py                 paced replay, live tail, checkpoint merge
│   ├── extractor.py              LLMExtractor + KeywordExtractor fallback
│   ├── reconciler.py             the safety logic
│   ├── store.py                  SQLite history
│   └── run.py                    CLI
├── alerting/                     tts_engine.py, lan_broadcaster.py
├── dashboard/                    app.py, templates/index.html
└── tests/                        93 tests
```

---

## 3. Input Contract

One JSON object per line, ordered by `ts`. This is the entire contract with whatever
produces the timeline.

```jsonl
{"id":"e005","ts":"2026-03-14T07:58:22","patient_id":"P104","source":"nurse_note","text":"John opened the blue bottle and swallowed one beta blocker tablet with a glass of water."}
```

- `id` — **required and stable.** Reconciliation is idempotent on this.
- `ts` — ISO 8601, facility-local. **This is the system clock.**
- `source` — `sensor` | `nurse_note` | `device`. Informs how far the text is trusted.
- `text` — free-form English, one observation.

Entries are sorted on load. Duplicate ids and malformed lines are reported and skipped:
one bad line from an upstream sensor must not take the bedside loop offline.

### 3.1 Streaming

Three ingest modes, all through the same pipeline (`careshell/run.py`):

| Mode | Flag | Use |
| --- | --- | --- |
| batch | *(default)* | tests, CI — as fast as possible |
| stream | `--stream --speed N` | paced replay of a complete file, for a self-contained demo |
| follow | `--follow PATH` | tail a file something else is appending to — live ingest |

Stream mode does not just sleep between entries. The reconciler is time-driven as well as
event-driven, so `careshell/stream.py` merges entry timestamps with medication
window-close moments into one ordered walk. A `MISSED_DOSE` then fires at 08:30 of virtual
time rather than being lumped into whatever observation happens next. Events sort before
bare ticks at the same instant, so a dose taken exactly at window close counts as taken.

For the demo, `tools/feed_timeline.py` appends a source timeline into the live feed file at
a configurable rate while CareShell tails it from inside the sandbox. Both see the same
file through the rw bind mount in the policy delta.

---

## 4. Component Decisions

Implementation is in the repository. This section records *why*, not *what*.

### 4.1 Care plan is YAML, not markdown prose

In v1 the schedule lived in `PATIENT_SCHEDULE.md` as prose that **no code ever opened** —
the windows and `Max_Daily_Doses` were decorative. `workspace/care_plan.yaml` is
machine-read and validated at startup by `careshell/care_plan.py`, which fails loudly on a
malformed plan rather than silently degrading a safety rule at 3am.

`MEMORY.md` stays markdown, because that one is genuinely for humans.

One sharp edge the loader absorbs: YAML 1.1 parses an unquoted `08:00` as sexagesimal —
the integer 480. Times are coerced back to `HH:MM`, and the shipped plan quotes them.

### 4.2 History is SQLite

v1 recovered dose state by string-splitting markdown log lines
(`line.split("]")[0]`) — fragile and unqueryable. `careshell/store.py` uses SQLite:
stdlib, no server, `UNIQUE` constraints give replay idempotency for free, and the nurse
console gets a real queryable history. Three tables: `observations` (what we have
interpreted), `doses` (what we believe happened), `decisions` (what we emitted).

Timestamps are ISO 8601 strings — SQLite has no datetime type, and ISO 8601 compares
correctly as text.

### 4.3 Reconciler behaviours v1 got wrong

- **Lockout is per medication.** v1 used one global 4-hour timer, so two different drugs
  an hour apart would falsely block each other.
- **`max_daily_doses` is enforced.** v1 read it from nowhere.
- **Missed doses fire.** A window closing empty is the highest-value eldercare signal and
  v1 had no code path that could produce it. `_closed_windows` runs on every clock
  advance, backfills across multi-day gaps, and is capped at 31 days so a stale timeline
  cannot emit thousands of retroactive alerts.
- **A cold start does not backfill.** The first event of a run establishes the baseline.
- **Success messages name the real medication.** v1 hardcoded `"Morning medication
  recorded"` and said it for evening doses too.
- **Repeated "still out of bed" lines do not restart the night timer.** Otherwise a long
  absence never crosses the threshold.
- **Flush stops at end-of-day, not `+24h`.** Advancing past midnight would fabricate
  missed-dose alerts for a day the timeline says nothing about. Silence after the last
  observation is missing data, not a missed dose.

### 4.4 Everything downstream fails soft

- Extractor: network error, HTTP error, or unparseable JSON → one retry → then a
  `NEEDS_HUMAN_REVIEW` item. It never raises, never halts the timeline, never silently
  drops an observation.
- The model is not trusted to stay inside the medication catalogue: an invented `med_id`
  is dropped and the event downgraded to `UNCLEAR`.
- TTS: no speech engine (the normal case in a headless sandbox) → stdout.
- LAN broadcaster: console down → warn once, continue. SQLite already holds the
  authoritative history the console re-reads on reconnect.

### 4.5 The offline extractor is a fallback, not a classifier

`KeywordExtractor` exists so the pipeline, the tests, and a demo can run with no inference
available. It refuses to return `CONFIRMED` unless `--trust-keywords` is passed, and even
then requires an ingestion verb plus a recognised medication with no hedge or negation
present. It matches on words that *distinguish* one medication from another — "round" and
"tablet" appear in every description and carry no signal.

---

## 5. Deployment on GB10 (NemoClaw + OpenShell + existing vLLM)

Assumed in place: NemoClaw installed on the Dell Pro Max GB10, with a vLLM attached and
exposed inside the sandbox as `inference.local`.

```bash
./deploy/deploy.sh careshell      # policy delta, deps, nurse console
./deploy/preflight.sh careshell   # 8 checks, all from inside the sandbox
./deploy/demo.sh careshell 240    # streaming demo
```

**Both NemoClaw and OpenShell are Apache-2.0 and explicitly alpha — verify CLI flag names
against your installed version before demo day.**

### Step 1 — verify vLLM from *inside* the sandbox

```bash
nemoclaw careshell exec -- curl -s http://inference.local/v1/models
```

Never test from the host: the host can reach things the sandbox cannot. The returned `id`
is what `--model` needs. `deploy.sh` resolves it automatically and writes it to
`.careshell-model`.

### Step 2 — apply a policy *delta*

```bash
nemoclaw careshell policy-add --from-file openshell/careshell-preset.yaml
nemoclaw careshell policy-show | grep inference.local
```

`policy-add` layers onto NemoClaw's baseline. **Never `openshell policy set`** — that
replaces the baseline and silently strips `inference.local`, the gateway dial-back
WebSocket, and writable `/dev/pts`, which breaks the exec tool. The failure is invisible
until the agent tries to use them, so `deploy.sh` greps `policy-show` and aborts if
`inference.local` disappeared.

### Step 3 — state the egress guarantee accurately

OpenShell enforces egress with a **userspace HTTP CONNECT proxy plus OPA**, not a kernel
packet filter. There is no `egress: deny-all` key; denial is by *omission* from the
allowlist. The kernel-enforced parts of the sandbox are **Landlock** (filesystem) and
**seccomp** (syscalls).

Say it that way in the demo, because what appears on screen is:

```
curl: (56) Received HTTP code 403 from proxy after CONNECT
```

— visibly a proxy refusal. Claiming a "kernel-level network lock" and then showing a 403
from a proxy is the kind of thing a judge notices.

For a genuine kernel-level guarantee on the process handling PHI, the `Dockerfile`
supports `docker run --network=none --cap-drop=ALL`. Trade-off: `--network=none` also
blocks `inference.local`, so that mode runs the offline extractor. It proves the air gap;
it does not run the model.

---

## 6. Verification Protocols

`deploy/preflight.sh` automates checks 1–8 below against a live sandbox.

1. **Safety logic first.** `pytest tests/test_reconciler.py` — pure functions over
   fixtures, no GPU, model, or network. If only one thing works on demo day, make it this.
2. **Full suite.** `PYTHONPATH=. pytest` — 93 tests, all offline.
3. **Demo timeline.** `python -m careshell.run data/patients/P104.jsonl --offline
   --trust-keywords --console '' --no-tts` must surface `MED_ON_TIME`,
   `DOUBLE_DOSE_BLOCKED`, `NIGHT_OUT_OF_BED`, and `MISSED_DOSE`.
4. **Double-dose intercept.** Exactly one `DOSE` row for `MED_MORNING` on 2026-03-14,
   and the spoken line names the real medication and the real elapsed minutes.
5. **Idempotency.** Re-run the same timeline against the same database: zero new
   decisions, zero new doses.
6. **Uncertainty guard.** `"reached for the blue bottle, then set it back down"` must
   yield `NEEDS_HUMAN_REVIEW` and write no dose.
7. **Inference outage.** Point `--endpoint` at a dead port. The run completes, every line
   becomes a review item, no traceback, no doses recorded.
8. **Egress.** `nemoclaw careshell exec -- curl -I https://example.com` → refused at the
   proxy. Then `curl -s http://inference.local/v1/models` → success. Both halves matter:
   the second proves the policy delta did not strip the baseline.
9. **Streaming parity.** Batch and `--stream` over the same timeline produce an identical
   decision set (`tests/test_pipeline.py::test_stream_mode_produces_the_same_decisions`).
10. **Live feed.** `deploy/demo.sh` — the console fills in real time as
    `tools/feed_timeline.py` appends.

---

## 7. Known Gaps

Deliberate, for a hackathon build:

- **One patient per run.** The store is multi-patient and every row is patient-scoped; the
  CLI takes one timeline file.
- **Night-absence state is per-run and in-memory.** Restarting mid-episode resets the
  timer. Dose state is durable; this is not.
- **Missed-dose backfill is capped at 31 days.**
- **No auth on the nurse console.** Protected only by the LAN CIDR in the ingress rule.
- **`--trust-keywords` is a demo affordance.** Pattern matching is not clinical evidence.
- **The LLM remains a single point of misclassification.** The `CONFIRMED`-only guard
  makes it fail toward under-recording — a missed-dose alert rather than a false block.
  That is the correct direction to fail, not zero risk.
