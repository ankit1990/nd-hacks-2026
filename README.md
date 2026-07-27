# CareShell

Local-first eldercare care-plan agent. Reads a per-patient timeline of observed events as
plain text, reconciles each one against that patient's care plan, and raises local alerts:
double-dose interception, missed doses, out-of-window intakes, and night-time wandering.

Runs entirely on a Dell Pro Max GB10 inside a NemoClaw / OpenShell sandbox with no
outbound network route. No patient data leaves the device.

See [`spec.md`](spec.md) for the full technical specification.

---

## The one rule that shapes everything

**The LLM interprets. Python decides.**

The model's only job is turning one line of messy text into a typed event:

> `"John opened the blue bottle and swallowed one beta blocker tablet with a glass of water."`
> → `{kind: MED_INTAKE, med_id: MED_MORNING, certainty: CONFIRMED}`

Everything with a safety consequence — lockout windows, daily dose caps, missed-dose
timers — is deterministic Python in `careshell/reconciler.py`. No model in that path.

A double-dose lockout a language model can talk itself out of is not a safety control.
It also means the demo is reproducible: the same timeline yields the same alerts, run
after run.

Two corollaries:

- Only `certainty=CONFIRMED` records a dose. "Reached for the bottle" is not "took the
  pill", and it becomes a `NEEDS_HUMAN_REVIEW` item instead.
- The timeline's own timestamps are the clock. Nothing calls `datetime.now()` during a
  replay, so an 08:00 medication window evaluates as 08:00 no matter when you run it.

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Safety logic. No GPU, no model, no network.
PYTHONPATH=. .venv/bin/python -m pytest

# Full pipeline against the demo timeline, no inference required.
PYTHONPATH=. .venv/bin/python -m careshell.run data/patients/P104.jsonl \
  --offline --trust-keywords --console '' --no-tts
```

That surfaces all four scenarios: an on-time dose, an intercepted double dose, a
night-time wandering alert, and a missed morning dose.

---

## Input format

One JSON object per line, in `data/patients/<patient_id>.jsonl`:

```jsonl
{"id":"e005","ts":"2026-03-14T07:58:22","patient_id":"P104","source":"nurse_note","text":"John opened the blue bottle and swallowed one beta blocker tablet with a glass of water."}
```

| Field | Meaning |
| --- | --- |
| `id` | Stable and unique. Reconciliation is idempotent on this — replaying never double-counts. |
| `ts` | ISO 8601, facility-local. This is the system clock. |
| `source` | `sensor` \| `nurse_note` \| `device`. Informs how much the text is trusted. |
| `text` | One free-form English observation. |

Whatever produces these lines — bedside sensors, a nurse's tablet, a device bridge — is
outside CareShell's scope. This file is the entire contract.

---

## Running modes

```bash
# batch: as fast as possible. For tests and CI.
python -m careshell.run data/patients/P104.jsonl

# stream: paced replay of a complete file. --speed 120 = 2 minutes of timeline per second.
python -m careshell.run data/patients/P104.jsonl --stream --speed 120

# follow: tail a file something else is appending to. This is live ingest.
python -m careshell.run --follow data/live/P104.jsonl --idle-timeout 120
```

Stream mode merges entry timestamps with medication window-close moments, so a
`MISSED_DOSE` fires at 08:30 of virtual time rather than being lumped into whatever
observation happens next.

### Streaming demo

Two processes sharing one file:

```bash
# terminal 1 — CareShell, tailing the live feed
python -m careshell.run --follow data/live/P104.jsonl --from-start --idle-timeout 60

# terminal 2 — push the source timeline into that feed over time
python tools/feed_timeline.py data/patients/P104.jsonl data/live/P104.jsonl --speed 240
```

`deploy/demo.sh` runs both against a NemoClaw sandbox.

---

## Nurse console

```bash
CARESHELL_DB=workspace/careshell.db uvicorn dashboard.app:app --host 0.0.0.0 --port 8000
```

`http://<host>:8000` — decision history from SQLite on load, then live over a WebSocket.
Reconnects automatically. LAN-only; the OpenShell ingress rule restricts port 8000 to the
nurse-station subnet.

---

## Deploying to NemoClaw

```bash
./deploy/deploy.sh <sandbox>        # sync, deps, resolve model, start console, forward :8000
./deploy/preflight.sh <sandbox>     # 11 checks, all run from inside the sandbox
./deploy/demo.sh <sandbox> 2        # streaming demo, 2s between entries
```

Verified end to end against **nemoclaw v0.0.93 / openshell 0.0.85**. Both are alpha —
re-check subcommand names against your installed version.

### What the deployment actually looks like

Several things differ from what a reading of the NemoClaw docs suggests. These were found
by running it, not by guessing:

- **`inference.local` is https.** Plain `http://inference.local` is refused by the egress
  proxy with a 403. The scheme is not cosmetic.
- **No policy delta is needed.** The baseline already grants `read_write` on `/sandbox`
  and routes `inference.local`. Policy presets are **network-only** — the schema's only
  top-level keys are `preset` and `network_policies`, so a preset *cannot* declare
  filesystem mounts or ingress rules. See `openshell/careshell-preset.yaml`.
- **Code gets in via `nemoclaw <sandbox> upload`, not a host bind mount.** Note that
  `upload <dir> <dest>` places the directory *inside* `dest` as `dest/<basename>`.
- **The console reaches the host through a gRPC tunnel**, not a policy ingress rule:
  `openshell forward service --target-port 8000 --local 8000 <sandbox>`.
- **The sandbox image is PEP 668 managed.** A plain `pip install` fails; a venv is
  required.
- **Background processes need `setsid` with stdin detached.** A plain `nohup ... &` inside
  `exec` is reaped when the exec channel closes.
- **Don't resolve the model as `data[0]`** from `/v1/models` — on an OpenAI-backed gateway
  that is often `text-embedding-ada-002`, which cannot serve chat completions.
  `deploy/pick_model.py` prefers the sandbox's configured model and skips non-chat ids.
- **Servers disagree on the output-length parameter.** vLLM takes `max_tokens`; newer
  OpenAI-hosted models reject it and require `max_completion_tokens`. The extractor
  negotiates this once on the first rejection and remembers the answer, so the same build
  works against either.

If you apply a preset anyway (`CARESHELL_APPLY_PRESET=1`), use `policy-add`, which layers
onto the baseline. Never `openshell policy set` — that replaces the baseline and strips
`inference.local` and the gateway dial-back WebSocket.

### The egress guarantee, stated accurately

OpenShell enforces egress with a userspace HTTP CONNECT proxy plus OPA, **not** a kernel
packet filter. There is no `egress: deny-all` key; denial is by omission from the
allowlist. The kernel-enforced parts are Landlock (filesystem) and seccomp (syscalls).

The air-gap check prints exactly this:

```
curl: (56) CONNECT tunnel failed, response 403
```

— visibly a proxy refusal. Describe it as one. The `Dockerfile` has a `--network=none`
variant for a genuine kernel-level guarantee, with the caveat that it also blocks
`inference.local`.

### GB10 specifics

On a Dell Pro Max GB10 with a local vLLM, everything above holds except that
`preflight.sh` will also report a GPU, and the resolved model will be whatever vLLM
serves rather than an OpenAI-hosted id. Dropping the VLM from the design means there is
no sm_121 vision-kernel risk and no NIM support-matrix constraint — any instruct text
model works.

---

## Layout

```
careshell/          schemas, care-plan loader, timeline reader, extractor,
                    reconciler (the safety logic), SQLite store, streaming, CLI
alerting/           bedside TTS, LAN broadcaster — both fail soft
dashboard/          FastAPI nurse console
openshell/          policy delta for the NemoClaw sandbox
deploy/             deploy.sh, preflight.sh, demo.sh
tools/              feed_timeline.py — pushes a source file into a live feed
workspace/          care_plan.yaml, careshell.db (SQLite), MEMORY.md
tests/              123 tests; test_reconciler.py needs no GPU, model, or network
```

### Timestamps

`TimelineEntry` normalises any offset-bearing `ts` (`...Z`, `+05:30`) to a naive,
facility-local wall clock at the schema boundary. Medication windows are wall-clock
times, and everything downstream compares against `datetime.combine` results, which are
naive — mixing the two raises `TypeError`. Normalising once at ingestion means a sensor
bridge emitting UTC cannot crash the run.

### Why SQLite

Dose state was previously recovered by string-splitting markdown log lines. SQLite is
stdlib, needs no server, and gives idempotency through `UNIQUE` constraints plus a real
queryable history for the console. `MEMORY.md` remains as the human-readable narrative;
the database is authoritative.

---

## Known gaps

Deliberate, for a hackathon build:

- **One patient per run.** The store is multi-patient; the CLI takes one timeline.
- **Night-absence state is per-run and in-memory.** Restarting mid-episode resets the
  timer. Dose state is durable; this one is not.
- **Missed-dose backfill is capped at 31 days** so a stale timeline cannot emit thousands
  of retroactive alerts.
- **No auth on the nurse console.** It is protected only by the LAN CIDR in the ingress
  rule.
- **The offline keyword extractor is a fallback, not a classifier.** It refuses to return
  `CONFIRMED` unless `--trust-keywords` is passed, which is a demo affordance only.
- **The LLM remains a single point of misclassification.** The `CONFIRMED`-only guard
  means it fails toward under-recording — a missed-dose alert rather than a false block.
  That is the correct direction to fail, not zero risk.
