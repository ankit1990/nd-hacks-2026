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

## Deploying to NemoClaw on GB10

Assumes NemoClaw is installed with a vLLM already attached, exposed inside the sandbox as
`inference.local`.

```bash
./deploy/deploy.sh careshell      # apply policy delta, install deps, start the console
./deploy/preflight.sh careshell   # verify GPU, vLLM, egress, mounts, deps, tests
./deploy/demo.sh careshell 240    # streaming demo
```

Three things that matter, in order:

1. **Verify vLLM from inside the sandbox, never the host.** The host can reach things the
   sandbox cannot. `preflight.sh` does this and prints the model id to pass to `--model`.

2. **Apply a policy *delta*, never a full replacement.**
   `nemoclaw careshell policy-add --from-file openshell/careshell-preset.yaml`.
   Running `openshell policy set` replaces NemoClaw's baseline and silently strips
   `inference.local`, the gateway dial-back WebSocket, and writable `/dev/pts` — which
   breaks the exec tool. `deploy.sh` checks `policy-show` afterward and aborts if
   `inference.local` vanished.

3. **State the egress guarantee accurately.** OpenShell enforces egress with a userspace
   HTTP CONNECT proxy plus OPA, *not* a kernel packet filter. There is no `egress:
   deny-all` key; denial is by omission from the allowlist. The kernel-enforced parts are
   Landlock (filesystem) and seccomp (syscalls). Your air-gap demo prints:

   ```
   curl: (56) Received HTTP code 403 from proxy after CONNECT
   ```

   which is visibly a proxy refusal. Describe it as one. The `Dockerfile` has a
   `--network=none` variant for a genuine kernel-level guarantee, with the caveat that it
   also blocks `inference.local`.

Both NemoClaw and OpenShell are Apache-2.0 and explicitly alpha — verify CLI flag names
against your installed version before demo day.

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
tests/              93 tests; test_reconciler.py needs no GPU, model, or network
```

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
