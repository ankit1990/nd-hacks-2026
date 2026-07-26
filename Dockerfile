# Optional: for the `--network=none` variant described in spec.md section 5.
#
# The normal deployment path is NemoClaw + OpenShell (see deploy/deploy.sh). Use this
# image only when you want a genuine kernel-level network guarantee for the process that
# touches PHI:
#
#   docker build -t careshell:local .
#   docker run --rm --network=none --cap-drop=ALL \
#     -v "$PWD/data/patients:/app/data/patients:ro" \
#     -v "$PWD/workspace:/app/workspace:rw" \
#     careshell:local python -m careshell.run /app/data/patients/P104.jsonl \
#       --offline --trust-keywords --console '' --no-tts
#
# Note the trade-off: --network=none also blocks inference.local, so this mode runs the
# offline extractor. It proves the air gap; it does not run the model.

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# pyttsx3 needs a system speech engine that a slim image does not have, and
# alerting.tts_engine already degrades to stdout without it.
RUN grep -v '^pyttsx3' requirements.txt > /tmp/reqs.txt \
 && pip install --no-cache-dir -r /tmp/reqs.txt

COPY careshell/ ./careshell/
COPY alerting/ ./alerting/
COPY dashboard/ ./dashboard/
COPY tools/ ./tools/
COPY tests/ ./tests/
COPY pytest.ini ./

RUN mkdir -p /app/data/patients /app/data/live /app/workspace /app/logs

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CARESHELL_DB=/app/workspace/careshell.db

CMD ["python", "-m", "careshell.run", "--help"]
