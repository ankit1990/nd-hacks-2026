```markdown
# CareShell: Technical Specification (`spec.md`)

**System Name:** CareShell  
**Target Environment:** Local Dell Pro Max (NVIDIA Grace Blackwell GB10 GPU)  
**Security Model:** Air-gapped, zero-egress OS container (`egress: deny-all`)  
**Core Frameworks:** OpenClaw Gateway Daemon, OpenShell, Local VLM (Qwen-2.5-VL / vLLM / Ollama), OpenCV, FastAPI  
**Primary Input:** Static video feed (`data/sample_video.mp4`)  

---

## 1. System Overview & Architecture

CareShell is an air-gapped, local-first eldercare vision agent designed to continuously monitor patient bedside activity, track medication adherence, evaluate sleep behavior, and enforce double-dosing protection.

Zero video or visual frame data leaves the physical device. OpenShell enforces an OS kernel-level network lock (`egress: deny-all`), while local vision inference and schedule reconciliation are handled by a local VLM on Dell GB10 hardware and an OpenClaw gateway daemon.

### Data Flow Diagram


```

+-----------------------------------------------------------------------------------+
| INPUT LAYER                                                                       |
|  [ Static Video File Feed (data/sample_video.mp4) ] ---> OpenCV Frame Buffer      |
+-----------------------------------------------------------------------------------+
|
v
+-----------------------------------------------------------------------------------+
| TIER 1: SPATIAL MOTION & TRIGGER ENGINE (`vision_engine/trigger.py`)             |
|  - Frame-differencing / motion thresholding on static video stream                |
|  - Target ROI: Bed Area + Bedside Table                                           |
|  - Triggers Tier 2 only on state change or scheduled evaluation windows           |
+-----------------------------------------------------------------------------------+
| (Frame Buffer)
v
+-----------------------------------------------------------------------------------+
| TIER 2: LOCAL VLM EVENT EXTRACTOR (`vision_engine/vlm_client.py`)                  |
|  - Engine: Local Qwen-2.5-VL (7B/72B) via local OpenAI-compatible endpoint        |
|  - Execution: Dell GB10 Unified VRAM                                             |
|  - Output: Enforced JSON Schema (Pill intake, sleep state, posture)               |
+-----------------------------------------------------------------------------------+
| (Structured Observation Event)
v
+-----------------------------------------------------------------------------------+
| OPENCLAW CARE DAEMON (`openclaw_workspace/`)                                      |
|  - Workspace: PATIENT_SCHEDULE.md, MEMORY.md, HEARTBEAT.md                        |
|  - Reconciler: Compares Observation JSON vs. Target Schedule                     |
|  - Action Dispatcher: Triggers local TTS or Nurse Dashboard Alert                 |
+-----------------------------------------------------------------------------------+
|
v
+-----------------------------------------------------------------------------------+
| SECURITY & OPENSHELL CONTAINER (`openshell/care_shell_sandbox.yaml`)               |
|  - Kernel policy: egress: deny-all (Zero outbound internet)                       |
|  - Local LAN Ingress: Port 8000 (Nurse Station Console)                           |
+-----------------------------------------------------------------------------------+

```

---

## 2. Repository Layout


```

careshell/
├── README.md
├── spec.md
├── docker-compose.yml
├── requirements.txt
├── data/
│   └── sample_video.mp4
├── openshell/
│   └── care_shell_sandbox.yaml
├── vision_engine/
│   ├── **init**.py
│   ├── capture.py
│   ├── trigger.py
│   ├── vlm_client.py
│   └── schemas.py
├── openclaw_workspace/
│   ├── openclaw.json
│   ├── PATIENT_SCHEDULE.md
│   ├── MEMORY.md
│   ├── HEARTBEAT.md
│   └── reconciler.py
├── alerting/
│   ├── **init**.py
│   ├── tts_engine.py
│   └── lan_broadcaster.py
├── dashboard/
│   ├── app.py
│   └── templates/
│       └── index.html
└── tests/
├── test_vlm_schema.py
├── test_reconciler.py
└── test_pipeline.py

```

---

## 3. Core Component Specifications

### 3.1 Pydantic Event Schemas (`vision_engine/schemas.py`)

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

class MedicationEvent(BaseModel):
    detected: bool = Field(description="True if patient is attempting or taking medication")
    action_stage: Optional[Literal["REACHING", "CAP_OPEN", "SWALLOWING", "WATER_DRINKING"]] = None
    bottle_color: Optional[str] = Field(default=None, description="Color or label of pill bottle")
    pill_type: Optional[str] = Field(default=None, description="Visual description of pill (e.g. blue_round)")

class SleepBehaviorEvent(BaseModel):
    bed_occupied: bool = Field(description="True if patient is currently in bed")
    posture: Literal["LYING_STILL", "TOSSING_TURNING", "SITTING_ON_EDGE", "OUT_OF_BED"]
    restlessness_score: int = Field(ge=1, le=5, description="1=Calm, 5=Severe tossing/restlessness")

class VisionObservation(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    medication: MedicationEvent
    behavior: SleepBehaviorEvent
    confidence: float = Field(ge=0.0, le=1.0)
    raw_summary: str = Field(description="Brief 1-sentence visual description of frame")

```

---

### 3.2 Local VLM Extraction Interface (`vision_engine/vlm_client.py`)

```python
import cv2
import base64
import json
import requests
from vision_engine.schemas import VisionObservation

class LocalVLMClient:
    def __init__(self, endpoint_url: str = "http://localhost:11434/v1/chat/completions", model_name: str = "qwen2.5-vl"):
        self.endpoint_url = endpoint_url
        self.model_name = model_name

    def encode_frame(self, frame) -> str:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def analyze_frame(self, frame) -> VisionObservation:
        base64_image = self.encode_frame(frame)
        
        prompt = """
        Analyze this bedside room frame from the static video feed for eldercare monitoring.
        Extract medication actions (reaching, opening cap, swallowing, drinking water) and patient posture/bed state.
        Respond STRICTLY in JSON matching this schema:
        {
          "timestamp": "<ISO timestamp>",
          "medication": {
            "detected": true/false,
            "action_stage": "REACHING" | "CAP_OPEN" | "SWALLOWING" | "WATER_DRINKING" | null,
            "bottle_color": "string or null",
            "pill_type": "string or null"
          },
          "behavior": {
            "bed_occupied": true/false,
            "posture": "LYING_STILL" | "TOSSING_TURNING" | "SITTING_ON_EDGE" | "OUT_OF_BED",
            "restlessness_score": 1-5
          },
          "confidence": 0.0-1.0,
          "raw_summary": "1 sentence description"
        }
        """

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1
        }

        response = requests.post(self.endpoint_url, json=payload, timeout=10)
        response.raise_for_status()
        result_json = response.json()['choices'][0]['message']['content']
        
        parsed_data = json.loads(result_json)
        return VisionObservation(**parsed_data)

```

---

### 3.3 OpenClaw Care Plan Specification (`openclaw_workspace/PATIENT_SCHEDULE.md`)

```markdown
# Patient Plan: Room 104 - John Doe

## Scheduled Medications
- ID: MED_MORNING
  Time: 08:00 AM
  Window_Start: 07:30 AM
  Window_End: 08:30 AM
  Name: Beta Blocker (Blue Pill)
  Max_Daily_Doses: 1

- ID: MED_EVENING
  Time: 08:00 PM
  Window_Start: 07:30 PM
  Window_End: 08:30 PM
  Name: Evening Statin (White Round Pill)
  Max_Daily_Doses: 1

## Behavior Rules
- Target Wake Time: 08:00 AM
- Max Inactivity Out of Bed at Night: 15 minutes

```

---

### 3.4 Event & Schedule Reconciler (`openclaw_workspace/reconciler.py`)

```python
import json
import os
from datetime import datetime
from vision_engine.schemas import VisionObservation
from alerting.tts_engine import LocalTTS
from alerting.lan_broadcaster import LANBroadcaster

class CareReconciler:
    def __init__(self, workspace_path: str = "./openclaw_workspace"):
        self.workspace_path = workspace_path
        self.tts = LocalTTS()
        self.broadcaster = LANBroadcaster()
        self.memory_file = os.path.join(workspace_path, "MEMORY.md")

    def process_observation(self, obs: VisionObservation):
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

        # 1. Handle Medication Event
        if obs.medication.detected and obs.medication.action_stage == "SWALLOWING":
            last_taken = self._get_last_med_time()
            
            # Check for Double-Dosing (Lockout window: 4 hours = 14400 seconds)
            if last_taken and (now - last_taken).total_seconds() < 14400:
                alert_msg = "WARNING: You already took your scheduled medication."
                self.tts.speak(alert_msg)
                self.broadcaster.send_alert("DOUBLE_DOSE_PREVENTED", alert_msg)
                self._log_memory(f"[{timestamp_str}] ALERT: Intercepted double-dose attempt.")
            else:
                success_msg = "Morning medication recorded. Thank you, John."
                self.tts.speak(success_msg)
                self.broadcaster.send_event("MED_TAKEN_SUCCESS", {"time": timestamp_str, "type": obs.medication.pill_type})
                self._log_memory(f"[{timestamp_str}] SUCCESS: Took medication ({obs.medication.pill_type}).")

        # 2. Handle Nighttime Out-of-Bed Anomaly
        if not obs.behavior.bed_occupied and (now.hour >= 22 or now.hour < 6):
            self.broadcaster.send_event("NIGHT_OUT_OF_BED", {"duration_check_needed": True})
            self._log_memory(f"[{timestamp_str}] BEHAVIOR: Patient out of bed during sleep hours.")

    def _log_memory(self, entry: str):
        with open(self.memory_file, "a") as f:
            f.write(f"{entry}\n")

    def _get_last_med_time(self):
        if not os.path.exists(self.memory_file):
            return None
        with open(self.memory_file, "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            if "SUCCESS: Took medication" in line:
                time_str = line.split("]")[0].replace("[", "")
                return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        return None

```

---

### 3.5 Local Audio TTS Output (`alerting/tts_engine.py`)

```python
import pyttsx3

class LocalTTS:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 140)  # Slower speed for elderly clarity
            self.engine.setProperty('volume', 1.0)
        except Exception as e:
            print(f"[TTS Initialization Warning] {e}")
            self.engine = None

    def speak(self, text: str):
        print(f"[LOCAL TTS SPEAKER Output]: {text}")
        if self.engine:
            self.engine.say(text)
            self.engine.runAndWait()

```

---

### 3.6 OpenShell Sandbox Policy (`openshell/care_shell_sandbox.yaml`)

```yaml
version: "1.0"
name: careshell-kernel-sandbox

runtime:
  gpu:
    enabled: true
    devices: ["0"] # Access to Dell GB10 GPU

network:
  # Hard Kernel Lockdown: All outbound traffic dropped
  egress:
    mode: deny-all
  # Local Ingress: Only permit Local Area Network access for Nurse Station Console
  ingress:
    - port: 8000
      protocol: tcp
      cidr: 192.168.1.0/24

filesystem:
  read_only_root: true
  mounts:
    - host: ./data
      container: /app/data
      mode: ro
    - host: ./openclaw_workspace
      container: /app/openclaw_workspace
      mode: rw
    - host: ./logs
      container: /app/logs
      mode: rw

security:
  capabilities:
    drop: ["ALL"]
  seccomp: strict

```

---

### 3.7 Nurse Console Backend (`dashboard/app.py`)

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json

app = FastAPI(title="CareShell Local Nurse Console")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_text(json.dumps(message))

manager = ConnectionManager()

@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/event")
async def receive_event(event: dict):
    await manager.broadcast(event)
    return {"status": "broadcasted"}

```

---

## 4. Execution Pipeline & Test Harness (`tests/test_pipeline.py`)

```python
import cv2
import time
import os
from vision_engine.vlm_client import LocalVLMClient
from openclaw_workspace.reconciler import CareReconciler

def run_static_video_harness(video_path="data/sample_video.mp4"):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Static video file not found at {video_path}")

    print(f"[CareShell] Starting Execution Loop on static video feed: {video_path}...")
    cap = cv2.VideoCapture(video_path)
    vlm = LocalVLMClient()
    reconciler = CareReconciler()

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[CareShell] Reached end of static video feed.")
            break

        # Process 1 frame every 90 frames (e.g. every 3 seconds of 30fps video)
        if frame_count % 90 == 0:
            print(f"\n--- Analyzing Frame {frame_count} ---")
            try:
                obs = vlm.analyze_frame(frame)
                print(f"[VLM Summary]: {obs.raw_summary}")
                print(f"[Posture]: {obs.behavior.posture} | [Medication Detected]: {obs.medication.detected}")
                
                # Reconcile event against schedule & memory
                reconciler.process_observation(obs)
            except Exception as e:
                print(f"[Error in pipeline]: {e}")

        frame_count += 1
        time.sleep(0.01)

    cap.release()

if __name__ == "__main__":
    run_static_video_harness()

```

---

## 5. Verification Protocols

1. **Static Pipeline Execution:** Place `sample_video.mp4` into `data/` and run `python tests/test_pipeline.py`. Ensure structured Pydantic visual observations print to stdout without errors.
2. **Double-Dosing Intercept:** Ensure static video footage contains sequential medication attempts within 4 hours. Verify `LocalTTS` generates `"WARNING: You already took your scheduled medication."`
3. **Air-Gap Verification:** Inside the OpenShell container, run `curl -I https://google.com` to confirm network rejection (`egress: deny-all`), while internal LAN port `8000` remains accessible.

```

```
