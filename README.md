# Smart Bag Vision

A paper-bag handling cell with three parts:

- A **uArm Swift Pro** with a suction cup picks bags from a stack.
- **SO-100 arms** (Arm 1, Arm 2, Buddy) move the bags on.
- An **Intel RealSense D435** camera checks every bag. AI decides GOOD / BAD and sends it to the matching place.

Everything is run from one web dashboard, the **BagVision Control Center**.

## Hardware

| Device | Connection | Notes |
|---|---|---|
| uArm Swift Pro (suction) | USB (Arduino Mega, 2341:0042), 115200 baud | Found automatically |
| SO-100 arms | USB servo-bus adapter (CH343, 1a86:55d3), 1 Mbit/s, servos ID 1–6 | Each arm is recognised by its adapter serial number, see `so100_settings.json` |
| SO-100 leader arm (optional) | Same adapter type | For teaching by moving the leader |
| RealSense D435 | USB 3 | Colour stream only, top-down over the board |

Arm names and adapter serials (`so100_settings.json`):

| Name | Adapter serial |
|---|---|
| `1` | `5B14030143` |
| `2` | `5B14113529` |
| `buddy` | `5A7C123263` (7.4 V motors, needs a **5 V** supply) |
| `leader` | `5B14030142` |

The arm program refuses to switch motors on when the supply is above the motors' voltage limit.

## Setup

Ubuntu, Python 3.12, NVIDIA GPU recommended.

```bash
pip install -r requirements.txt
sudo usermod -aG dialout $USER        # access to the robot USB ports (log in again)
```

## Run the dashboard

```bash
python3 dashboard.py                  # http://localhost:8000
```

| Page | What it does |
|---|---|
| **Live** | Live video with GOOD / BAD / EMPTY per bag place; switch between YOLO classifier and defect model; KPIs |
| **Production** | Counts, yield, defect rate, review queue, CSV export |
| **Robots** | Every robot: connect, run a job (runs, speed), home; **sequence** (New / Continue); live status bar; rename jobs |
| **Teach** | Teach a job by hand (arm goes limp) or with the leader arm; uArm stack height |
| **Dataset & AI** | Mark bag places, empty board, collect photos, train YOLO, bag types (defect model), where the AI runs (laptop / IPC) |

**The sequence** (Robots page) runs one cycle like this:
1. uArm picks a bag. Photo.
2. Arm 1 job. Photo.
3. Arm 2 runs its GOOD job or BAD job, depending on the camera. Meanwhile the uArm picks the next bag.

The bag count of the uArm stack and the sequence progress are saved, so **Continue** carries on after a stop or restart.

## Command-line tools

| File | Use |
|---|---|
| `uarm_pick_place.py` | uArm: `info`, `move`, `pump`, `home`, `learn N`, `replay N [--count --step --speed]`, `jobs` |
| `so100_arm.py` | SO-100: `--arm NAME info / learn N [--leader] / replay N / jobs / park / free / copy-jobs / leader-calibrate / calibrate / move X Y Z` |
| `run_sequence.py` | The production sequence without the dashboard (`--runs N`, `--check`) |
| `bag_inspect.py` | YOLO classifier: `capture`, `train`, `inspect`, `check FOLDER` |
| `anomaly.py` | Defect model (PatchCore, WideResNet-50): `build`, `test IMAGE` |
| `enroll.py` | One-shot set-up of a new bag type (used by the dashboard) |
| `inspect_server.py` / `ai_client.py` | AI as a network service (e.g. on a Siemens IPC); the dashboard sends pictures and falls back to local models |
| `realsense_view.py`, `capture_dataset.py` | Camera viewer, older photo capture tool |
| `so100_sim_view.py` | Renders a simulation video of the SO-100 (`so100_sim.mp4`) |

## Data in this repository

```
jobs/                  uArm jobs (jobN.json)
so100_jobs/<serial>/   SO-100 jobs per arm (jobN.json, optional "name")
so100_settings.json    arm names, end stops, rest poses
so101_follower.urdf    arm kinematics
bag_models/            best.pt (YOLO classifier), anomaly.pt (defect model), profiles/ (bag types), history/ (older models)
edge/                  Docker app for the AI on a Siemens Industrial Edge IPC (build.sh, docker-compose.yml)
```

Not included, because they are large and stay on the machine: bag photos (`bag_data/`), sequence photos (`captures/`), inspection results (`inspections/`) and logs.

## AI on a Siemens IPC (optional)

`edge/build.sh` builds the inspection service image (`bagvision-ai_<version>.tar`). Install it with the Industrial Edge App Publisher and Industrial Edge Management, using `edge/docker-compose.yml` (port 8100, `AI_TOKEN`).

Then in the dashboard go to **Dataset & AI → Where the AI runs**, enter `http://<ipc>:8100` and the token, and press **Use IPC**.
