#!/usr/bin/env python3
"""
uArm Swift Pro (UFactory) - teach by hand, replay the same path.

Usage:
  python3 uarm_pick_place.py learn 1                     # teach job 1 by hand (records the whole path)
  python3 uarm_pick_place.py learn 2                     # teach job 2, and so on
  python3 uarm_pick_place.py replay 1                    # run job 1 - asks how many pick-ups
  python3 uarm_pick_place.py replay 2 --count 5          # run job 2 five times without asking
  python3 uarm_pick_place.py jobs                        # list the taught jobs
  python3 uarm_pick_place.py info                        # device details + live position
  python3 uarm_pick_place.py move X Y Z [speed]          # move the suction tip to X Y Z (mm)
  python3 uarm_pick_place.py pump on|off
  python3 uarm_pick_place.py home                        # go to the home pose, close to the robot

Stacks: each job remembers how far the pick height goes down after every pick-up
(e.g. 2 mm = paper thickness for a stack, 0 = always the same height).

Coordinates: mm, X forward, Y left(+)/right(-), Z up, base at the origin.

How it works: with the motors off the firmware's X/Y/Z report (P2220) is frozen,
so the live position is read from the joint angle sensors (P2200) and converted
with the firmware's own forward kinematics (M2221). The recorded path is
smoothed, simplified and replayed as a stream of small moves at the taught speed.
"""
import argparse
import json
import math
import os
import select
import sys
import termios
import threading
import time
import tty

import serial

PORT = None                 # None = find the uArm by its Arduino Mega USB ID (2341:0042)


def find_uarm_port():
    """The uArm's Arduino Mega - never another device that also shows up as /dev/ttyACM*."""
    from serial.tools import list_ports
    for p in list_ports.comports():
        if p.vid == 0x2341 and p.pid == 0x0042:
            return p.device
    raise SystemExit("uArm not found - is its USB cable plugged in? "
                     "(other devices on /dev/ttyACM* are ignored on purpose)")
BAUD = 115200
JOBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")

SPEED = 1.0                 # replay speed: 1.0 = as taught, 1.5 = 50% faster (or use --speed)
SAFE_POSE = (120, 0, 60)    # home: tucked in close to the robot, out of the work area
                            # (closest reach: 116 mm at this height; higher up = further out,
                            #  e.g. 135 mm at Z 110)
TRAVEL_FEED = 3000          # mm/min to the start point / parking pose (x SPEED)
PRESS_FEED = 600            # mm/min for the short press onto the paper (x SPEED)
PRESS = 1.0                 # mm to push below the taught grab point for a good seal
SUCTION_DELAY = 0.8         # s to let vacuum build before moving on
RELEASE_DELAY = 1.0         # s to let the paper drop after the pump turns off
PATH_TOL = 0.8              # mm - recorded path may be simplified by at most this much
MIN_FEED, MAX_FEED = 300, 12000  # mm/min clamp (measured: the arm still gets faster up to ~12000)


class UArmError(RuntimeError):
    pass


class StackTooLow(UArmError):
    """The lowered pick height is out of reach - the stack is probably empty."""


class SwiftPro:
    ERRORS = {"E20": "command not supported", "E21": "parameter error",
              "E22": "target unreachable (outside workspace)", "E23": "command failed",
              "E24": "power/limit problem", "E25": "timeout"}

    def __init__(self, port=PORT, baud=BAUD):
        self.ser = serial.Serial(port or find_uarm_port(), baud, timeout=0.1)
        self.seq = 0
        # Keep DTR up when the port closes, so the NEXT open does not reset the
        # Arduino (a reset switches the motors off and the arm drops).
        attrs = termios.tcgetattr(self.ser.fd)
        attrs[2] &= ~termios.HUPCL
        termios.tcsetattr(self.ser.fd, termios.TCSANOW, attrs)
        # If this open did reset the board, Marlin prints "start" within ~2 s;
        # stay quiet until it has booted (bytes sent to the bootloader can hang it).
        buf, deadline = "", time.time() + 2.5
        while time.time() < deadline and "start" not in buf:
            buf += self.ser.read(1024).decode(errors="replace")
        if "start" in buf:
            deadline = time.time() + 8
            while time.time() < deadline and "@5 V" not in buf:
                buf += self.ser.read(1024).decode(errors="replace")
        if "@5 V0" in buf:
            raise UArmError("Arm reports no motor power - is the 12V adapter plugged in?")
        for _ in range(20):                # firmware can need a moment after boot
            try:
                self.cmd("P2201", timeout=0.5)
                break
            except UArmError:
                pass
        else:
            raise UArmError("Arm does not answer - unplug/replug USB and try again")

    def close(self):
        self.ser.close()

    def cmd(self, gcode, timeout=10.0):
        """Send one command with a sequence id and return its reply payload.
        Thread-safe: only one command is on the wire at a time (dashboard threads share the arm)."""
        if not hasattr(self, "_io"):
            self._io = threading.RLock()
        with self._io:
            return self._cmd(gcode, timeout)

    def _cmd(self, gcode, timeout):
        self.seq = self.seq % 9999 + 1
        tag = f"${self.seq} "
        self.ser.reset_input_buffer()
        self.ser.write(f"#{self.seq} {gcode}\n".encode())
        deadline = time.time() + timeout
        line = b""
        while time.time() < deadline:
            line += self.ser.readline()
            if not line.endswith(b"\n"):
                continue
            text = line.decode(errors="replace").strip()
            line = b""
            if not text.startswith(tag):
                continue  # async report (@...) or a stale reply
            reply = text[len(tag):]
            code = reply.split()[0]
            if code in self.ERRORS:
                raise UArmError(f"{gcode!r} -> {code}: {self.ERRORS[code]}")
            return reply[3:] if reply.startswith("ok") else reply
        raise UArmError(f"No reply to {gcode!r}")

    @staticmethod
    def _nums(reply):
        return [float(t[1:]) for t in reply.split()]

    # --- sensing ----------------------------------------------------------
    def angles(self):
        """Live base/left/right joint angles from the angle sensors."""
        return self._nums(self.cmd("P2200"))

    def to_xyz(self, b, l, r):
        """Firmware forward kinematics (includes the suction-cup offset)."""
        return self._nums(self.cmd(f"M2221 B{b:.2f} L{l:.2f} R{r:.2f}"))

    def position(self):
        """Live X/Y/Z - correct with motors on or off."""
        return self.to_xyz(*self.angles())

    def reachable(self, x, y, z):
        return self.cmd(f"M2222 X{x:.2f} Y{y:.2f} Z{z:.2f} P0") == "V1"

    def info(self):
        q = lambda c: self.cmd(c).lstrip("V")
        return {"device": q("P2201"), "hardware": q("P2202"), "firmware": q("P2203"),
                "uid": q("P2205"), "mode": q("P2400"),
                "pump": {"0": "off", "1": "working", "2": "on/holding"}.get(q("P2231"), "?"),
                "position": tuple(round(v, 1) for v in self.position())}

    # --- acting -----------------------------------------------------------
    def wait_idle(self, timeout=60):
        # M2200 can briefly report "stopped" between queued moves, so require it twice.
        deadline = time.time() + timeout
        time.sleep(0.15)
        stopped = 0
        while time.time() < deadline:
            stopped = stopped + 1 if self.cmd("M2200") == "V0" else 0
            if stopped >= 2:
                return
            time.sleep(0.05)
        raise UArmError("Timed out waiting for motion to finish")

    def move(self, x, y, z, speed=3000, wait=True):
        if not self.reachable(x, y, z):
            raise UArmError(f"X{x:.1f} Y{y:.1f} Z{z:.1f} is outside the arm's reach")
        self.cmd(f"G0 X{x:.2f} Y{y:.2f} Z{z:.2f} F{speed:.0f}", timeout=60)
        if wait:
            self.wait_idle()

    def pump(self, on):
        self.cmd(f"M2231 V{1 if on else 0}")

    def detach(self):
        self.cmd("M2019")   # motors off - arm can be moved by hand

    def attach(self):
        self.cmd("M17")     # motors on; firmware re-reads the real position


class Keys:
    """Single-keypress reader for the Linux terminal (no Enter needed)."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        termios.tcflush(self.fd, termios.TCIFLUSH)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def get(self, timeout=None):
        if not select.select([sys.stdin], [], [], timeout)[0]:
            return None
        k = os.read(self.fd, 8).decode(errors="ignore")
        return k if k.startswith("\x1b") else k.lower()


# --- learn: record the path ---------------------------------------------------
LEARN_HELP = """
=====================  TEACH THE ROBOT BY HAND  =====================
 The robot records the WHOLE path you move it along, and replays it.
 Keys (no Enter needed):
   S      start recording            G   GRAB  - pump ON  (cup on paper)
   ENTER  stop recording & save      R   RELEASE - pump OFF (paper on target)
   X      throw away & record again  Q   quit without saving
 Move smoothly. Hold still for a moment when you press G or R.
======================================================================
"""


def learn_hint(recording, events):
    acts = [a for _, a in events]
    if not recording:
        return ("STEP 1/4  Move the cup to where the path should START\n"
                "          (e.g. ~3 cm above the paper), then press  S  to start recording.")
    if not acts:
        return ("STEP 2/4  [REC] Lower the cup until it touches the MIDDLE of the paper,\n"
                "          hold still and press  G  (pump turns ON).")
    if acts[-1] == "grab":
        return ("STEP 3/4  [REC] Lift and carry the paper to the target area. Lower it onto\n"
                "          the target, hold still and press  R  (pump turns OFF).")
    return ("STEP 4/4  [REC] Lift the cup away from the paper - this is the end of the path.\n"
            "          Then press  ENTER  to stop & save.  (G = pick another sheet)")


def learn(arm, job):
    path = job_path(job)
    print(f"\n######## Teaching JOB {job} ########")
    if os.path.exists(path):
        if input(f"Job {job} already exists - replace it with a new teaching? [y/N] ").strip().lower() != "y":
            print("Kept the old job - nothing changed.")
            return
    print(LEARN_HELP)
    fmt = lambda p: f"X{p[0]:7.1f} Y{p[1]:7.1f} Z{p[2]:7.1f}"
    samples, events, recording, pump, saved, msg = [], [], False, False, False, ""

    with Keys() as kb:
        print("STEP 0    Hold the arm with one hand (motors go off next),\n"
              "          then press any key.")
        kb.get()
        arm.detach()
        print("\n>> Motors OFF - the arm moves freely. Watch the numbers change as you move it.\n")
        print(learn_hint(recording, events) + "\n")
        last_draw = 0.0
        while True:
            pos = arm.position()
            now = time.time()
            if recording:
                samples.append((now, *pos))
            if now - last_draw > 0.1:
                rec = f"[REC {now - samples[0][0]:5.1f}s]" if recording else "[not recording]"
                print(f"\r\033[K  {rec} {fmt(pos)}  pump {'ON ' if pump else 'off'}  {msg}",
                      end="", flush=True)
                last_draw = now
            k = kb.get(0)
            if k is None:
                continue
            msg = ""
            if k == "s" and not recording:
                recording, samples, events = True, [(now, *pos)], []
                print(f"\r\033[K  * recording started at {fmt(pos)}\n")
                print(learn_hint(recording, events) + "\n")
            elif k in ("g", "r"):
                action = "grab" if k == "g" else "release"
                if not recording:
                    msg = "press S to start recording first"
                elif pump == (action == "grab"):
                    msg = "pump is already " + ("ON" if pump else "OFF")
                else:
                    pump = action == "grab"
                    arm.pump(pump)
                    events.append((len(samples) - 1, action))
                    label = "GRAB    - pump ON " if pump else "RELEASE - pump OFF"
                    print(f"\r\033[K  * {label} at {fmt(pos)}\n")
                    print(learn_hint(recording, events) + "\n")
            elif k == "x" and recording:
                recording, samples, events = False, [], []
                if pump:
                    pump = False
                    arm.pump(False)
                print("\r\033[K  * recording thrown away\n")
                print(learn_hint(recording, events) + "\n")
            elif k in ("\n", "\r"):
                if not recording or len(samples) < 2:
                    msg = "nothing recorded yet - press S to start"
                    continue
                saved = True
                break
            elif k in ("q", "\x1b"):
                break
            else:
                msg = f"key '{k.strip() or 'space'}' does nothing here"

        arm.pump(False)
        print("\r\033[K\nLAST STEP Hold the arm again, then press any key to lock the motors.")
        kb.get()
        arm.attach()
        print(">> Motors ON.")

    if not saved:
        print("\nQuit without saving - nothing changed.")
        return
    program = build_program(samples, events)
    print("\nAfter every pick-up, should the pick height go DOWN (a stack getting thinner)?")
    step = ask_number("  Go down by how many mm each time? (e.g. 2 = paper thickness, 0 = stay the same) [0]: ", 0)
    program.update(job=job, stack_step_mm=step, taught=time.strftime("%Y-%m-%d %H:%M"))
    os.makedirs(JOBS_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(program, f, indent=1)
    pts = program["points"]
    length = sum(math.dist(p_xyz(a), p_xyz(b)) for a, b in zip(pts, pts[1:]))
    print(f"\nSaved JOB {job} -> {path}")
    print(f"  {len(samples)} samples over {samples[-1][0] - samples[0][0]:.1f}s -> "
          f"{len(pts)} path points, {length:.0f} mm long, "
          f"{sum(1 for p in pts if 'action' in p)} pump actions, "
          + (f"pick height goes down {step:g} mm after each pick-up" if step else "pick height stays the same"))
    print(f"Next: python3 uarm_pick_place.py replay {job}")


def ask_number(prompt, default):
    while True:
        ans = input(prompt).strip()
        if not ans:
            return default
        try:
            v = float(ans)
            if v >= 0:
                return v
        except ValueError:
            pass
        print("   please type a number like 2 or 0")


def p_xyz(p):
    return (p["x"], p["y"], p["z"])


def simplify(pts, tol):
    """Ramer-Douglas-Peucker: indices of the points that keep the path within tol mm."""
    def dist_to_segment(p, a, b):
        ab = [b[i] - a[i] for i in range(3)]
        L2 = sum(v * v for v in ab)
        if L2 == 0:
            return math.dist(p, a)
        t = max(0.0, min(1.0, sum((p[i] - a[i]) * ab[i] for i in range(3)) / L2))
        return math.dist(p, [a[i] + t * ab[i] for i in range(3)])

    keep, stack = {0, len(pts) - 1}, [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        best, idx = 0.0, None
        for k in range(i + 1, j):
            d = dist_to_segment(pts[k], pts[i], pts[j])
            if d > best:
                best, idx = d, k
        if idx is not None and best > tol:
            keep.add(idx)
            stack += [(i, idx), (idx, j)]
    return sorted(keep)


def build_program(samples, events):
    """Smooth the raw samples, simplify the path, attach speeds and pump actions."""
    t = [s[0] for s in samples]
    raw = [s[1:] for s in samples]
    n = len(raw)

    # Pump actions split the path into pieces. Each piece is lightly smoothed
    # (+-0.1 s, removes sensor jitter) without touching its end points, so the
    # grab/release positions stay exactly where they were taught.
    cuts = sorted({0, n - 1, *(i for i, _ in events)})
    smooth = [list(p) for p in raw]
    for a, b in zip(cuts, cuts[1:]):
        for k in range(a + 1, b):
            win = [raw[j] for j in range(max(a + 1, k - 10), min(b, k + 11))
                   if abs(t[j] - t[k]) <= 0.1]
            smooth[k] = [sum(p[i] for p in win) / len(win) for i in range(3)]

    keep = set()
    for a, b in zip(cuts, cuts[1:]):
        keep.update(a + k for k in simplify(smooth[a:b + 1], PATH_TOL))
    keep = sorted(keep)

    actions = dict(events)
    points = []
    for prev, cur in zip([None] + keep, keep):
        p = {"x": round(smooth[cur][0], 2), "y": round(smooth[cur][1], 2),
             "z": round(smooth[cur][2], 2)}
        if prev is not None:
            # Speed = distance travelled / time spent actually moving (pauses ignored).
            length = moving = 0.0
            for k in range(prev, cur):
                d, dt = math.dist(smooth[k], smooth[k + 1]), t[k + 1] - t[k]
                length += d
                if dt > 0 and d / dt > 5:
                    moving += dt
            p["f"] = round(length / moving * 60) if moving else MIN_FEED
        if cur in actions:
            p["action"] = actions[cur]
        points.append(p)
    return {"version": 2, "points": points}


# --- replay -------------------------------------------------------------------
def job_path(job):
    return os.path.join(JOBS_DIR, f"job{job}.json")


def load_job(job):
    path = job_path(job)
    if not os.path.exists(path):
        have = list_jobs()
        sys.exit(f"Job {job} has not been taught yet - run:  python3 uarm_pick_place.py learn {job}"
                 + (f"\n(taught jobs: {', '.join(str(j) for j, _ in have)})" if have else ""))
    with open(path) as f:
        prog = json.load(f)
    if not isinstance(prog, dict) or prog.get("version") != 2:
        sys.exit(f"{path} is from the old teaching code - teach it again.")
    return prog


def list_jobs():
    jobs = []
    if os.path.isdir(JOBS_DIR):
        for name in os.listdir(JOBS_DIR):
            if name.startswith("job") and name.endswith(".json") and name[3:-5].isdigit():
                with open(os.path.join(JOBS_DIR, name)) as f:
                    jobs.append((int(name[3:-5]), json.load(f)))
    return sorted(jobs, key=lambda j: j[0])


def feed(f, speed):
    return min(MAX_FEED, max(MIN_FEED, f * speed))


TOUCH_MAX_STACK = 70.0     # touch-down: first search starts this far above the taught pick height
TOUCH_ABOVE = 15.0         # later searches start this far above the last stack top found
TOUCH_NEAR = 45.0          # path points closer than this (mm, sideways) to the pick are replaced
TOUCH_FLOOR = 3.0          # never search lower than the taught pick height minus this (= table)
TRAVEL_CLEARANCE = 15.0    # touch-down jobs: travel at least this far above the stack top found
APPROACH_ABOVE = 30.0      # the stack is approached / left vertically from this far above the search start
APPROACH_MIN = 70.0        # ... but never lower than this
touch_memory = {}          # (x, y) of a pick spot -> last stack top height found
SHEET_MM = 2.4             # thickness of one bag in the stack (20 bags ~ 48 mm)
MAX_SQUASH = 3.0           # with a bag count: never press more than this below the expected top
stack_left = {}            # (x, y) of a pick spot -> bags still in the stack (set by the operator)
STACK_FILE = os.path.join(JOBS_DIR, "stack_state.json")    # both kept here -> a restart continues where it stopped


def save_stack():
    try:
        keys = set(stack_left) | set(touch_memory)
        data = {f"{k[0]},{k[1]}": {"left": stack_left.get(k), "top": touch_memory.get(k)} for k in keys}
        with open(STACK_FILE, "w") as f:
            json.dump(data, f, indent=1)
    except OSError as e:
        print(f"  (could not save the stack state: {e})")


def load_stack():
    try:
        with open(STACK_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return
    for k, v in data.items():
        key = tuple(int(n) for n in k.split(","))
        if v.get("left") is not None:
            stack_left[key] = int(v["left"])
        if v.get("top") is not None:
            touch_memory[key] = float(v["top"])


def set_stack(x, y, count):
    stack_left[(round(x), round(y))] = int(count)
    save_stack()


load_stack()


def touch_down(arm, x, y, z_top, z_floor, stop=None):
    """Lower the cup 1 mm at a time until its touch switch clicks. Returns the height, or None."""
    z = z_top
    arm.move(x, y, z, 1500)
    if arm.cmd("P2233") == "V1":
        return z
    step, feed_mm_min = 2.0, 900                       # 2 mm steps at 15 mm/s (~0.2 s per step)
    while z > z_floor:
        if stop:
            stop()
        z = max(z_floor, z - step)
        arm.cmd(f"G0 X{x:.2f} Y{y:.2f} Z{z:.2f} F{feed_mm_min}", timeout=10)
        time.sleep(step / (feed_mm_min / 60) + 0.06)    # let the short move finish
        if arm.cmd("P2233") == "V1":
            return z
    return None


def touch_pick(arm, x, y, z_taught, speed, stop=None, pick_bags=1):
    """Find the stack top, then grab the top sheet. Checks the reach first.
    With a bag count (set_stack) the top is calculated - the cup comes down gently onto it and
    the touch switch is only a safety stop, so the soft stack is not squashed. Without a count
    the cup searches down until the switch clicks."""
    key = (round(x), round(y))
    count = stack_left.get(key)
    # z_taught = top of the stack when it held `pick_bags` bags (set when teaching)
    bottom = z_taught - (pick_bags - 1) * SHEET_MM - TOUCH_FLOOR      # cup on the last bag / table
    if count is not None and count <= 0:
        print(f"  bag count is {count} - searching down to the bottom of the stack instead")
        count = None
        touch_memory.pop(key, None)                      # start the search above a full-height stack
    expected = z_taught + (count - pick_bags) * SHEET_MM if count is not None else None
    z_floor = expected - MAX_SQUASH if expected is not None else bottom
    z_top = (expected + 10 if expected is not None else
             touch_memory.get(key, z_taught + TOUCH_MAX_STACK - TOUCH_ABOVE) + TOUCH_ABOVE)
    while z_top > z_floor and not arm.reachable(x, y, z_top):
        z_top -= 5                                       # start as high as the arm can reach
    if not (arm.reachable(x, y, z_top) and arm.reachable(x, y, z_floor)):
        raise UArmError(f"pick spot X{x:.0f} Y{y:.0f} cannot be reached from Z{z_top:.0f} down to Z{z_floor:.0f}")
    # come in HIGH and go straight down - a slanted approach skims over the stack and its handles
    z_high = max(z_top + APPROACH_ABOVE, APPROACH_MIN)
    while z_high > z_top and not arm.reachable(x, y, z_high):
        z_high -= 5
    arm.move(x, y, z_high, 1500)
    print(f"  TOUCH-DOWN at X{x:.1f} Y{y:.1f}: straight down from Z{z_high:.0f}, searching from Z{z_top:.0f} to Z{z_floor:.0f}")
    hit = touch_down(arm, x, y, z_top, z_floor, stop)
    if hit is None and expected is not None:
        hit = z_floor + PRESS              # counted top reached: grab there (no squashing)
        print(f"  {count} bag(s) in the stack -> top at Z{expected:.1f}; grabbing at Z{hit - PRESS:.1f}")
    elif hit is None:
        hit = z_floor + PRESS              # nothing felt: grab at the bottom (the camera tells if a bag came)
        print(f"  no sheet felt down to Z{z_floor:.0f} - grabbing at the bottom of the stack")
    else:
        print(f"  stack top felt at Z{hit:.1f} ({hit - z_taught:+.1f} mm vs taught) - GRAB")
    touch_memory[key] = hit
    if count is not None:
        stack_left[key] = count - 1
    save_stack()
    arm.move(x, y, hit - PRESS, feed(PRESS_FEED, speed))
    arm.pump(True)
    time.sleep(SUCTION_DELAY)
    arm.move(x, y, hit + TOUCH_ABOVE, feed(PRESS_FEED, speed))
    arm.move(x, y, max(z_high, hit + TOUCH_ABOVE), 1500)     # straight up before moving sideways
    return hit


def replay(arm, points, speed=SPEED, drop=0.0, stop=None, touch=False, pick_bags=1):
    """Follow the taught path once. drop = mm to lower every GRAB point (stack getting thinner).
    touch = find the stack top with the cup's touch switch instead (any stack height).
    stop = optional function called at every point; it raises to abort (used by the dashboard)."""
    def xyz(p):
        x, y, z = p_xyz(p)
        return x, y, z - drop if (p.get("action") == "grab" and not touch) else z

    if touch:
        # path points around a pick spot would cut through the stack: replace them with a
        # straight touch-down from above (and a straight lift afterwards)
        grabs = [p_xyz(p) for p in points if p.get("action") == "grab"]
        near = lambda p: any(math.hypot(p["x"] - g[0], p["y"] - g[1]) < TOUCH_NEAR and p["z"] < g[2] + TOUCH_MAX_STACK
                             for g in grabs)
        points = [p for p in points if p.get("action") or not near(p)]
        releases = [p_xyz(p) for p in points if p.get("action") == "release"]

    def travel_floor(p):
        """touch-down jobs: lowest allowed height for a travel point (clear of the stack)."""
        if not touch or p.get("action"):
            return None
        if any(math.hypot(p["x"] - r[0], p["y"] - r[1]) < TOUCH_NEAR for r in releases):
            return None                                   # final descent onto the place point
        tops = [touch_memory.get((round(g[0]), round(g[1]))) for g in grabs]
        tops = [t for t in tops if t is not None]
        return (max(tops) if tops else max(g[2] for g in grabs)) + TRAVEL_CLEARANCE

    bad = [i for i, p in enumerate(points) if not arm.reachable(*xyz(p))]
    if bad:
        p = points[bad[0]]
        if p.get("action") == "grab" and drop:
            raise StackTooLow(f"the pick height is now {drop:.1f} mm below the taught one and the "
                              f"arm cannot reach that low - is the stack empty?")
        raise UArmError(f"{len(bad)} taught point(s) are outside the arm's reach "
                        f"(first: #{bad[0] + 1} {p_xyz(p)}) - teach again closer in")

    def lifted(p):
        """xyz of a point, raised to the travel height if needed (as high as the arm can reach)."""
        x, y, z = xyz(p)
        floor = travel_floor(p)
        if floor is not None and z < floor:
            zz = floor
            while zz > z and not arm.reachable(x, y, zz):
                zz -= 2
            z = max(z, zz)
        return x, y, z

    if not (touch and points[0].get("action") == "grab"):     # touch: the pick is approached from above
        print(f"  going to start point {p_xyz(points[0])}")
        arm.move(*lifted(points[0]), feed(TRAVEL_FEED, speed))
    n_act = 0
    for i, p in enumerate(points):
        if stop:
            stop()
        x, y, z = lifted(p)                               # stay above the bags while travelling
        if i > 0 and not (touch and p.get("action") == "grab"):
            arm.cmd(f"G0 X{x:.2f} Y{y:.2f} Z{z:.2f} F{feed(p.get('f', 2000), speed):.0f}",
                    timeout=60)
        act = p.get("action")
        if act:
            arm.wait_idle()
            n_act += 1
            if act == "grab" and touch:
                touch_pick(arm, x, y, z, speed, stop, pick_bags)
            elif act == "grab":
                print(f"  GRAB    at X{x:.1f} Y{y:.1f} Z{z:.1f}")
                arm.move(x, y, z - PRESS, feed(PRESS_FEED, speed))
                arm.pump(True)
                time.sleep(SUCTION_DELAY)
                arm.move(x, y, z, feed(PRESS_FEED, speed))
            else:
                print(f"  RELEASE at X{x:.1f} Y{y:.1f} Z{z:.1f}")
                arm.pump(False)
                time.sleep(RELEASE_DELAY)
    arm.wait_idle()
    print(f"  path finished ({len(points)} points, {n_act} pump actions)")


def run_job(arm, a, job):
    """Run a job in batches: ask how many pick-ups, do them, ask again."""
    points = job["points"]
    step = a.step if a.step is not None else job.get("stack_step_mm", 0)
    print(f"\n######## JOB {a.job} ########  speed x{a.speed}, "
          + (f"pick height goes {step:g} mm lower after each pick-up" if step else "same pick height every time"))
    taken = a.taken
    if step and taken is None:
        taken = int(ask_number("Sheets already taken from this stack? (0 = full stack) [0]: ", 0))
    taken = taken or 0

    arm.attach()
    travel = feed(TRAVEL_FEED, a.speed)
    arm.move(*SAFE_POSE, travel)
    total = 0
    count = a.count
    while True:
        if count is None:
            hint = ", r = stack refilled" if step and taken else ""
            ans = input(f"\nHow many {'more ' if total else ''}pick-ups? (Enter = finish{hint}) > ").strip().lower()
            if not ans:
                break
            if ans == "r" and step:
                taken = 0
                print("   stack refilled - next pick-up is at the taught (top) height again")
                continue
            if not ans.isdigit() or int(ans) == 0:
                print("   type a number, e.g. 5")
                continue
            count = int(ans)
        done = 0
        for i in range(count):
            drop = taken * step
            print(f"\n=== JOB {a.job}: pick-up {i + 1}/{count}"
                  + (f"  (sheet {taken + 1} of the stack, {drop:.1f} mm lower)" if step else "") + " ===")
            t0 = time.time()
            try:
                replay(arm, points, a.speed, drop, touch=bool(job.get("touch")), pick_bags=job.get("pick_bags", 1))
            except StackTooLow as e:
                print(f"  STOPPED: {e}")
                if a.count is not None:
                    raise
                print("  Refill the stack and answer 'r', or press Enter to finish.")
                break
            arm.move(*SAFE_POSE, travel)
            taken += 1
            total += 1
            done += 1
            print(f"  done in {time.time() - t0:.1f}s")
            if a.pause and i + 1 < count:
                input("Put the next sheet in place, then press Enter...")
        print(f"\nBatch finished: {done} of {count} pick-up(s). Total this run: {total}"
              + (f", {taken} sheet(s) taken from the stack" if step else ""))
        if a.count is not None:
            break
        count = None
    print("Done - the arm is at home.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=PORT)
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("info")
    sub.add_parser("home")
    sub.add_parser("jobs")
    ln = sub.add_parser("learn")
    ln.add_argument("job", type=int, nargs="?", default=1, help="job number (default 1)")
    rp = sub.add_parser("replay")
    rp.add_argument("job", type=int, nargs="?", default=1, help="job number (default 1)")
    rp.add_argument("--count", type=int, help="number of pick-ups (otherwise it asks)")
    rp.add_argument("--step", type=float, help="mm lower after each pick-up (overrides the job)")
    rp.add_argument("--taken", type=int, help="sheets already taken from the stack")
    rp.add_argument("--pause", action="store_true", help="wait for Enter between pick-ups")
    rp.add_argument("--speed", type=float, default=SPEED,
                    help=f"1.0 = taught speed, 1.5 = 50%% faster, 0.5 = half (default {SPEED})")
    m = sub.add_parser("move")
    m.add_argument("xyz", type=float, nargs=3)
    m.add_argument("speed", type=int, nargs="?", default=3000)
    p = sub.add_parser("pump")
    p.add_argument("state", choices=["on", "off"])
    a = ap.parse_args()

    if a.action == "jobs":
        jobs = list_jobs()
        if not jobs:
            print("No jobs taught yet - run:  python3 uarm_pick_place.py learn 1")
        for j, prog in jobs:
            step = prog.get("stack_step_mm", 0)
            acts = [p["action"] for p in prog["points"] if "action" in p]
            print(f"  job {j}: {len(prog['points'])} points, {acts.count('grab')} pick / "
                  f"{acts.count('release')} place, taught {prog.get('taught', '?')}, "
                  + ("touch-down pick (finds the stack height)" if prog.get("touch") else
                     f"goes {step:g} mm lower after each pick-up" if step else "same height every time"))
        return
    job = load_job(a.job) if a.action == "replay" else None
    arm = SwiftPro(a.port)
    try:
        if a.action == "info":
            for k, v in arm.info().items():
                print(f"{k:>9}: {v}")
        elif a.action == "home":
            arm.attach()
            arm.move(*SAFE_POSE)
        elif a.action == "move":
            arm.attach()
            arm.move(*a.xyz, a.speed)
            print("now at", tuple(round(v, 1) for v in arm.position()))
        elif a.action == "pump":
            arm.pump(a.state == "on")
            print("pump", a.state)
        elif a.action == "learn":
            learn(arm, a.job)
        elif a.action == "replay":
            run_job(arm, a, job)
    except KeyboardInterrupt:
        print("\nInterrupted - turning pump off.")
    except UArmError as e:
        print("\nERROR:", e)
    finally:
        try:
            arm.pump(False)
        except Exception:
            pass
        arm.close()


if __name__ == "__main__":
    main()
