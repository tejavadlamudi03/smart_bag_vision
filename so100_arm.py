#!/usr/bin/env python3
"""
SO-100 / SO-101 arm (6x Feetech STS3215) - move the gripper to X/Y/Z coordinates.

  python3 so100_arm.py info                        # motors: position, voltage, temperature
  python3 so100_arm.py calibrate                   # ONE TIME: touch points on a folded A4 sheet
  python3 so100_arm.py where [--live]              # gripper position in mm (works with motors off)
  python3 so100_arm.py move X Y Z [--speed 40] [--pitch -90] [--direct]
  python3 so100_arm.py gripper open|close|PERCENT
  python3 so100_arm.py park                        # fold into the rest pose, then motors off
  python3 so100_arm.py free                        # motors off - HOLD THE ARM, it will drop
  python3 so100_arm.py learn 1                     # teach job 1 by hand (G = grip, R = release)
  python3 so100_arm.py replay 1 [--count N]        # repeat job 1 (asks how many runs)
  python3 so100_arm.py leader-calibrate             # once: move each LEADER joint to both ends
  python3 so100_arm.py --arm 2 learn 5 --leader    # teach job 5 by moving the LEADER arm
  python3 so100_arm.py jobs                        # list taught jobs
  python3 so100_arm.py set-rest                    # remember the folded pose (used by 'park')
  python3 so100_arm.py --arm 2 name 2                # give the connected arm a short name
  python3 so100_arm.py --arm 2 copy-jobs --from 1    # convert arm 1's jobs for arm 2
Several arms connected? Add --arm with the arm's name, e.g. --arm 1 (see 'jobs').
(learn / replay / gripper / park / free / info need NO calibration)

Coordinates (after calibrate), in mm from the CENTRE of the robot base, axes along the
calibration sheet: X = forward (away from the robot), Y = robot's left (+) / right (-),
Z = height above the table.
--pitch: gripper angle, -90 = pointing straight down (default), "free" = any angle.
"""
import argparse
import json
import math
import os
import sys
import threading
import time
import xml.etree.ElementTree as ET
from itertools import permutations, product

import numpy as np
import serial
from serial.tools import list_ports

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "so101_follower.urdf")
CALIB_FILE = os.path.join(HERE, "so100_calib.json")

IDS = [1, 2, 3, 4, 5, 6]
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TICK = 2 * math.pi / 4096           # STS3215: 4096 steps per turn
ADDR = dict(min=9, max=11, vmax=14, torque=40, acc=41, goal=42, speed=46, pos=56, load=60, volt=62, temp=63,
            moving=66)
SAFE_Z = 80        # mm - travel height for moves that lift over the table
ACCEL = 30         # servo acceleration register (x100 steps/s^2)
SERVO_SPEED = 1000  # servo speed cap, steps/s (~88 deg/s); path speed is set by streaming


class ArmError(RuntimeError):
    pass


# --- Feetech bus ----------------------------------------------------------------
def arm_names():
    """{adapter serial: name} as set with the 'name' command (e.g. '1', '2')."""
    if not os.path.exists(SETTINGS_FILE):
        return {}
    with open(SETTINGS_FILE) as f:
        return {k: v["name"] for k, v in json.load(f).items() if v.get("name")}


def resolve_arm(arm, serials):
    """Serials (from `serials`) meant by `arm`: its name ('1') or the end of its serial ('3529')."""
    by_name = [k for k, v in arm_names().items() if v == arm]
    if by_name:
        return [s for s in serials if s == by_name[0]]
    return [s for s in serials if str(s).endswith(arm)]


def label(serial):
    name = arm_names().get(str(serial))
    return f"arm {name} ({serial})" if name else f"arm {serial}"


def find_port(arm=None):
    """The SO-100's CH343 servo-bus adapter (1a86:55d3) - never the uArm's Arduino.
    arm = its name ('1', '2') or the end of its adapter serial, when several are connected."""
    adapters = [p for p in list_ports.comports() if p.vid == 0x1A86 and p.pid == 0x55D3]
    if arm:
        keep = resolve_arm(arm, [p.serial_number for p in adapters])
        adapters = [p for p in adapters if p.serial_number in keep]
        if not adapters:
            raise SystemExit(f"SO-100 '{arm}' is not connected.")
    if not adapters:
        raise SystemExit("SO-100 arm not found - is its USB cable plugged in?")
    if len(adapters) > 1:
        names = arm_names()
        raise SystemExit("More than one SO-100 is connected: "
                         + ", ".join(f"{label(p.serial_number)} on {p.device}" for p in adapters)
                         + "\nChoose one with --arm, e.g.  python3 so100_arm.py --arm "
                         + f"{names.get(adapters[0].serial_number, str(adapters[0].serial_number)[-4:])} info")
    return adapters[0].device


def adapter_serial(port):
    for p in list_ports.comports():
        if p.device == port:
            return p.serial_number
    return None


class Bus:
    def __init__(self, port=None, arm=None):
        self.port = port or find_port(arm)
        self.serial = adapter_serial(self.port)
        self.s = serial.Serial(self.port, 1_000_000, timeout=0.02)
        self._io = threading.RLock()      # one request/reply on the bus at a time (thread-safe)

    def _send(self, sid, instr, params=()):
        body = [sid, len(params) + 2, instr, *params]
        self.s.reset_input_buffer()
        self.s.write(bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF]))

    def _reply(self, sid, n):
        raw = self.s.read(6 + n)
        if (len(raw) != 6 + n or raw[:2] != b"\xff\xff" or raw[2] != sid
                or (~sum(raw[2:-1])) & 0xFF != raw[-1]):
            return None
        return raw[5:5 + n]

    def read(self, sid, addr, n):
        with self._io:
            return self._read(sid, addr, n)

    def _read(self, sid, addr, n):
        for _ in range(3):
            self._send(sid, 0x02, (addr, n))
            data = self._reply(sid, n)
            if data is not None:
                return data
        raise ArmError(f"motor {sid} does not answer - check the 12 V power and cables")

    def read_u16(self, sid, addr):
        d = self.read(sid, addr, 2)
        return d[0] | d[1] << 8

    def write(self, sid, addr, data):
        with self._io:
            self._send(sid, 0x03, (addr, *data))
            self._reply(sid, 0)      # status packet (ignored if the servo is set not to reply)

    def sync_write(self, addr, items):
        """Write the same register on several motors in one packet: items = {id: bytes}."""
        n = len(next(iter(items.values())))
        params = [addr, n]
        for sid, data in items.items():
            params += [sid, *data]
        with self._io:
            self._send(0xFE, 0x83, params)


u16 = lambda v: [int(v) & 0xFF, (int(v) >> 8) & 0xFF]


class Arm:
    def __init__(self, port=None, arm=None):
        self.bus = Bus(port, arm)
        self.limits = {i: (self.bus.read_u16(i, ADDR["min"]), self.bus.read_u16(i, ADDR["max"]))
                       for i in IDS}
        st = load_settings(self)                # remember this arm's end stops (for copy-jobs)
        if st.get("limits") != [list(self.limits[i]) for i in IDS]:
            st["limits"] = [list(self.limits[i]) for i in IDS]
            save_settings(self, st)

    def raw(self):
        return [self.bus.read_u16(i, ADDR["pos"]) for i in IDS]

    def raw_avg(self, n=10):
        return [float(v) for v in np.mean([self.raw() for _ in range(n)], axis=0)]

    def torque_on(self, ids=IDS):
        volt = self.bus.read(1, ADDR["volt"], 1)[0] / 10
        if volt < 4.5:          # 5 V (7.4 V motors) and 12 V (12 V motors) supplies are both fine
            raise ArmError(f"servo supply is only {volt:.1f} V - is the arm's power adapter plugged in?")
        vmax = self.bus.read(1, ADDR["vmax"], 1)[0] / 10
        if volt > vmax:         # e.g. 7.4 V motors (limit 8 V) on a 12 V adapter - would overheat them
            raise ArmError(f"servo supply is {volt:.1f} V but these motors are set for at most {vmax:.1f} V - "
                           "wrong power adapter? (7.4 V motors need a 5 V supply)")
        here = dict(zip(IDS, self.raw()))
        # Goal = where each motor is NOW, so switching torque on does not make it jump
        # (the goal registers hold 0 after power-up -> the arm would slam to position 0).
        self.bus.sync_write(ADDR["goal"], {i: u16(here[i]) for i in ids})
        self.bus.sync_write(ADDR["acc"], {i: [ACCEL] for i in ids})
        self.bus.sync_write(ADDR["speed"], {i: u16(SERVO_SPEED) for i in ids})
        self.bus.sync_write(ADDR["torque"], {i: [1] for i in ids})
        if any(self.bus.read(i, ADDR["torque"], 1)[0] != 1 for i in ids):
            raise ArmError("could not switch the motors on")

    def torque_off(self, ids=IDS):
        self.bus.sync_write(ADDR["torque"], {i: [0] for i in ids})

    def torque_is_on(self):
        return all(self.bus.read(i, ADDR["torque"], 1)[0] == 1 for i in IDS)

    def set_goal(self, raw_by_id):
        """raw_by_id = {motor id: target position}; clamped to each motor's safe range."""
        items = {}
        for i, v in raw_by_id.items():
            lo, hi = self.limits[i]
            items[i] = u16(round(min(max(v, lo), hi)))
        self.bus.sync_write(ADDR["goal"], items)

    def wait_still(self, timeout=4.0):
        """Wait until no motor moves more than 1 step between reads."""
        last, still, t_end = self.raw(), 0, time.time() + timeout
        while time.time() < t_end:
            time.sleep(0.05)
            now = self.raw()
            still = still + 1 if max(abs(a - b) for a, b in zip(now, last)) <= 1 else 0
            last = now
            if still >= 3:
                return now
        return last


# --- kinematics (from the SO-101 URDF; same link lengths as the SO-100) -------------
def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Kinematics:
    def __init__(self, urdf=URDF):
        root = ET.parse(urdf).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], "gripperframe"
        while link in by_child:
            j = by_child[link]
            chain.append(j)
            link = j.find("parent").get("link")
        self.chain = []
        for j in reversed(chain):                       # base -> tool
            o = j.find("origin")
            T = np.eye(4)
            T[:3, :3] = _rpy(*map(float, o.get("rpy").split()))
            T[:3, 3] = list(map(float, o.get("xyz").split()))
            self.chain.append((T, j.get("type") == "revolute", j.get("name")))
        self.limits = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
                       for j in root.findall("joint") if j.get("type") == "revolute"}

    def fk(self, q, tool=0.0):
        """q = 5 joint angles (rad). Returns tool point (mm, base frame) and gripper direction.
        tool = extra length (mm) along the gripper beyond the URDF tool point."""
        T, k, frames = np.eye(4), 0, {}
        for Tj, revolute, name in self.chain:
            T = T @ Tj
            if revolute and name in ("1", "2", "3", "4", "5"):
                R = np.eye(4)
                R[:3, :3] = _rz(q[k])
                T = T @ R
                k += 1
            frames[name] = T
        g = frames["5"]                                  # gripper link: its z axis = roll axis
        direction = -g[:3, 2]                            # points from the wrist towards the tip
        self.last_joints = {n: frames[n][:3, 3] * 1000 for n in ("2", "3", "4")}  # shoulder/elbow/wrist
        return T[:3, 3] * 1000 + direction * tool, direction


# --- calibration: raw servo steps -> joint angles, robot -> table frame ----------------
class Model:
    """q_i = sign_i * (raw_i - 2048) * TICK + offset_i ; table = Rz(yaw) * base + t"""

    def __init__(self, kin, c):
        self.kin, self.c = kin, c

    def q(self, raw):
        return [s * (r - 2048) * TICK + o for s, r, o in zip(self.c["sign"], raw[:5], self.c["offset"])]

    def raw_from_q(self, q):
        return [2048 + (qi - o) / (s * TICK) for qi, s, o in zip(q, self.c["sign"], self.c["offset"])]

    def to_table(self, p):
        return _rz(self.c["yaw"]) @ p + np.array(self.c["t"])

    def to_base(self, p):
        return _rz(-self.c["yaw"]) @ (np.asarray(p, float) - np.array(self.c["t"]))

    def tool(self, q):
        p, d = self.kin.fk(q, self.c["tool"])
        return self.to_table(p), math.degrees(math.asin(max(-1, min(1, d[2]))))


def lm(residual, x0, iters=60, lo=None, hi=None):
    """Small Levenberg-Marquardt least-squares solver (numerical Jacobian)."""
    x = np.array(x0, float)
    r = residual(x)
    cost, mu = r @ r, 1e-3
    for _ in range(iters):
        J = np.empty((len(r), len(x)))
        for k in range(len(x)):
            dx = np.zeros_like(x)
            dx[k] = 1e-6
            J[:, k] = (residual(x + dx) - r) / 1e-6
        A, g = J.T @ J, J.T @ r
        while True:
            step = -np.linalg.solve(A + mu * np.diag(np.diag(A) + 1e-9), g)
            xn = x + step
            if lo is not None:
                xn = np.clip(xn, lo, hi)
            rn = residual(xn)
            if rn @ rn < cost:
                x, r, cost, mu = xn, rn, rn @ rn, max(mu / 3, 1e-9)
                break
            mu *= 4
            if mu > 1e8:
                return x, cost
        if np.abs(step).max() < 1e-9:
            break
    return x, cost


def fit_calibration(kin, samples, servo_limits):
    """samples = [(raw[6], table_xyz_mm)]. Finds joint signs/offsets, tool length and where
    the robot stands on the table. Pan offset is folded into the yaw (they are the same thing)."""
    raws = [s[0] for s in samples]
    pts = np.array([s[1] for s in samples], float)
    names = ["1", "2", "3", "4", "5"]
    starts = []
    for signs in product([1, -1], repeat=5):
        # Start: servo end stops <-> URDF joint limits (centre of one = centre of the other).
        off0 = []
        for k, n in enumerate(names):
            lo, hi = kin.limits[n]
            smin, smax = servo_limits[k + 1]
            off0.append((lo + hi) / 2 - signs[k] * ((smin + smax) / 2 - 2048) * TICK)

        def model_points(x, signs=signs, off1=off0[0], physics=None):
            c = dict(sign=signs, offset=[off1, *x[0:4]], tool=x[4], yaw=x[5], t=list(x[6:9]))
            m = Model(kin, c)
            out = []
            for r in raws:
                out.append(m.tool(m.q(r))[0])
                if physics is not None:
                    # elbow and wrist can never be below the table (removes the mirror-image
                    # "arm under the table" solution that fits flat touch points equally well)
                    for n in ("3", "4"):
                        physics.append(max(0.0, 10 - m.to_table(kin.last_joints[n])[2]))
            if physics is not None:
                # the base stands on the table: its origin is at table height (+-40 mm)
                bz = m.to_table(np.zeros(3))[2]
                physics.append(max(0.0, abs(bz) - 40))
            return np.array(out)

        def residual(x, model_points=model_points, off5=off0[4]):
            physics = []
            err = (model_points(x, physics=physics) - pts).ravel()
            # weak priors keep the tool length and wrist-roll offset sensible
            return np.concatenate([err, 3 * np.array(physics), [0.2 * x[4], 20 * (x[3] - off5)]])

        for d2, d3 in [(0, 0), (0.3, -0.3), (-0.3, 0.3)]:
            x0 = np.array([off0[1] + d2, off0[2] + d3, off0[3], off0[4], 0, 0, 0, 0, 0], float)
            # yaw + translation from a 2D best fit of the model's points onto the marked points
            mp = model_points(x0)
            a, b = mp[:, :2] - mp[:, :2].mean(0), pts[:, :2] - pts[:, :2].mean(0)
            x0[5] = math.atan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
            x0[6:9] = pts.mean(0) - model_points(x0).mean(0)
            starts.append((residual, x0, signs, off0[0]))

    # Quick pass over every start, then polish the 4 most promising ones.
    quick = sorted(((lm(r, x0, iters=12)[1], k) for k, (r, x0, _, _) in enumerate(starts)))
    best = None
    for _, k in quick[:4]:
        r, x0, signs, off1 = starts[k]
        x, cost = lm(r, x0, iters=150)
        if best is None or cost < best[0]:
            best = (cost, signs, off1, x)
    cost, signs, off1, x = best
    c = dict(sign=list(signs), offset=[off1, *map(float, x[0:4])], tool=float(x[4]),
             yaw=float(x[5]), t=[float(v) for v in x[6:9]])
    m = Model(kin, c)
    errs = [float(np.linalg.norm(m.tool(m.q(r))[0] - p)) for r, p in zip(raws, pts)]
    return c, errs


# --- inverse kinematics + motion -----------------------------------------------------
class Controller:
    def __init__(self, arm, model):
        self.arm, self.m = arm, model
        a = np.array(model.q([arm.limits[i][0] for i in IDS]))   # joint angles at the end stops
        b = np.array(model.q([arm.limits[i][1] for i in IDS]))
        self.qlo, self.qhi = np.minimum(a, b), np.maximum(a, b)

    def current(self):
        raw = self.arm.raw()
        q = self.m.q(raw)
        p, pitch = self.m.tool(q)
        return raw, q, p, pitch

    def ik(self, target, pitch, q_seed):
        """Joints 1-4 for the tool point at `target` (table mm); wrist roll stays as it is.
        pitch in degrees (-90 = straight down) or None for any angle."""
        q5 = q_seed[4]

        def residual(x):
            q = [*x, q5]
            p, ptc = self.m.tool(q)
            r = list(p - target)
            r.append(0 if pitch is None else (ptc - pitch))
            r += list(0.02 * np.degrees(np.array(x) - q_seed[:4]))   # stay close to the seed
            return np.array(r)

        best = None
        for seed in (q_seed[:4], (self.qlo[:4] + self.qhi[:4]) / 2):
            x, _ = lm(residual, seed, iters=60, lo=self.qlo[:4], hi=self.qhi[:4])
            q = [*x, q5]
            p, ptc = self.m.tool(q)
            err = float(np.linalg.norm(p - target))
            perr = 0 if pitch is None else abs(ptc - pitch)
            score = err + perr
            if best is None or score < best[0]:
                best = (score, q, err, perr, ptc)
        return best[1], best[2], best[3], best[4]

    def line(self, q0, target, pitch, dt, step_mm=3.0):
        """Straight line from the pose q0 to target; IK every 3 mm. None if any point fails."""
        p0, pitch0 = self.m.tool(q0)
        n = max(1, int(np.linalg.norm(target - p0) / step_mm))
        steps, q = [], list(q0)
        for k in range(1, n + 1):
            p = p0 + (target - p0) * k / n
            ptc = None if pitch is None else pitch0 + (pitch - pitch0) * min(1, 3 * k / n)
            q, err, perr, _ = self.ik(p, ptc, q)
            if err > 3 or perr > 5 or p[2] < -3:
                return None
            steps.append((q, dt))
        return steps

    def joint_move(self, q0, q1, max_step=15):
        """Joint-space move q0 -> q1 (max 15 servo steps per 1/30 s ~ 40 deg/s). Tries all
        joints together and each joint one after another in every order, and keeps the
        option whose gripper stays highest. None if every option dips below the table."""
        q0, q1 = np.array(q0, float), np.array(q1, float)
        per_tick = max_step * TICK
        z_floor = min(self.m.tool(list(q0))[0][2], 0) - 3
        options = [[[0, 1, 2, 3]]] + [[[j] for j in order] for order in permutations(range(4))]
        best = None
        for groups in options:
            q, steps, low = q0.copy(), [], np.inf
            for g in groups:
                start = q.copy()
                n = max(1, int(np.abs(q1[g] - start[g]).max() / per_tick))
                for k in range(1, n + 1):
                    q = start.copy()
                    q[g] = start[g] + (q1[g] - start[g]) * k / n
                    low = min(low, self.m.tool(list(q))[0][2])
                    steps.append((list(q), 1 / 30))
            if low >= z_floor and (best is None or low > best[0] + 20):
                best = (low, steps)       # all-together wins unless an order is 20 mm safer
        return best[1] if best else None

    def plan(self, target, pitch, q0, speed, direct=False):
        """Steps [(q, seconds)] to reach target, or None. Nothing moves while planning.
        Normal moves go up to SAFE_Z, across, and straight down onto the target."""
        dt = 3.0 / speed
        p0, _ = self.m.tool(q0)
        if direct:
            return self.line(q0, target, pitch, dt)
        hz = max(SAFE_Z, p0[2], target[2])
        above_here, above_target = np.array([p0[0], p0[1], hz]), np.array([target[0], target[1], hz])
        steps = self.line(q0, above_here, pitch, dt)
        if steps:
            across = self.line(steps[-1][0], above_target, pitch, dt)
            steps = steps + across if across else None
        if not steps:
            # e.g. starting from the folded rest pose (joints on their end stops): unfold with a
            # joint move straight to the point above the target instead.
            q_above, err, perr, _ = self.ik(above_target, pitch, q0)
            if err > 3 or perr > 5:
                return None
            steps = self.joint_move(q0, q_above)
            if steps is None:
                return None
        down = self.line(steps[-1][0], target, pitch, dt)
        return steps + down if down is not None else None

    def run(self, steps):
        for q, dt in steps:
            t = time.time()
            raw = self.m.raw_from_q(q)
            self.arm.set_goal({i: raw[k] for k, i in enumerate(IDS[:5])})
            time.sleep(max(0.0, dt - (time.time() - t)))
        # Gravity makes the arm sag a little: nudge each joint until it reads its target.
        goal = self.m.raw_from_q(steps[-1][0])
        cmd = list(goal)
        for _ in range(3):
            now = self.arm.wait_still()
            err = [g - n for g, n in zip(goal, now[:5])]
            if max(abs(e) for e in err) <= 3:
                break
            cmd = [c + max(-60, min(60, e)) for c, e in zip(cmd, err)]
            self.arm.set_goal({i: cmd[k] for k, i in enumerate(IDS[:5])})


# --- commands --------------------------------------------------------------------
def load_model(kin, arm):
    if not os.path.exists(CALIB_FILE):
        raise SystemExit("Not calibrated yet - run:  python3 so100_arm.py calibrate")
    with open(CALIB_FILE) as f:
        c = json.load(f)
    if c.get("arm_serial") and arm.bus.serial and c["arm_serial"] != arm.bus.serial:
        raise SystemExit(f"The calibration belongs to the SO-100 with adapter {c['arm_serial']}, but the "
                         f"arm connected now is {arm.bus.serial} - run calibrate on this arm.")
    return Model(kin, c)


def cmd_info(arm):
    print(f"{label(arm.bus.serial)} on {arm.bus.port}")
    print(f"{'ID':<3} {'joint':<14} {'pos':>5} {'range':>11} {'torque':>6} {'volt':>5} {'temp':>5}")
    for i, n in zip(IDS, NAMES):
        b = arm.bus
        print(f"{i:<3} {n:<14} {b.read_u16(i, ADDR['pos']):>5} {arm.limits[i][0]:>5}-{arm.limits[i][1]:<5} "
              f"{'on' if b.read(i, ADDR['torque'], 1)[0] else 'off':>6} {b.read(i, ADDR['volt'], 1)[0] / 10:>5} "
              f"{b.read(i, ADDR['temp'], 1)[0]:>4}C")


PAPERS = {"a4": (297.0, 210.0), "letter": (279.4, 215.9)}   # long side, short side (mm)


def sheet_points(paper):
    """Touch list. Sheet frame: origin = middle of the edge nearest the robot,
    X = away from the robot, Y = to the robot's left. Lying sideways (long side facing the robot)."""
    W, D = PAPERS[paper]
    L = "robot's LEFT (your right when you face the robot)"
    R = "robot's RIGHT (your left when you face the robot)"
    return [
        ("near-middle", (0, 0), "the NEAR edge (closest to the robot), where the fold line meets it"),
        ("near-left", (0, W / 2), f"the NEAR corner on the {L}"),
        ("near-right", (0, -W / 2), f"the NEAR corner on the {R}"),
        ("centre", (D / 2, 0), "the CENTRE, where the two fold lines cross"),
        ("left-middle", (D / 2, W / 2), f"the edge on the {L}, where the fold line meets it"),
        ("right-middle", (D / 2, -W / 2), f"the edge on the {R}, where the fold line meets it"),
        ("far-middle", (D, 0), "the FAR edge, where the fold line meets it"),
        ("far-left", (D, W / 2), f"the FAR corner on the {L}  (s = skip if it can't reach)"),
        ("far-right", (D, -W / 2), f"the FAR corner on the {R}  (s = skip if it can't reach)"),
        ("centre", (D / 2, 0), "the CENTRE again - this time with the gripper LEANING (tilted)"),
        ("near-middle", (0, 0), "the NEAR-edge middle again - gripper LEANING"),
        ("centre", (D / 2, 0), "the CENTRE again - gripper straight DOWN, wrist TURNED a quarter turn"),
        ("centre", (D / 2, 0), "the CENTRE again - gripper straight DOWN, wrist turned a quarter turn the OTHER way"),
    ]


def cmd_calibrate(arm, kin, paper):
    W, D = PAPERS[paper]
    print(f"""
CALIBRATION WITH A SHEET OF PAPER ({paper.upper()}, {W:g} x {D:g} mm) - no measuring needed
  1. Fold the sheet in half, unfold; fold it in half the other way, unfold.
     The two fold lines mark the middle of every edge and the centre.
  2. Lay it flat in front of the robot, LONG side facing the robot, about a hand's
     width (~10 cm) from the base and roughly centred. Tape the corners down.
     The robot's X axis will run along the sheet (straight away from the robot).
  3. The motors are switched OFF so you can move the arm by hand.
     CLOSE the gripper so the two fingertips touch - that is the point you touch with.
  4. For each step: put the fingertips exactly on the point, hold still, press Enter.
     s = skip a point the arm cannot reach, q = stop.
""")
    if arm.torque_is_on():
        input("Hold the arm (motors are about to switch off), then press Enter...")
    arm.torque_off()
    samples = []
    steps = sheet_points(paper)
    for k, (name, (sx, sy), what) in enumerate(steps, 1):
        ans = input(f"[{k}/{len(steps)}] Touch {what}\n        Enter = recorded, s = skip, q = stop > ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            continue
        samples.append((arm.raw_avg(), (sx, sy, 0.0)))
        print(f"        ok ({len(samples)} touches)")
    if len(samples) < 7:
        print(f"Only {len(samples)} touches - at least 7 are needed. Run calibrate again.")
        return

    print("\nWorking out the calibration (about 10 s)...")
    c, errs = fit_calibration(kin, samples, arm.limits)
    worst = int(np.argmax(errs))
    if len(samples) >= 9 and errs[worst] > 3 * max(np.median(errs), 2):
        print(f"   touch #{worst + 1} looks wrong ({errs[worst]:.0f} mm off) - working it out again without it")
        samples.pop(worst)
        c, errs = fit_calibration(kin, samples, arm.limits)
    rms = float(np.sqrt(np.mean(np.square(errs))))
    for k, e in enumerate(errs, 1):
        print(f"   touch #{k:<2} off by {e:4.1f} mm")
    print(f"   average {rms:.1f} mm")
    if rms > 6:
        print("   That is too large - check the left/right corners and that the sheet is flat,"
              " then run calibrate again.")
        return

    # Fitted frame = the sheet. Move the origin to the centre of the robot base (axes stay
    # along the sheet), so coordinates are "mm from the robot".
    pan = kin.chain[0][0][:3, 3] * 1000                     # base rotation axis, URDF base frame
    o = Model(kin, c).to_table(np.array([pan[0], pan[1], 0.0]))
    c["t"] = [c["t"][0] - o[0], c["t"][1] - o[1], c["t"][2]]
    c.update(method=f"sheet {paper}", rms_mm=rms, date=time.strftime("%Y-%m-%d %H:%M"),
             arm_serial=arm.bus.serial,
             gripper_closed=float(np.median([r[5] for r, _ in samples])),   # closed during calibration
             sheet_near_edge_mm=round(float(-o[0]), 1),
             points=[{"raw": r, "sheet_xyz": list(p)} for r, p in samples])
    with open(CALIB_FILE, "w") as f:
        json.dump(c, f, indent=1)
    print(f"Saved -> {CALIB_FILE}")
    print(f"The sheet's near edge is {-o[0]:.0f} mm in front of the robot base centre.")
    print("Check it:  python3 so100_arm.py where --live   (move the arm by hand onto the sheet centre:\n"
          f"           it should read about X {-o[0] + D / 2:.0f}  Y {-o[1]:.0f}  Z 0)")


def cmd_where(arm, kin, live):
    m = load_model(kin, arm)
    while True:
        p, pitch = m.tool(m.q(arm.raw()))
        line = f"X {p[0]:7.1f}  Y {p[1]:7.1f}  Z {p[2]:7.1f} mm   gripper pitch {pitch:6.1f} deg"
        if not live:
            print(line)
            return
        print("\r" + line + "   (Ctrl+C to stop)", end="", flush=True)
        time.sleep(0.1)


def cmd_move(arm, kin, target, speed, pitch, direct):
    ctl = Controller(arm, load_model(kin, arm))
    raw, q0, p0, pitch0 = ctl.current()
    target = np.array(target, float)
    print(f"from X{p0[0]:.0f} Y{p0[1]:.0f} Z{p0[2]:.0f}  to X{target[0]:.0f} Y{target[1]:.0f} Z{target[2]:.0f}")
    steps = ctl.plan(target, pitch, q0, speed, direct)
    if steps is None and pitch is not None:
        print(f"   can't keep the gripper at {pitch:.0f} deg on this path - allowing it to tilt")
        steps = ctl.plan(target, None, q0, speed, direct)
    if steps is None:
        raise ArmError(f"X{target[0]:.0f} Y{target[1]:.0f} Z{target[2]:.0f} is out of reach "
                       f"(or the path would hit the table) - nothing moved")
    if not arm.torque_is_on():
        arm.torque_on()
    ctl.run(steps)
    _, _, p, ptc = ctl.current()
    print(f"now at X{p[0]:.1f} Y{p[1]:.1f} Z{p[2]:.1f}  (off by {np.linalg.norm(p - target):.1f} mm), "
          f"gripper pitch {ptc:.0f} deg")


def load_settings(arm):
    """Settings of THIS arm (keyed by its USB adapter serial): rest pose, gripper direction."""
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            return json.load(f).get(str(arm.bus.serial), {})
    return {}


def save_settings(arm, s):
    allarms = {}
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            allarms = json.load(f)
    allarms[str(arm.bus.serial)] = s
    with open(SETTINGS_FILE, "w") as f:
        json.dump(allarms, f, indent=1)


def gripper_targets(arm):
    """(closed, open) gripper positions. 'closed' squeezes a little past touching the stop
    side, so an object between the fingers is held with force."""
    lo, hi = arm.limits[6]
    st = load_settings(arm)
    side = st.get("gripper_closed_side")
    if side is None:                       # guess: the arm rests with the gripper closed
        g = st.get("rest", arm.raw())[5]
        side = "min" if abs(g - lo) < abs(g - hi) else "max"
    closed = lo + 30 if side == "min" else hi - 30          # stay just off the end stop
    opened = lo + 0.6 * (hi - lo) if side == "min" else hi - 0.6 * (hi - lo)
    return closed, opened


GRIP_LOAD_MAX = 180    # 0.1 % units: hold below 18 % - the gripper switches itself off above
                       # its Overload_Torque (25 % on this arm) after Protection_Time (2 s)


def grip(arm, closed):
    """Close until the fingers touch the object, then hold with a gentle squeeze that stays
    under the gripper's overload limit (a hard squeeze would trip it and drop the object)."""
    lo, hi = arm.limits[6]
    toward = -1 if abs(closed - lo) < abs(closed - hi) else 1     # direction of closing
    arm.set_goal({6: closed})
    time.sleep(GRIP_WAIT)
    pos = arm.bus.read_u16(6, ADDR["pos"])
    if abs(pos - closed) < 20:
        return "nothing between the fingers"
    squeeze = 60
    while True:
        arm.set_goal({6: pos + toward * squeeze})
        time.sleep(0.2)
        load = arm.bus.read_u16(6, ADDR["load"]) & 0x3FF
        if load <= GRIP_LOAD_MAX or squeeze <= 10:
            return f"holding (load {load / 10:.0f} %)"
        squeeze -= 15


def cmd_gripper(arm, how):
    closed, opened = gripper_targets(arm)
    pct = {"open": 100, "close": 0}.get(how)
    pct = float(how) if pct is None else pct
    if not arm.torque_is_on():
        arm.torque_on()
    arm.set_goal({6: closed + (opened - closed) * max(0, min(100, pct)) / 100})
    arm.wait_still(2)
    print(f"gripper {how}")


# --- teach by hand & replay (joint space - needs no calibration) -----------------------
JOBS_DIR = os.path.join(HERE, "so100_jobs")
SETTINGS_FILE = os.path.join(HERE, "so100_settings.json")
GRIP_WAIT = 0.6        # s for the gripper to close / open during replay


class Keys:
    """Single-keypress reader for the Linux terminal (no Enter needed)."""

    def __enter__(self):
        import termios
        import tty
        self.termios = termios
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        termios.tcflush(self.fd, termios.TCIFLUSH)
        return self

    def __exit__(self, *exc):
        self.termios.tcsetattr(self.fd, self.termios.TCSADRAIN, self.old)

    def get(self, timeout=None):
        import select
        if not select.select([sys.stdin], [], [], timeout)[0]:
            return None
        k = os.read(self.fd, 8).decode(errors="ignore")
        return k if k.startswith("\x1b") else k.lower()


LEARN_HELP = """
===================  TEACH THE SO-100 BY HAND  ===================
 The arm goes limp; move it with your hand through the whole job
 (e.g. lift the board on the left, then on the right). It records
 the WHOLE path and later repeats it exactly, at the same speed.
 Keys (no Enter needed):
   S      start recording            ENTER  stop recording & save
   X      throw away & record again  Q      quit without saving
   (optional, only if you use the gripper: G = close, R = open)
===================================================================
"""


def learn_hint(recording, events):
    if not recording:
        return ("STEP 1    Move the arm to where the job should START (e.g. just above the board),\n"
                "          then press  S  to start recording.")
    if events and events[-1][1] == "grip":
        return ("          [REC] gripper is closed - press  R  to open it again.")
    return ("STEP 2    [REC] Move the arm through the whole job at a calm speed:\n"
            "          lift the board on the LEFT, set it down, then lift it on the RIGHT, set it down.\n"
            "STEP 3    Move back to a safe pose and press  ENTER  to stop and save.")


def gripper_check(arm, kb):
    """First time only: make sure 'close' really closes."""
    s = load_settings(arm)
    if s.get("gripper_checked"):
        return
    closed, opened = gripper_targets(arm)
    print("GRIPPER TEST (first time only): the gripper will OPEN, then CLOSE.")
    arm.set_goal({6: opened})
    time.sleep(1.2)
    arm.set_goal({6: closed})
    time.sleep(1.2)
    print("Did the gripper end up CLOSED?  y = yes / n = no (it opened last)")
    while (k := kb.get()) not in ("y", "n"):
        pass
    side = "min" if closed < opened else "max"
    if k == "n":
        side = "max" if side == "min" else "min"
        arm.set_goal({6: gripper_targets(arm)[1]})
    s.update(gripper_closed_side=side, gripper_checked=True)
    save_settings(arm, s)
    print("   ok - remembered.\n")


def job_path(serial, job):
    return os.path.join(JOBS_DIR, str(serial), f"job{job}.json")


def migrate_jobs():
    """Older versions kept all jobs in one folder - move each into its arm's folder."""
    if not os.path.isdir(JOBS_DIR):
        return
    for name in os.listdir(JOBS_DIR):
        src = os.path.join(JOBS_DIR, name)
        if name.startswith("job") and name.endswith(".json") and os.path.isfile(src):
            with open(src) as f:
                serial = json.load(f).get("arm_serial", "unknown")
            os.makedirs(os.path.join(JOBS_DIR, str(serial)), exist_ok=True)
            os.replace(src, os.path.join(JOBS_DIR, str(serial), name))


def cmd_learn(arm, job):
    path = job_path(arm.bus.serial, job)
    print(f"\n######## Teaching SO-100 JOB {job} ########")
    if os.path.exists(path) and input(f"Job {job} already exists - replace it? [y/N] ").strip().lower() != "y":
        print("Kept the old job - nothing changed.")
        return
    print(LEARN_HELP)
    arm_ids = IDS[:5]
    samples, events, recording, gripping, saved, msg = [], [], False, False, False, ""
    closed, opened = gripper_targets(arm)
    with Keys() as kb:
        print("STEP 0    HOLD THE ARM with one hand (it goes limp next), then press any key.")
        kb.get()
        arm.torque_on([6])                 # gripper powered, holds its current opening
        arm.torque_off(arm_ids)            # arm joints limp
        print("\n>> Arm is limp - move it by hand.\n")
        print(learn_hint(recording, events) + "\n")
        last_draw = 0.0
        while True:
            raw = arm.raw()
            now = time.time()
            if recording:
                samples.append((now, *raw[:5]))
            if now - last_draw > 0.1:
                rec = f"[REC {now - samples[0][0]:5.1f}s]" if recording else "[not recording]"
                grp = f"gripper {'CLOSED' if gripping else 'open'}  " if events else ""
                print(f"\r\033[K  {rec}  joints {' '.join(f'{v:5.0f}' for v in raw[:5])}  "
                      f"{grp}{msg}", end="", flush=True)
                last_draw = now
            k = kb.get(0)
            if k is None:
                continue
            msg = ""
            if k == "s" and not recording:
                recording, samples, events = True, [(now, *raw[:5])], []
                print("\r\033[K  * recording started\n")
                print(learn_hint(recording, events) + "\n")
            elif k in ("g", "r"):
                action = "grip" if k == "g" else "release"
                if not recording:
                    msg = "press S to start recording first"
                elif gripping == (action == "grip"):
                    msg = "gripper is already " + ("closed" if gripping else "open")
                else:
                    if not events:                 # first gripper use: make sure 'close' closes
                        print("\r\033[K")
                        gripper_check(arm, kb)
                        closed, opened = gripper_targets(arm)
                    gripping = action == "grip"
                    if gripping:
                        grip(arm, closed)
                    else:
                        arm.set_goal({6: opened})
                    events.append((len(samples) - 1, action))
                    print(f"\r\033[K  * {'GRIP    - gripper closes' if gripping else 'RELEASE - gripper opens'}\n")
                    print(learn_hint(recording, events) + "\n")
            elif k == "x" and recording:
                if events:
                    arm.set_goal({6: opened})
                recording, samples, events, gripping = False, [], [], False
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
        print("\r\033[K\nLAST STEP HOLD THE ARM again, then press any key to lock the motors.")
        kb.get()
        arm.torque_on()
        print(">> Motors ON (holding).")

    if not saved:
        print("\nQuit without saving - nothing changed.")
        return
    job_data = build_joint_job(samples, events)
    job_data.update(job=job, taught=time.strftime("%Y-%m-%d %H:%M"), closed=closed, opened=opened,
                    arm_serial=arm.bus.serial)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(job_data, f, indent=1)
    acts = [a for _, a in events]
    print(f"\nSaved JOB {job} -> {path}")
    print(f"  {samples[-1][0] - samples[0][0]:.1f} s, {len(job_data['points'])} path points, "
          f"{acts.count('grip')} grip(s) / {acts.count('release')} release(s)")
    if gripping:
        print("  NOTE: the recording ended with the gripper still CLOSED.")
    print(f"Next: python3 so100_arm.py replay {job}")


def build_joint_job(samples, events):
    """Smooth (+-0.1 s, never across a grip/release) and thin out the recorded joint path."""
    t = [s[0] - samples[0][0] for s in samples]
    raw = [list(s[1:]) for s in samples]
    n = len(raw)
    cuts = sorted({0, n - 1, *(i for i, _ in events)})
    smooth = [list(r) for r in raw]
    for a, b in zip(cuts, cuts[1:]):
        for k in range(a + 1, b):
            win = [raw[j] for j in range(max(a + 1, k - 10), min(b, k + 11)) if abs(t[j] - t[k]) <= 0.1]
            smooth[k] = [sum(w[i] for w in win) / len(win) for i in range(5)]
    actions = dict(events)
    points, last = [], None
    for k in range(n):
        keep = (last is None or k in cuts or max(abs(a - b) for a, b in zip(smooth[k], last)) >= 2)
        if keep:
            p = {"t": round(t[k], 3), "raw": [round(v, 1) for v in smooth[k]]}
            if k in actions:
                p["action"] = actions[k]
            points.append(p)
            last = smooth[k]
    return {"version": 1, "points": points}


def joint_glide(arm, target5, max_step=8):
    """Slow joint-space move of joints 1-5 to target (about 20 deg/s)."""
    start = arm.raw()[:5]
    n = max(1, int(max(abs(a - b) for a, b in zip(target5, start)) / max_step))
    for k in range(1, n + 1):
        arm.set_goal({i: s + (g - s) * k / n for i, s, g in zip(IDS[:5], start, target5)})
        time.sleep(1 / 30)
    arm.wait_still(3)


# --- teaching with the leader arm ---------------------------------------------------
FOLLOW_STEP = 40       # max ticks (~3.5 deg) the follower goal may run ahead of the follower per update


def find_leader_port(follower_serial, leader=None):
    """The leader arm's adapter: by name/serial digits, else the one named 'leader', else the
    only other SO-100 adapter without a name."""
    others = [p for p in list_ports.comports()
              if p.vid == 0x1A86 and p.pid == 0x55D3 and p.serial_number != follower_serial]
    names = arm_names()
    if leader and leader != "auto":
        keep = resolve_arm(leader, [p.serial_number for p in others])
    else:
        keep = [p.serial_number for p in others if names.get(p.serial_number) == "leader"] or \
               [p.serial_number for p in others if p.serial_number not in names]
    found = [p for p in others if p.serial_number in keep]
    if len(found) != 1:
        seen = ", ".join(label(p.serial_number) for p in others) or "none"
        raise ArmError(f"leader arm not found (other SO-100 adapters connected: {seen}) - plug it in, "
                       "or choose it with --leader NAME")
    return found[0].device


def leader_range(leader):
    """End stops of the leader's joints 1-4: measured with leader-calibrate, else the servo limits."""
    st = load_settings(leader)
    if st.get("leader_range"):
        return [tuple(r) for r in st["leader_range"]]
    regs = [leader.limits[i] for i in IDS[:4]]
    if all(1000 < hi - lo < 4000 for lo, hi in regs):
        return regs
    raise ArmError("the leader's end stops are not known yet - run:  python3 so100_arm.py leader-calibrate")


class Teleop:
    """The follower copies the leader. Joints 1-4 are matched through both arms' end stops (same
    physical pose -> same physical pose); the wrist roll has no end stops and follows the leader's
    turning relative to where both were at the start. The gripper is not used."""

    def __init__(self, leader, follower):
        self.leader, self.follower = leader, follower
        self.lim_l, self.lim_f = leader_range(leader), [follower.limits[i] for i in IDS[:4]]
        self.roll0 = None

    def target(self, lraw):
        out = [b0 + (lraw[k] - a0) * (b1 - b0) / (a1 - a0)
               for k, ((a0, a1), (b0, b1)) in enumerate(zip(self.lim_l, self.lim_f))]
        roll = -(lraw[4] & 0x7FFF) if lraw[4] & 0x8000 else lraw[4]     # bit 15 = negative position
        if self.roll0 is None:
            self.roll0 = (roll, self.follower.raw()[4])
        out.append(self.roll0[1] + (roll - self.roll0[0] + 2048) % 4096 - 2048)
        return out

    def start(self):
        """Leader limp, follower powered and slowly brought to the leader's pose."""
        self.leader.torque_off()
        if not self.follower.torque_is_on():
            self.follower.torque_on()
        joint_glide(self.follower, self.target(self.leader.raw()), max_step=6)

    def step(self):
        """One update: follower goal = leader pose (rate-limited). Returns the follower's joints 1-5."""
        goal = self.target(self.leader.raw())
        here = self.follower.raw()[:5]
        goal = [h + max(-FOLLOW_STEP, min(FOLLOW_STEP, g - h)) for h, g in zip(here, goal)]
        self.follower.set_goal(dict(zip(IDS[:5], goal)))
        return here


LEADER_HELP = """
===============  TEACH THE SO-100 WITH THE LEADER ARM  ===============
 Move the LEADER arm by hand - the follower copies it live. The
 follower's own path is recorded and later repeated exactly.
 The first time: move slowly and check every joint goes the same way.
 Keys (no Enter needed):
   S      start recording            ENTER  stop recording & save
   X      throw away & record again  Q      quit without saving
 (the follower always holds its position when you stop)
=======================================================================
"""


def cmd_learn_leader(arm, leader_name, job):
    path = job_path(arm.bus.serial, job)
    print(f"\n######## Teaching SO-100 JOB {job} on {label(arm.bus.serial)} with the LEADER arm ########")
    if os.path.exists(path) and input(f"Job {job} already exists - replace it? [y/N] ").strip().lower() != "y":
        print("Kept the old job - nothing changed.")
        return
    leader = Arm(find_leader_port(arm.bus.serial, leader_name))
    tele = Teleop(leader, arm)
    print(f"leader: {label(leader.bus.serial)}")
    print(LEADER_HELP)
    samples, recording, saved, msg = [], False, False, ""
    with Keys() as kb:
        print("STEP 0    Put the LEADER in roughly the same pose as the follower, then press any key.\n"
              "          The follower then moves SLOWLY to the leader's pose and starts following.")
        kb.get()
        tele.start()
        print("\n>> Following - move the leader.\n")
        print("STEP 1    Move to where the job should START, then press  S.\n")
        last_draw, t_rate, n_rate, rate = 0.0, time.time(), 0, 0.0
        while True:
            pos = tele.step()
            now = time.time()
            if recording:
                samples.append((now, *pos))
            n_rate += 1
            if now - t_rate >= 1:
                rate, n_rate, t_rate = n_rate / (now - t_rate), 0, now
            if now - last_draw > 0.1:
                rec = f"[REC {now - samples[0][0]:5.1f}s]" if recording else "[not recording]"
                print(f"\r\033[K  {rec}  follower {' '.join(f'{v:5.0f}' for v in pos)}  {rate:3.0f} Hz  {msg}",
                      end="", flush=True)
                last_draw = now
            k = kb.get(0)
            if k is None:
                continue
            msg = ""
            if k == "s" and not recording:
                recording, samples = True, [(now, *pos)]
                print("\r\033[K  * recording started\n")
                print("STEP 2    [REC] Move the leader through the whole job at a calm speed.\n"
                      "STEP 3    Press  ENTER  to stop and save.\n")
            elif k == "x" and recording:
                recording, samples = False, []
                print("\r\033[K  * recording thrown away - press S to start again\n")
            elif k in ("\n", "\r"):
                if not recording or len(samples) < 2:
                    msg = "nothing recorded yet - press S to start"
                    continue
                saved = True
                break
            elif k in ("q", "\x1b"):
                break
            elif k != "s":
                msg = f"key '{k.strip() or 'space'}' does nothing here"
    arm.set_goal(dict(zip(IDS[:5], arm.raw()[:5])))          # follower holds where it is
    print("\r\033[K\n>> Follower holds its position.")
    if not saved:
        print("Quit without saving - nothing changed.")
        return
    job_data = build_joint_job(samples, [])
    closed, opened = gripper_targets(arm)
    job_data.update(job=job, taught=time.strftime("%Y-%m-%d %H:%M"), closed=closed, opened=opened,
                    arm_serial=arm.bus.serial, taught_with_leader=leader.bus.serial)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(job_data, f, indent=1)
    print(f"\nSaved JOB {job} -> {path}")
    print(f"  {samples[-1][0] - samples[0][0]:.1f} s, {len(job_data['points'])} path points")
    print(f"Next: python3 so100_arm.py --arm {arm_names().get(str(arm.bus.serial), arm.bus.serial)} replay {job}")


def cmd_leader_calibrate(leader):
    """Measure the leader's end stops by hand (joints 1-4)."""
    leader.torque_off()
    lo, hi = [9999] * 4, [-1] * 4
    print("\nLEADER CALIBRATION - move EVERY joint of the leader slowly to BOTH of its ends:\n"
          "  shoulder pan (turn left/right), shoulder lift, elbow, wrist up/down.\n"
          "  Press ENTER when every joint has touched both ends, Q to quit.\n")
    with Keys() as kb:
        while True:
            raw = leader.raw()
            for k in range(4):
                lo[k], hi[k] = min(lo[k], raw[k]), max(hi[k], raw[k])
            print("\r\033[K  " + "  ".join(f"{NAMES[k]} {lo[k]}-{hi[k]} ({(hi[k] - lo[k]) * 360 / 4096:3.0f} deg)"
                                          for k in range(4)), end="", flush=True)
            k = kb.get(0.03)
            if k in ("q", "\x1b"):
                print("\nnothing saved")
                return
            if k in ("\n", "\r"):
                break
    small = [NAMES[k] for k in range(4) if hi[k] - lo[k] < 1000]
    if small:
        print(f"\n{', '.join(small)} hardly moved (less than 90 deg) - not saved, please try again")
        return
    st = load_settings(leader)
    st.update(leader_range=[[lo[k], hi[k]] for k in range(4)], name=st.get("name") or "leader")
    save_settings(leader, st)
    print(f"\nsaved - {label(leader.bus.serial)} is the leader arm")


LOG_DIR = os.path.join(HERE, "so100_logs")
HOLD_GAP = 0.4         # s - a pause this long in the teaching is a "hold pose" that gets measured


def tip_model(arm):
    """Arm model to turn joint errors into an (approximate) gripper-tip displacement in mm.
    Uses this arm's calibration if it has one, otherwise the nominal SO-100 geometry."""
    kin = Kinematics()
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE) as f:
            c = json.load(f)
        if c.get("arm_serial") == arm.bus.serial:
            m = Model(kin, c)
            return lambda raw: m.tool(m.q(list(raw)))[0], "calibrated"
    signs = [1, -1, 1, -1, 1]
    off = [(kin.limits[n][0] + kin.limits[n][1]) / 2
           - signs[k] * ((arm.limits[k + 1][0] + arm.limits[k + 1][1]) / 2 - 2048) * TICK
           for k, n in enumerate("12345")]
    m = Model(kin, dict(sign=signs, offset=off, tool=0.0, yaw=0.0, t=[0, 0, 0]))
    return lambda raw: m.tool(m.q(list(raw)))[0], "estimated"


def read_avg(arm, n=3):
    return [float(v) for v in np.mean([arm.raw()[:5] for _ in range(n)], axis=0)]


def replay_joint_job(arm, job, speed, meas=None, stop=None):
    """Run the job once. With meas={} it also measures how far the arm is from the taught
    pose at the start, at every hold pose (pauses in the teaching) and at the end."""
    pts = job["points"]
    closed, opened = job.get("closed"), job.get("opened")
    # Copied jobs: the wrist roll has no end stops to match between arms, so it keeps
    # this arm's current roll and only repeats the taught changes.
    roll_shift = arm.raw()[4] - pts[0]["raw"][4] if job.get("roll_relative") else 0.0
    at = lambda p: [*p["raw"][:4], p["raw"][4] + roll_shift]
    print("  going to the start pose")
    if any("action" in p for p in pts):        # gripper only touched if the job uses it
        arm.set_goal({6: opened})
    joint_glide(arm, at(pts[0]))
    if meas is not None:
        meas.update(checkpoints=[("start", at(pts[0]), read_avg(arm))], lag=[0.0] * 5)
        last_sample = 0.0
    t0, t_ref = time.time(), pts[0]["t"]
    prev = None
    for p in pts:
        if stop:
            stop()                                   # dashboard E-STOP: raises to abort
        due = t0 + (p["t"] - t_ref) / speed
        if meas is not None and prev is not None and p["t"] - prev["t"] >= HOLD_GAP:
            # the teaching paused here: measure the settled pose just before moving on
            time.sleep(max(0.0, due - 0.1 - time.time()))
            n_hold = sum(1 for c in meas["checkpoints"] if c[0].startswith("hold"))
            meas["checkpoints"].append((f"hold {n_hold + 1}", at(prev), read_avg(arm)))
        elif meas is not None and time.time() - last_sample > 0.1:
            act = arm.raw()[:5]                      # following lag while moving
            meas["lag"] = [max(l, abs(a - g)) for l, a, g in zip(meas["lag"], act, at(prev or p))]
            last_sample = time.time()
        time.sleep(max(0.0, due - time.time()))
        arm.set_goal({i: v for i, v in zip(IDS[:5], at(p))})
        prev = p
        if "action" in p:
            arm.wait_still(3)
            is_grip = p["action"] == "grip"
            print(f"  {'GRIP' if is_grip else 'RELEASE'}")
            if is_grip:
                grip(arm, closed)
            else:
                arm.set_goal({6: opened})
                time.sleep(GRIP_WAIT)
            t0, t_ref = time.time(), p["t"]        # carry on with the taught timing from here
    arm.wait_still(3)
    if meas is not None:
        meas["checkpoints"].append(("end", at(pts[-1]), read_avg(arm, 5)))


class Measurements:
    """Collects the per-run measurements, writes a CSV log and prints the summary."""

    def __init__(self, arm, job):
        self.arm, self.job, self.runs = arm, job, []
        self.tip, self.kind = tip_model(arm)
        os.makedirs(LOG_DIR, exist_ok=True)
        name = arm_names().get(str(arm.bus.serial), arm.bus.serial)
        self.path = os.path.join(LOG_DIR, f"arm{name}_job{job}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        with open(self.path, "w") as f:
            f.write("run,checkpoint," + ",".join(f"target_{n}" for n in NAMES[:5]) + ","
                    + ",".join(f"actual_{n}" for n in NAMES[:5]) + ",max_joint_err_deg,tip_err_mm,"
                    + "tip_x,tip_y,tip_z,max_temp_C\n")
        self.temp_start = self.max_temp()

    def max_temp(self):
        return max(self.arm.bus.read(i, ADDR["temp"], 1)[0] for i in IDS)

    def add(self, meas):
        run = len(self.runs) + 1
        temp = self.max_temp()
        rows, worst = [], (0, 0, "")
        with open(self.path, "a") as f:
            for name, target, actual in meas["checkpoints"]:
                jerr = max(abs(a - t) for a, t in zip(actual, target)) * 360 / 4096
                tip_a = self.tip(actual)
                terr = float(np.linalg.norm(tip_a - self.tip(target)))
                rows.append((name, jerr, terr, tip_a))
                worst = max(worst, (terr, jerr, name))
                f.write(f"{run},{name}," + ",".join(f"{v:.1f}" for v in [*target, *actual])
                        + f",{jerr:.2f},{terr:.2f}," + ",".join(f"{v:.1f}" for v in tip_a) + f",{temp}\n")
        self.runs.append({"rows": rows, "lag": max(meas["lag"]) * 360 / 4096, "temp": temp})
        print(f"  measured: worst {worst[0]:.1f} mm at the tip ({worst[1]:.2f} deg) at '{worst[2]}', "
              f"lag while moving up to {self.runs[-1]['lag']:.1f} deg, motors max {temp} C")

    def summary(self):
        if not self.runs:
            return
        n = len(self.runs)
        print(f"\n================ DISPLACEMENT ERROR - {n} run(s) ================")
        print(f"(tip mm = {self.kind} from the arm model; joint degrees are measured directly)")
        print(f"{'pose':<9} {'avg error':>10} {'max error':>10} {'max joint':>10} {'repeatability':>14}")
        names = [r[0] for r in self.runs[0]["rows"]]
        for k, name in enumerate(names):
            vals = [run["rows"][k] for run in self.runs if k < len(run["rows"]) and run["rows"][k][0] == name]
            terr = [v[2] for v in vals]
            tips = np.array([v[3] for v in vals])
            spread = float(np.max(np.linalg.norm(tips - tips.mean(0), axis=1))) if len(tips) > 1 else 0.0
            print(f"{name:<9} {np.mean(terr):>8.1f}mm {np.max(terr):>8.1f}mm {max(v[1] for v in vals):>8.2f}deg "
                  f"{'+-%.1f mm' % spread:>14}")
        print(f"lag while moving: up to {max(r['lag'] for r in self.runs):.1f} deg (catches up when it stops)")
        if n >= 10:
            end = np.array([r["rows"][-1][3] for r in self.runs])
            drift = float(np.linalg.norm(end[-5:].mean(0) - end[:5].mean(0)))
            print(f"drift of the end pose, first 5 runs -> last 5 runs: {drift:.1f} mm")
        print(f"motor temperature: {self.temp_start} C at start -> {self.runs[-1]['temp']} C now (max of 6 motors)")
        print(f"log: {self.path}")


def cmd_replay(arm, job, count, speed):
    path = job_path(arm.bus.serial, job)
    if not os.path.exists(path):
        others = [d for d in (os.listdir(JOBS_DIR) if os.path.isdir(JOBS_DIR) else [])
                  if os.path.exists(job_path(d, job))]
        me = arm_names().get(str(arm.bus.serial), str(arm.bus.serial)[-4:])
        hint = (f"\nJob {job} exists for {label(others[0])} - convert it for this arm with:\n"
                f"  python3 so100_arm.py --arm {me} copy-jobs --from "
                f"{arm_names().get(others[0], others[0][-4:])}"
                if others else f"\nTeach it:  python3 so100_arm.py --arm {me} learn {job}")
        raise ArmError(f"{label(arm.bus.serial)} has no job {job} yet.{hint}")
    with open(path) as f:
        data = json.load(f)
    taught_on = data.get("arm_serial")
    if taught_on and arm.bus.serial and taught_on != arm.bus.serial:
        raise ArmError(f"job {job} was taught on the SO-100 with adapter {taught_on}, but the arm "
                       f"connected now is {arm.bus.serial} - its joints differ, so it would not follow "
                       f"the same path. Teach the job again on this arm.")
    print(f"\n######## {label(arm.bus.serial).upper()} - JOB {job} ########  speed x{speed}")
    arm.torque_on()
    meas_all = Measurements(arm, job)
    total = 0
    while True:
        n = count
        if n is None:
            ans = input(f"\nHow many {'more ' if total else ''}runs? (Enter = finish) > ").strip()
            if not ans:
                break
            if not ans.isdigit() or int(ans) == 0:
                print("   type a number, e.g. 3")
                continue
            n = int(ans)
        for i in range(n):
            print(f"\n=== JOB {job}: run {i + 1}/{n} ===")
            t = time.time()
            meas = {}
            try:
                replay_joint_job(arm, data, speed, meas)
            except KeyboardInterrupt:
                meas_all.summary()
                raise
            total += 1
            print(f"  done in {time.time() - t:.1f}s")
            meas_all.add(meas)
        if count is not None:
            break
    meas_all.summary()
    print("Done - the arm is holding its position (use 'park' to fold it away).")


def cmd_jobs():
    arms = sorted(d for d in os.listdir(JOBS_DIR)) if os.path.isdir(JOBS_DIR) else []
    if not arms:
        print("No SO-100 jobs taught yet - run:  python3 so100_arm.py learn 1")
        return
    names = arm_names()
    for serial in sorted(arms, key=lambda s: names.get(s, s)):
        print(f"{label(serial)}  (use --arm {names.get(serial, serial[-4:])}):")
        files = os.listdir(os.path.join(JOBS_DIR, serial))
        for name in sorted(files, key=lambda s: int(''.join(c for c in s if c.isdigit()) or 0)):
            with open(os.path.join(JOBS_DIR, serial, name)) as f:
                d = json.load(f)
            src = f", converted from {label(d['copied_from'])}" if d.get("copied_from") else ""
            print(f"   job {d.get('job')}: {d['points'][-1]['t']:.0f} s, taught {d.get('taught')}{src}")


def cmd_copy_jobs(arm, src, jobs):
    """Convert jobs taught on another arm for this one. The joints are matched through the
    arms' mechanical end stops (both ranges were measured stop-to-stop), so the same physical
    pose maps onto the same physical pose."""
    srcs = resolve_arm(src, os.listdir(JOBS_DIR) if os.path.isdir(JOBS_DIR) else [])
    if len(srcs) != 1:
        raise ArmError(f"no jobs found for arm '{src}' - see: python3 so100_arm.py jobs")
    src = srcs[0]
    if src == str(arm.bus.serial):
        raise ArmError("that is the arm connected now - choose the OTHER arm with --from")
    with open(SETTINGS_FILE) as f:
        lim_src = json.load(f).get(src, {}).get("limits")
    if not lim_src:
        raise ArmError(f"the end stops of arm {src} are unknown - connect it once and run  info  on it")
    lim_dst = [list(arm.limits[i]) for i in IDS]

    def convert(k, v):
        (a0, a1), (b0, b1) = lim_src[k], lim_dst[k]
        return b0 + (v - a0) * (b1 - b0) / (a1 - a0)

    print(f"Converting jobs of {label(src)} for {label(arm.bus.serial)}:")
    for k in range(4):
        (a0, a1), (b0, b1) = lim_src[k], lim_dst[k]
        print(f"   {NAMES[k]:<14} {a0}-{a1} -> {b0}-{b1}  (range differs by {abs((a1 - a0) - (b1 - b0)) * 360 / 4096:.1f} deg)")
    names = sorted(os.listdir(os.path.join(JOBS_DIR, src)))
    wanted = [f"job{j}.json" for j in jobs] if jobs else names
    for name in wanted:
        if name not in names:
            print(f"   {name[:-5]}: not found on arm {src} - skipped")
            continue
        with open(os.path.join(JOBS_DIR, src, name)) as f:
            d = json.load(f)
        for p in d["points"]:
            p["raw"] = [round(convert(k, v), 1) for k, v in enumerate(p["raw"][:4])] + [p["raw"][4]]
        g0, g1 = lim_src[5]
        for key in ("closed", "opened"):
            if d.get(key) is not None:
                d[key] = round(convert(5, d[key]), 1)
        d.update(arm_serial=arm.bus.serial, copied_from=src, roll_relative=True,
                 converted=time.strftime("%Y-%m-%d %H:%M"))
        dst = job_path(arm.bus.serial, d["job"])
        if os.path.exists(dst) and input(f"   job {d['job']} already exists on this arm - replace? [y/N] ").strip().lower() != "y":
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as f:
            json.dump(d, f, indent=1)
        print(f"   job {d['job']}: converted -> {dst}")
    print("First run each converted job SLOWLY and watch it:  replay N --speed 0.3")


def cmd_park(arm, kin):
    if arm.torque_is_on() and os.path.exists(CALIB_FILE):
        try:
            cmd_move(arm, kin, (150, 0, 120), 40, None, False)
        except ArmError as e:
            print("  (skipping the lift:", e, ")")
    if not arm.torque_is_on():
        arm.torque_on()
    rest = load_settings(arm).get("rest")
    if rest is None:
        raise ArmError("the folded rest pose of this arm is not known yet - fold it by hand with the "
                       "motors off and run:  python3 so100_arm.py set-rest")
    start = arm.raw()
    for k in range(1, 61):                       # 2 s, joint by joint interpolation
        arm.set_goal({i: s + (r - s) * k / 60 for i, s, r in zip(IDS, start, rest)})
        time.sleep(1 / 30)
    arm.wait_still()
    arm.torque_off()
    print("parked, motors off")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port")
    ap.add_argument("--arm", help="which SO-100 when several are connected: its name (1, 2) or serial digits")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    cal = sub.add_parser("calibrate")
    cal.add_argument("--paper", choices=sorted(PAPERS), default="a4", help="sheet used (default a4)")
    w = sub.add_parser("where")
    w.add_argument("--live", action="store_true")
    m = sub.add_parser("move")
    m.add_argument("xyz", type=float, nargs=3)
    m.add_argument("--speed", type=float, default=40, help="mm/s (default 40)")
    m.add_argument("--pitch", default="-90", help="-90 = straight down, 'free' = any angle")
    m.add_argument("--direct", action="store_true", help="straight line, no lift over the table")
    g = sub.add_parser("gripper")
    g.add_argument("how", help="open, close or 0-100")
    sub.add_parser("park")
    sub.add_parser("free")
    ln = sub.add_parser("learn")
    ln.add_argument("job", type=int, nargs="?", default=1)
    ln.add_argument("--leader", nargs="?", const="auto",
                    help="teach by moving the LEADER arm (the follower copies it); optional: the leader's name")
    lc = sub.add_parser("leader-calibrate", help="measure the leader arm's end stops (once)")
    lc.add_argument("--leader", help="the leader's adapter serial digits (default: the SO-100 without a name)")
    rp = sub.add_parser("replay")
    rp.add_argument("job", type=int, nargs="?", default=1)
    rp.add_argument("--count", type=int, help="number of runs (otherwise it asks)")
    rp.add_argument("--speed", type=float, default=1.0, help="1.0 = taught speed, 0.5 = half")
    sub.add_parser("jobs")
    sub.add_parser("set-rest", help="remember the current (folded) pose as this arm's rest pose")
    nm = sub.add_parser("name", help="give the connected arm a short name, e.g. 1 or 2")
    nm.add_argument("new_name")
    cj = sub.add_parser("copy-jobs", help="convert another arm's jobs for this arm")
    cj.add_argument("--from", dest="src", required=True, help="the other arm's name (1, 2) or serial digits")
    cj.add_argument("jobs", type=int, nargs="*", help="job numbers (default: all)")
    a = ap.parse_args()

    migrate_jobs()
    if a.cmd == "jobs":
        cmd_jobs()
        return
    if a.cmd == "leader-calibrate":
        try:
            cmd_leader_calibrate(Arm(find_leader_port(None, a.leader)))
        except ArmError as e:
            print("ERROR:", e)
            sys.exit(1)
        return
    arm = Arm(a.port, a.arm)
    kin = Kinematics()
    try:
        if a.cmd == "info":
            cmd_info(arm)
        elif a.cmd == "calibrate":
            cmd_calibrate(arm, kin, a.paper)
        elif a.cmd == "where":
            cmd_where(arm, kin, a.live)
        elif a.cmd == "move":
            pitch = None if a.pitch == "free" else float(a.pitch)
            cmd_move(arm, kin, a.xyz, a.speed, pitch, a.direct)
        elif a.cmd == "gripper":
            cmd_gripper(arm, a.how)
        elif a.cmd == "park":
            cmd_park(arm, kin)
        elif a.cmd == "free":
            input("HOLD THE ARM - it drops when the motors switch off. Press Enter...")
            arm.torque_off()
            print("motors off")
        elif a.cmd == "name":
            taken = [k for k, v in arm_names().items() if v == a.new_name and k != str(arm.bus.serial)]
            if taken:
                raise ArmError(f"the name '{a.new_name}' already belongs to {label(taken[0])}")
            st = load_settings(arm)
            st["name"] = a.new_name
            save_settings(arm, st)
            print(f"adapter {arm.bus.serial} is now: arm {a.new_name}")
        elif a.cmd == "copy-jobs":
            cmd_copy_jobs(arm, a.src, a.jobs)
        elif a.cmd == "set-rest":
            st = load_settings(arm)
            st["rest"] = arm.raw()
            save_settings(arm, st)
            print(f"rest pose of arm {arm.bus.serial} saved: {st['rest']}")
        elif a.cmd == "learn" and a.leader:
            cmd_learn_leader(arm, a.leader, a.job)
        elif a.cmd == "learn":
            cmd_learn(arm, a.job)
        elif a.cmd == "replay":
            cmd_replay(arm, a.job, a.count, a.speed)
    except KeyboardInterrupt:
        print("\nstopped (motors keep holding their position - use 'park' or 'free')")
    except ArmError as e:
        print("ERROR:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
