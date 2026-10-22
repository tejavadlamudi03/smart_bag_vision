#!/usr/bin/env python3
"""
Watch the VIRTUAL SO-100 (simulation only - the real arm does not move).

  python3 so100_sim_view.py                 # live window + saves so100_sim.mp4
  python3 so100_sim_view.py --no-window     # only save the video

Shows (1) the A4-sheet calibration touches and (2) a planned 'move' (up, across, down),
using the same arm model and planner as so100_arm.py.  Q / ESC closes the window.
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import so100_arm as A

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1280, 720
REST_RAW = [1897, 1600, 3007, 1896, 2055, 1805]   # folded pose of the arm the model is based on
LIM = {1: (779, 3356), 2: (1566, 3951), 3: (803, 3014), 4: (280, 2604), 5: (0, 4095), 6: (1776, 3278)}


# --- virtual arm (robot frame: origin = base centre on the table, X forward, Y left, Z up) ---
def virtual_arm():
    kin = A.Kinematics()
    signs = [1, -1, 1, -1, 1]
    off = []
    for k, n in enumerate("12345"):
        lo, hi = kin.limits[n]
        smin, smax = LIM[k + 1]
        off.append((lo + hi) / 2 - signs[k] * ((smin + smax) / 2 - 2048) * A.TICK)
    pan = kin.chain[0][0][:3, 3] * 1000
    model = A.Model(kin, dict(sign=signs, offset=off, tool=0.0, yaw=0.0, t=[-pan[0], -pan[1], 0.0]))

    class Limits:
        limits = LIM
    return kin, model, A.Controller(Limits(), model)


def arm_points(kin, model, q):
    """Robot-frame positions of base, shoulder, elbow, wrist, gripper and fingertip (mm)."""
    T, k, pts = np.eye(4), 0, [np.zeros(3)]          # base centre on the table (under the pan axis)
    for Tj, revolute, name in kin.chain:
        T = T @ Tj
        if revolute and name in ("1", "2", "3", "4", "5"):
            R = np.eye(4)
            R[:3, :3] = A._rz(q[k])
            T = T @ R
            k += 1
        pts.append(model.to_table(T[:3, 3] * 1000))
    pts[-1] = model.tool(q)[0]
    return pts


# --- drawing -------------------------------------------------------------------------
def view3d(p, cx=360, cy=470, s=1.15, az=math.radians(-125), el=math.radians(25)):
    x, y, z = p
    # rotate about Z by az, then tilt by el
    xr = x * math.cos(az) - y * math.sin(az)
    yr = x * math.sin(az) + y * math.cos(az)
    sx = xr
    sy = z * math.cos(el) - yr * math.sin(el)
    return int(cx + s * sx), int(cy - s * sy)


def top(p, cx=960, cy=560, s=1.25):
    """Top view: X (forward) points UP the screen, Y (left) points LEFT."""
    return int(cx - s * p[1]), int(cy - s * p[0])


def sheet_corners(near, Wd=297.0, D=210.0):
    return [(near, Wd / 2, 0), (near, -Wd / 2, 0), (near + D, -Wd / 2, 0), (near + D, Wd / 2, 0)]


def draw_scene(img, kin, model, q, trail, sheet_near, title, sub, target=None, touched=()):
    img[:] = (30, 30, 34)
    cv2.rectangle(img, (640, 0), (W, H), (24, 24, 28), -1)
    # table grid (3D) + top view grid
    for v in range(-200, 401, 50):
        cv2.line(img, view3d((v, -250, 0)), view3d((v, 250, 0)), (55, 55, 60), 1)
    for v in range(-250, 251, 50):
        cv2.line(img, view3d((-200, v, 0)), view3d((400, v, 0)), (55, 55, 60), 1)
        cv2.line(img, top((-60, v, 0)), top((380, v, 0)), (45, 45, 50), 1)
    for v in range(-50, 381, 50):
        cv2.line(img, top((v, -250, 0)), top((v, 250, 0)), (45, 45, 50), 1)
    # A4 sheet with its fold lines
    sc = sheet_corners(sheet_near)
    for proj in (view3d, top):
        pts = np.array([proj(p) for p in sc], np.int32)
        cv2.fillPoly(img, [pts], (215, 215, 210))
        mid = sheet_near + 105
        cv2.line(img, proj((sheet_near, 0, 0)), proj((sheet_near + 210, 0, 0)), (150, 150, 150), 1)
        cv2.line(img, proj((mid, 148.5, 0)), proj((mid, -148.5, 0)), (150, 150, 150), 1)
        for p in touched:
            cv2.circle(img, proj(p), 5, (60, 170, 60), -1)
        if target is not None:
            cv2.drawMarker(img, proj(target), (40, 40, 230), cv2.MARKER_CROSS, 18, 2)
    # axes in the top view
    o = top((0, 0, 0))
    cv2.arrowedLine(img, o, top((60, 0, 0)), (80, 80, 240), 2, tipLength=0.25)
    cv2.arrowedLine(img, o, top((0, 60, 0)), (80, 220, 80), 2, tipLength=0.25)
    cv2.putText(img, "X", top((70, -8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 240), 1)
    cv2.putText(img, "Y", top((8, 72, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 1)
    # fingertip trail
    for proj in (view3d, top):
        for a, b in zip(trail, trail[1:]):
            cv2.line(img, proj(a), proj(b), (0, 200, 255), 1)
    # the arm: shadow, links, joints
    pts = arm_points(kin, model, q)
    shadow = [(p[0], p[1], 0) for p in pts]
    for a, b in zip(shadow, shadow[1:]):
        cv2.line(img, view3d(a), view3d(b), (45, 45, 50), 6)
    cv2.circle(img, view3d(pts[0]), 26, (90, 90, 95), -1)                    # base
    colors = [(200, 140, 60), (230, 160, 70), (240, 180, 80), (250, 200, 90), (255, 220, 120), (255, 235, 160)]
    for (a, b), col in zip(zip(pts, pts[1:]), colors):
        cv2.line(img, view3d(a), view3d(b), col, 9, cv2.LINE_AA)
        cv2.line(img, top(a), top(b), col, 6, cv2.LINE_AA)
    for p in pts[1:-1]:
        cv2.circle(img, view3d(p), 7, (40, 40, 40), -1)
        cv2.circle(img, view3d(p), 7, (230, 230, 230), 1)
    tip = pts[-1]
    cv2.circle(img, view3d(tip), 6, (0, 0, 255), -1)
    cv2.circle(img, top(tip), 6, (0, 0, 255), -1)
    # text
    cv2.putText(img, "SIMULATION - virtual SO-100 (the real arm is not moving)", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(img, title, (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, sub, (16, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, f"fingertip  X {tip[0]:6.1f}  Y {tip[1]:6.1f}  Z {tip[2]:6.1f} mm", (16, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "TOP VIEW", (660, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "robot base", (top((0, 0, 0))[0] + 10, top((0, 0, 0))[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)


# --- motion helpers ----------------------------------------------------------------------
def interp(q0, q1, deg_per_frame=2.5):
    q0, q1 = np.array(q0), np.array(q1)
    n = max(1, int(np.degrees(np.abs(q1 - q0).max()) / deg_per_frame))
    return [list(q0 + (q1 - q0) * k / n) for k in range(1, n + 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "so100_sim.mp4"))
    a = ap.parse_args()

    kin, model, ctl = virtual_arm()
    sheet_near = 105.0
    frames = []            # (q, trail, title, sub, target, touched)

    q = model.q(REST_RAW)
    trail, touched = [], []

    def show(qs, title, sub, target=None, hold=0):
        nonlocal q
        for qq in qs:
            q = qq
            trail.append(model.tool(q)[0])
            del trail[:-400]
            frames.append((q, list(trail), title, sub, target, list(touched)))
        for _ in range(hold):
            frames.append((q, list(trail), title, sub, target, list(touched)))

    show([], "Start: folded rest pose", "", hold=30)

    # 1) calibration touches on the A4 sheet
    seed = list((ctl.qlo + ctl.qhi) / 2)
    steps = A.sheet_points("a4")
    for k, (name, (sx, sy), what) in enumerate(steps, 1):
        p = np.array([sheet_near + sx, sy, 0.0])
        pitch, sd = -90, list(q)
        if "LEANING" in what:
            pitch = -50
        if "TURNED" in what or "turned" in what:
            sd[4] = q[4] + math.radians(-90 if "OTHER" in what else 90)
        title = f"1) Calibration touch {k}/{len(steps)}: {name}"
        sub = what if len(what) < 80 else what[:77] + "..."
        q_above, err_a, _, _ = ctl.ik(p + [0, 0, 40], None, sd)
        q_touch, err, _, _ = ctl.ik(p, pitch, q_above if err_a < 3 else sd)
        if err > 3:
            q_touch, err, _, _ = ctl.ik(p, None, sd)
        if err > 3:
            show([], title, "(out of reach - skipped, press s)", p, hold=25)
            continue
        if err_a < 3:
            show(interp(q, q_above), title, sub, p)
        show(interp(q, q_touch, 1.5), title, sub, p)
        touched.append(tuple(p))
        show([], title, sub + "   -> Enter", p, hold=18)
        if err_a < 3:
            show(interp(q, q_above), title, sub, p)

    # 2) a planned move, exactly like:  python3 so100_arm.py move 220 -60 0
    trail.clear()
    for tgt in [(220, -60, 0), (160, 110, 20), (260, 40, 0)]:
        tgt = np.array(tgt, float)
        plan = ctl.plan(tgt, -90, q, 40) or ctl.plan(tgt, None, q, 40)
        title = f"2) move {tgt[0]:.0f} {tgt[1]:.0f} {tgt[2]:.0f}   (up to 80 mm, across, straight down)"
        if plan is None:
            show([], title, "out of reach - refused, nothing moves", tgt, hold=40)
            continue
        show([qq for qq, _ in plan], title, "gripper pointing straight down where possible", tgt, hold=30)

    # render
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    img = np.zeros((H, W, 3), np.uint8)
    if not a.no_window:
        cv2.namedWindow("SO-100 simulation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SO-100 simulation", W, H)
    print(f"{len(frames)} frames ({len(frames) / 30:.0f} s)")
    for q, tr, title, sub, target, tch in frames:
        draw_scene(img, kin, model, q, tr, sheet_near, title, sub, target, tch)
        writer.write(img)
        if not a.no_window:
            cv2.imshow("SO-100 simulation", img)
            if cv2.waitKey(33) & 0xFF in (ord("q"), 27):
                break
            try:
                if cv2.getWindowProperty("SO-100 simulation", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    writer.release()
    cv2.destroyAllWindows()
    print("video saved ->", a.out)


if __name__ == "__main__":
    main()
