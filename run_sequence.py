#!/usr/bin/env python3
"""
Launch file: uArm + camera + SO-100 arm 1 + SO-100 arm 2, repeated as many cycles as you want.

One cycle:
  1. uArm (suction) runs its job, goes home        -> camera photo
  2. SO-100 arm 1 runs its job (ends at its home)    -> camera photo
  3. SO-100 arm 2 runs its job  ||  meanwhile the uArm already runs the NEXT cycle
  (the next arm 1 run waits until arm 2 has finished)

  python3 run_sequence.py               # asks how many cycles
  python3 run_sequence.py --runs 10     # 10 cycles
  python3 run_sequence.py --check       # only checks robots, jobs and camera - nothing moves

Ctrl+C stops everything (the uArm pump switches off, the SO-100 arms hold their position).
Photos: captures/sequence_<date>_<time>/
"""
import argparse
import os
import sys
import threading
import time
import traceback

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import so100_arm as so          # noqa: E402
import uarm_pick_place as ua    # noqa: E402

# ---- what each robot does (change here) ------------------------------------------------
UARM_JOB = 1                    # uArm job number (see: python3 uarm_pick_place.py jobs)
ARM1 = ("1", 1)                 # (SO-100 arm name, job number)
ARM2 = ("2", 3)
UARM_SPEED = 1.0
SO_SPEED = 1.0


# ---- tidy output: every line says which robot it comes from -------------------------
class TaggedOutput:
    def __init__(self, out):
        self.out, self.lock, self.start = out, threading.Lock(), {}

    def write(self, text):
        tag = threading.current_thread().name
        with self.lock:
            for part in text.splitlines(keepends=True):
                if self.start.get(tag, True) and part.strip():
                    self.out.write(f"{time.strftime('%H:%M:%S')} [{tag:<6}] ")
                self.out.write(part)
                self.start[tag] = part.endswith("\n")
            self.out.flush()

    def flush(self):
        self.out.flush()


# ---- camera ---------------------------------------------------------------------------------
class Camera:
    def __init__(self, folder):
        import pyrealsense2 as rs
        if not rs.context().query_devices():
            raise SystemExit("RealSense camera not found - check its USB cable.")
        self.pipe = None
        for w, h, fps in [(1920, 1080, 15), (1280, 720, 15), (640, 480, 30)]:
            pipe, cfg = rs.pipeline(), rs.config()
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            try:
                pipe.start(cfg)
                self.pipe, self.size = pipe, (w, h)
                break
            except RuntimeError:
                continue
        if self.pipe is None:
            raise SystemExit("Could not start the RealSense - is it open in another program?")
        for _ in range(30):                       # let the auto-exposure settle
            self.pipe.wait_for_frames(5000)
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    def capture(self, name):
        for _ in range(4):                        # drop buffered frames -> a fresh picture
            frames = self.pipe.wait_for_frames(5000)
        img = np.asanyarray(frames.get_color_frame().get_data())
        path = os.path.join(self.folder, name + ".png")
        cv2.imwrite(path, img)
        print(f"PHOTO -> {os.path.relpath(path, HERE)}")
        return path

    def close(self):
        self.pipe.stop()


# ---- robots -----------------------------------------------------------------------------------
class UArmWorker:
    def __init__(self):
        self.arm = ua.SwiftPro()
        self.job = ua.load_job(UARM_JOB)
        self.step = self.job.get("stack_step_mm", 0)
        self.taken = 0

    def run(self, cycle):
        drop = self.taken * self.step
        print(f"uArm job {UARM_JOB} - cycle {cycle}" + (f" (sheet {self.taken + 1}, {drop:.1f} mm lower)" if self.step else ""))
        ua.replay(self.arm, self.job["points"], UARM_SPEED, drop, touch=bool(self.job.get("touch")),
                  pick_bags=self.job.get("pick_bags", 1))
        self.arm.move(*ua.SAFE_POSE, ua.feed(ua.TRAVEL_FEED, UARM_SPEED))
        self.taken += 1
        print("uArm at home")

    def close(self):
        try:
            self.arm.pump(False)
        finally:
            self.arm.close()


class SOWorker:
    def __init__(self, name, job):
        self.name, self.jobno = name, job
        self.arm = so.Arm(arm=name)
        path = so.job_path(self.arm.bus.serial, job)
        if not os.path.exists(path):
            raise SystemExit(f"arm {name} has no job {job} - teach it:  python3 so100_arm.py --arm {name} learn {job}")
        import json
        with open(path) as f:
            self.job = json.load(f)
        if self.job.get("arm_serial") not in (None, self.arm.bus.serial):
            raise SystemExit(f"arm {name} job {job} belongs to another arm")

    def run(self, cycle):
        print(f"arm {self.name} job {self.jobno} - cycle {cycle}")
        if not self.arm.torque_is_on():
            self.arm.torque_on()
        so.replay_joint_job(self.arm, self.job, SO_SPEED)
        print(f"arm {self.name} at home (end of its job)")


class Task(threading.Thread):
    """Runs one robot step in the background and remembers any error."""

    def __init__(self, name, fn, *args):
        super().__init__(name=name, daemon=True)
        self.fn, self.args, self.error = fn, args, None

    def run(self):
        try:
            self.fn(*self.args)
        except BaseException as e:            # noqa: BLE001 - reported by the main thread
            self.error = e
            traceback.print_exc()

    def finish(self):
        self.join()
        if self.error:
            raise RuntimeError(f"{self.name} stopped: {self.error}")


def ask_runs():
    while True:
        ans = input("How many cycles in total? > ").strip()
        if ans.isdigit() and int(ans) > 0:
            return int(ans)
        print("   type a number, e.g. 5")


def main():
    global UARM_SPEED, SO_SPEED
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, help="number of cycles (otherwise it asks)")
    ap.add_argument("--check", action="store_true", help="check everything, move nothing")
    ap.add_argument("--uarm-speed", type=float, default=UARM_SPEED)
    ap.add_argument("--so-speed", type=float, default=SO_SPEED)
    a = ap.parse_args()
    UARM_SPEED, SO_SPEED = a.uarm_speed, a.so_speed
    threading.current_thread().name = "main"
    sys.stdout = TaggedOutput(sys.stdout)

    print("connecting ...")
    folder = os.path.join(HERE, "captures", time.strftime("sequence_%Y%m%d_%H%M%S"))
    uarm = UArmWorker()
    arm1, arm2 = SOWorker(*ARM1), SOWorker(*ARM2)
    cam = Camera(folder)
    print(f"uArm job {UARM_JOB} ({'stack, %g mm lower per pick' % uarm.step if uarm.step else 'same height'}), "
          f"arm {ARM1[0]} job {ARM1[1]}, arm {ARM2[0]} job {ARM2[1]}, camera {cam.size[0]}x{cam.size[1]}")
    try:
        if a.check:
            cam.capture("check")
            print("CHECK OK - all robots, jobs and the camera are ready. Nothing moved.")
            return
        runs = a.runs or ask_runs()
        if uarm.step:
            uarm.taken = int(ua.ask_number("Sheets already taken from the uArm stack? (0 = full) [0]: ", 0))
        t_start = time.time()
        uarm.arm.attach()
        uarm.arm.move(*ua.SAFE_POSE, ua.feed(ua.TRAVEL_FEED, UARM_SPEED))

        uarm_task, arm2_task = None, None
        for cycle in range(1, runs + 1):
            print(f"========== CYCLE {cycle}/{runs} ==========")
            if uarm_task is None:                     # first cycle: nothing runs in parallel yet
                uarm_task = Task("uArm", uarm.run, cycle)
                uarm_task.start()
            uarm_task.finish()
            cam.capture(f"cycle{cycle:02d}_1_after_uarm")
            if arm2_task:                             # arm 1 waits for arm 2 of the last cycle
                arm2_task.finish()
            t = Task("arm1", arm1.run, cycle)
            t.start()
            t.finish()
            cam.capture(f"cycle{cycle:02d}_2_after_arm1")
            arm2_task = Task("arm2", arm2.run, cycle)
            arm2_task.start()
            uarm_task = None
            if cycle < runs:                          # uArm starts the next cycle in parallel
                uarm_task = Task("uArm", uarm.run, cycle + 1)
                uarm_task.start()
        arm2_task.finish()
        print(f"ALL {runs} CYCLES DONE in {(time.time() - t_start) / 60:.1f} min - photos in {os.path.relpath(folder, HERE)}")
    except KeyboardInterrupt:
        print("STOPPED by Ctrl+C")
    except (RuntimeError, ua.UArmError, so.ArmError) as e:
        print(f"STOPPED: {e}")
    finally:
        uarm.close()
        cam.close()
        print("uArm pump off. SO-100 arms are holding their position (park them with so100_arm.py --arm N park).")


if __name__ == "__main__":
    main()
