#!/usr/bin/env python3
"""
BagVision Control Center - one browser dashboard for the whole cell.

  python3 dashboard.py               # then open  http://localhost:8000
  python3 dashboard.py --port 8080

Pages: Live inspection | Production KPIs + review queue | Robots (run jobs, sequence) |
Teach (record jobs by hand) | Dataset & AI (photos, bag places, training).

The dashboard owns the camera and the robots while it runs - close it (Ctrl+C) before using
the command-line tools (capture, inspect, so100_arm.py, uarm_pick_place.py, run_sequence.py).
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bag_inspect as bi        # noqa: E402
import so100_arm as so          # noqa: E402
import uarm_pick_place as ua    # noqa: E402
import ai_client                # noqa: E402

try:
    import anomaly                # big model: PatchCore / WideResNet-50, learns from GOOD bags only
    import enroll                 # quick set-up of new bag types
except Exception:                 # (dashboard still works without it)
    anomaly = enroll = None
RECORD_SECONDS = 20        # one button press records this long (slide / turn the bag meanwhile)

UI_FILE = os.path.join(HERE, "dashboard_ui.html")
STABLE_FRAMES = 8          # same verdict this many frames = one inspection
EMPTY_FRAMES = 5           # place empty this long = ready for the next bag
ANOM_EVERY = 0.3           # s between anomaly checks of a place (~50 ms each on the GPU)
CHECK_BAND = 0.92          # anomaly above 92 % of the threshold (but below it) = CHECK, sent to review
CHECK_COLOR = (0, 170, 255)
CAMERA_FILE = os.path.join(HERE, "camera.json")   # which camera the dashboard uses
AI_FILE = os.path.join(HERE, "ai_server.json")     # where the AI runs (empty url = this laptop)
LAST_RUN_FILE = os.path.join(HERE, "captures", "last_sequence.json")     # for "Continue sequence"
MODE_FILE = os.path.join(HERE, "bag_models", "detector_mode.txt")   # "classifier" or "anomaly"
BOOT = str(time.time())    # changes on every restart -> the browser reconnects its video
REVIEW_CONF = 0.75         # below this confidence a bag counts as "uncertain" -> review queue
BAG_PRESENT = 40.0         # a place must differ this much from the empty board to hold a bag
                           # (a bag scores ~100; a hand reaching in only ~10-30 -> not counted)


class Stopped(Exception):
    """Raised inside a robot task when Stop / E-STOP is pressed."""


def log_exc(where):
    print(f"[{where}] " + traceback.format_exc().strip().splitlines()[-1])


# =============================================================================================
#  Vision: camera + model + bag counting + dataset capture
# =============================================================================================
CAMERA_STALL = 8.0          # s without a new picture -> open the camera again (cable pulled / re-plugged)


class CameraFeed:
    """Reads the chosen camera (RealSense, a webcam or a stream) in its own thread. librealsense can
    block for ever on a camera that was unplugged, so a watchdog opens the camera again (as a new
    reader) when pictures stop coming, and keeps trying while it is not connected."""

    def __init__(self):
        self.cv = threading.Condition()
        self.img, self.seq, self.t = None, 0, 0.0
        self.cam, self.gen, self.opening, self.error = None, None, 0.0, "starting"
        self.source = "realsense"
        if os.path.exists(CAMERA_FILE):
            self.source = json.load(open(CAMERA_FILE)).get("source") or "realsense"
        threading.Thread(target=self._watch, daemon=True, name="camera-watch").start()

    def set_source(self, source):
        """Switch to another camera: 'realsense', a webcam number or a stream / video address."""
        source = str(source or "realsense").strip()
        with open(CAMERA_FILE, "w") as f:
            json.dump({"source": source}, f)
        old, self.gen, self.cam = self.cam, None, None
        self.source = source
        if old is not None:
            self._stop_quietly(old)                   # free the device before the new one opens
        self.error, self.opening, self.t = "switching camera ...", 0.0, 0.0
        return f"camera: {source}"

    def _open(self):
        try:
            cam = bi.open_camera(self.source)
        except BaseException as e:                    # bi.Camera raises SystemExit when it is missing
            self.error = str(e) or "camera not found"
            self.opening = 0.0
            return
        gen = object()
        self.cam, self.gen, self.t, self.error, self.opening = cam, gen, time.time(), "", 0.0
        print(f"[camera] connected: {self.source}")
        while self.gen is gen:
            try:
                img = cam.frame()
            except Exception:
                time.sleep(0.2)
                continue
            if self.gen is not gen:
                break
            with self.cv:
                self.img, self.seq, self.t = img, self.seq + 1, time.time()
                self.cv.notify_all()

    def _watch(self):
        while True:
            now = time.time()
            stalled = now - self.t > CAMERA_STALL
            if stalled and (not self.opening or now - self.opening > 30):
                if self.cam is not None:
                    print(f"[camera] no picture for {CAMERA_STALL:.0f} s - opening the camera again")
                    self.error = "camera stopped sending pictures - reconnecting"
                    threading.Thread(target=self._stop_quietly, args=(self.cam,), daemon=True).start()   # may block
                self.gen, self.cam, self.opening = None, None, now      # the old reader is abandoned
                threading.Thread(target=self._open, daemon=True, name="camera").start()
            time.sleep(3 if self.error and not self.opening else 1)
            if self.error and not self.opening:
                self.t = 0.0                          # not connected: try again

    @staticmethod
    def _stop_quietly(cam):
        try:
            cam.close()
        except Exception:
            pass

    def frame(self, last_seq):
        """The next picture after last_seq (waits up to 1 s)."""
        with self.cv:
            self.cv.wait_for(lambda: self.seq != last_seq, timeout=1.0)
            if self.seq == last_seq:
                raise RuntimeError("no camera picture")
            return self.img, self.seq

    def close(self):
        cam, self.gen = self.cam, None
        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass


class Vision:
    def __init__(self):
        self.cam = CameraFeed()
        self.areas = bi.load_areas()
        self.areas_frame = None                  # picture size the bag places were marked on
        if os.path.exists(bi.AREAS_FILE):
            self.areas_frame = json.load(open(bi.AREAS_FILE)).get("frame")
        self.empty = cv2.imread(bi.EMPTY_FILE) if os.path.exists(bi.EMPTY_FILE) else None
        self.model = bi.load_model() if os.path.exists(bi.BEST) else None
        self.anom, self.profile = None, None
        self.enroll = None
        self.mode = open(MODE_FILE).read().strip() if os.path.exists(MODE_FILE) else "anomaly"
        self.threshold = 0.5
        self.lock, self.frame_cv = threading.Lock(), threading.Condition()
        self.raw = self.jpeg = None
        self.fps = 0.0
        self.places = {}
        self.scores = {}
        self.auto_label, self.auto_last = None, 0.0
        self.capture_history, self.capture_msg = [], ""
        self.day = os.path.join(bi.OUT, time.strftime("%Y%m%d"))
        os.makedirs(os.path.join(self.day, "crops"), exist_ok=True)
        self.events_file = os.path.join(self.day, "events.json")
        self.events = json.load(open(self.events_file)) if os.path.exists(self.events_file) else []
        self.session_start = time.time()
        self.remote, self.anom_rel = None, None
        self.load_anomaly()                      # needs self.lock / self.places from above
        if os.path.exists(AI_FILE):
            cfg = json.load(open(AI_FILE))
            self.set_remote(cfg.get("url", ""), cfg.get("token", ""))
        self.running = True
        threading.Thread(target=self.loop, daemon=True, name="vision").start()

    # ---------- main loop ----------
    def loop(self):
        while self.running:                      # never let one bad frame stop the live view
            try:
                self._loop()
            except Exception:
                log_exc("vision")
                time.sleep(0.2)

    def _loop(self):
        last, seq = time.time(), 0
        while self.running:
            try:
                img, seq = self.cam.frame(seq)
                if self.areas_frame != list(img.shape[:2]):
                    self.fit_to_frame(img.shape)
            except Exception:                        # no picture: the camera feed reconnects by itself
                self.fps = 0.0
                continue
            view = img.copy()
            s = view.shape[0] / 720
            with self.lock:
                areas, model = dict(self.areas), self.model
            for n in list(self.places):
                if n not in areas:
                    del self.places[n]
            self.scores = bi.bag_scores(img, self.empty, areas) if self.empty is not None else {}
            for n, box in areas.items():
                st = self.places.setdefault(n, {"result": "EMPTY", "conf": 0.0, "recent": [], "streak": 0,
                                                "last": None, "counted": False, "empty_streak": 0})
                if model is not None:
                    part = bi.crop(img, box)
                    bag_here = not self.scores or self.scores.get(n, 0) >= BAG_PRESENT
                    use_anom = self.anom is not None and self.mode == "anomaly"
                    remote = self.remote if self.remote is not None and self.remote.ok else None
                    if remote:                            # AI on the IPC: use its newest answer, never wait
                        if bag_here:
                            remote.submit(n, part, os.path.relpath(bi.BEST, bi.MODEL_DIR),
                                          self.anom_rel if use_anom else None)
                        else:
                            remote.forget(n)
                        r = remote.latest(n)
                        probs = r["probs"] if r else None
                    else:
                        r, probs = None, bi.classify(model, part)
                    if probs is not None:
                        st["recent"] = (st["recent"] + [probs])[-5:]
                    if st["recent"]:
                        avg = {c: float(np.mean([p.get(c, 0) for p in st["recent"]])) for c in st["recent"][-1]}
                        res, conf = bi.verdict(avg, self.threshold)
                    else:
                        res, conf = "EMPTY", 0.0          # no answer from the IPC yet
                    if not bag_here:
                        res, conf = "EMPTY", 1.0          # no bag-sized object here (e.g. just a hand)
                        st["recent"] = []                 # the next bag starts with fresh results
                        st.pop("anom", None)
                        st.pop("hm", None)
                    elif use_anom:
                        if remote:
                            if r and r.get("anom") is not None:
                                st["anom"], st["hm"], st["thr"] = r["anom"], r["hm"], r["thr"]
                        elif "anom" not in st or time.time() - st.get("anom_t", 0) >= ANOM_EVERY:
                            a, hm = self.anom.score(part)
                            st["anom"], st["hm"], st["anom_t"], st["thr"] = a, hm, time.time(), self.anom.threshold
                        a, thr = st.get("anom"), st.get("thr", self.anom.threshold)
                        classifier_bad = self.profile is None and res == "BAD" and conf >= 0.8   # brown bags only
                        if a is None:
                            res, conf = "EMPTY", 0.0      # the defect model has not answered yet
                        elif a > thr or classifier_bad:
                            res, conf = "BAD", min(0.99, 0.6 + 0.4 * (a - thr) / thr) if a > thr else conf
                        elif a > CHECK_BAND * thr:
                            res, conf = "CHECK", 0.5                    # not sure -> human review
                        else:
                            res, conf = "GOOD", min(0.99, 0.75 + 0.25 * (thr - a) / thr)
                    st["result"], st["conf"] = res, conf
                    self.count_bag(n, "BAD" if res == "CHECK" else res, conf, img, box)
                    col = CHECK_COLOR if res == "CHECK" else bi.COLORS[res]
                    label = f"PLACE {n}  {res}" + ("" if res in ("EMPTY", "CHECK") else f"  {100 * conf:.0f}%")
                    if st.get("hm") is not None:              # show WHERE it differs from a good bag
                        self.paint_heat(view, box, st["hm"])
                else:
                    col, label = (200, 160, 60), f"PLACE {n}"
                x, y, w, h = box
                cv2.rectangle(view, (x, y), (x + w, y + h), col, max(3, int(4 * s)))
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85 * s, 2)
                cv2.rectangle(view, (x, y), (x + tw + 20, y + th + 18), col, -1)
                cv2.putText(view, label, (x + 10, y + th + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.85 * s,
                            (255, 255, 255), 2, cv2.LINE_AA)
            self.enroll_step(img)
            if self.auto_label and time.time() - self.auto_last >= 1.0:
                self.auto_last = time.time()
                if self.auto_label == "empty":
                    self.set_empty()                    # board without bags: reference + empty photo
                else:
                    self.capture(self.auto_label, img)
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-6)
            last = now
            ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with self.frame_cv:
                self.raw, self.jpeg = img, buf.tobytes()
                self.frame_cv.notify_all()

    def fit_to_frame(self, shape):
        """The camera can come up at another resolution (USB 2 / USB 3 port): scale the bag places and
        the empty-board picture so they stay on the same spot of the board."""
        h, w = shape[:2]
        if self.areas_frame and self.areas:
            fy, fx = h / self.areas_frame[0], w / self.areas_frame[1]
            with self.lock:
                self.areas = {n: [round(b[0] * fx), round(b[1] * fy), round(b[2] * fx), round(b[3] * fy)]
                              for n, b in self.areas.items()}
                bi.save_areas(self.areas, shape)
            print(f"camera picture is now {w}x{h} - bag places scaled to match")
        self.areas_frame = [h, w]
        if self.empty is not None and self.empty.shape[:2] != (h, w):
            self.empty = cv2.resize(self.empty, (w, h), interpolation=cv2.INTER_AREA)
        for st in self.places.values():
            st["recent"] = []

    def paint_heat(self, view, box, hm):
        """Blend only the hot spots of the anomaly heatmap into the live video."""
        x, y, w, h = box
        thr = self.anom.threshold
        norm = np.clip((hm - 0.75 * thr) / (0.75 * thr), 0, 1)
        if norm.max() <= 0:
            return
        col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        roi = view[y:y + h, x:x + w]
        alpha = (norm * 0.6)[..., None]
        roi[:] = (roi * (1 - alpha) + col * alpha).astype(np.uint8)

    # ---------- counting ----------
    def count_bag(self, n, res, conf, img, box):
        st = self.places[n]
        if res == "EMPTY":
            st["empty_streak"] += 1
            st["streak"], st["last"] = 0, None
            if st["empty_streak"] >= EMPTY_FRAMES:
                st["counted"] = False
            return
        st["empty_streak"] = 0
        st["streak"] = st["streak"] + 1 if res == st["last"] else 1
        st["last"] = res
        if st["counted"] or st["streak"] < STABLE_FRAMES:
            return
        st["counted"] = True
        crop = bi.crop(img, box)
        with self.lock:
            eid = (self.events[-1]["id"] + 1) if self.events else 1
            stamp = time.strftime("%H%M%S")
            crop_path = os.path.join(self.day, "crops", f"bag{eid}_{stamp}.png")
            thumb_path = os.path.join(self.day, "crops", f"bag{eid}_{stamp}_t.jpg")
            cv2.imwrite(crop_path, crop)
            st_hm = self.places[n].get("hm")
            shown = self.anom.heatmap(crop, st_hm) if (self.anom is not None and st_hm is not None) else crop
            cv2.imwrite(thumb_path, cv2.resize(shown, (200, int(200 * crop.shape[0] / crop.shape[1]))))
            self.events.append({"id": eid, "ts": time.time(), "time": time.strftime("%H:%M:%S"), "place": n,
                                "result": res, "conf": round(conf, 3), "uncertain": conf < REVIEW_CONF,
                                "review": None, "crop": crop_path, "thumb": thumb_path})
            self.save_events()

    def save_events(self):
        tmp = self.events_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.events, f)
        os.replace(tmp, self.events_file)

    def review(self, eid, label):
        with self.lock:
            ev = next((e for e in self.events if e["id"] == eid), None)
            if ev is None:
                return "not found"
            ev["review"] = label
            self.save_events()
        # the human answer becomes new training data (active learning)
        os.makedirs(os.path.join(bi.CROPS, label), exist_ok=True)
        shutil.copy(ev["crop"], os.path.join(bi.CROPS, label, f"{label}_review_{os.path.basename(ev['crop'])}"))
        return "ok"

    def reset_kpis(self):
        with self.lock:
            archive = self.events_file.replace(".json", f"_until_{time.strftime('%H%M%S')}.json")
            if self.events:
                shutil.copy(self.events_file, archive)
            self.events = []
            self.save_events()
            self.session_start = time.time()

    def kpis(self):
        with self.lock:
            ev = list(self.events)
        total = len(ev)
        good = sum(e["result"] == "GOOD" for e in ev)
        bad = total - good
        uncertain = sum(e["uncertain"] for e in ev)
        pending = sum(e["uncertain"] and not e["review"] for e in ev)
        escapes = sum(e["result"] == "GOOD" and e["review"] == "bad" for e in ev)
        false_rej = sum(e["result"] == "BAD" and e["review"] == "good" for e in ev)
        reviewed = sum(1 for e in ev if e["review"])
        now = time.time()
        last_hour = sum(1 for e in ev if now - e["ts"] < 3600)
        span_h = max((now - ev[0]["ts"]) / 3600, 1 / 60) if ev else 1
        buckets = []
        for k in range(11, -1, -1):              # last 2 hours, 10-minute buckets
            a, b = now - (k + 1) * 600, now - k * 600
            sel = [e for e in ev if a <= e["ts"] < b]
            buckets.append({"label": time.strftime("%H:%M", time.localtime(b)),
                            "good": sum(e["result"] == "GOOD" for e in sel),
                            "bad": sum(e["result"] == "BAD" for e in sel)})
        per_place = {}
        for e in ev:
            p = per_place.setdefault(e["place"], {"good": 0, "bad": 0})
            p["good" if e["result"] == "GOOD" else "bad"] += 1
        avg_conf = float(np.mean([e["conf"] for e in ev])) * 100 if ev else 0
        return {"total": total, "good": good, "bad": bad, "uncertain": uncertain, "pending_review": pending,
                "escapes": escapes, "false_rejects": false_rej, "reviewed": reviewed,
                "yield": round(100 * good / total, 1) if total else 0,
                "defect_rate": round(100 * bad / total, 1) if total else 0,
                "ppm": round(1e6 * bad / total) if total else 0,
                "per_hour": last_hour if span_h >= 1 else round(total / span_h),
                "avg_conf": round(avg_conf, 1), "trend": buckets, "per_place": per_place,
                "session_min": round((now - self.session_start) / 60, 1)}

    def recent_events(self, n=14):
        with self.lock:
            return [{k: e[k] for k in ("id", "time", "place", "result", "conf", "uncertain", "review")}
                    for e in self.events[-n:][::-1]]

    def review_queue(self, n=24):
        with self.lock:
            return [{k: e[k] for k in ("id", "time", "place", "result", "conf")}
                    for e in self.events if e["uncertain"] and not e["review"]][-n:][::-1]

    def thumb(self, eid):
        with self.lock:
            ev = next((e for e in self.events if e["id"] == eid), None)
        return ev["thumb"] if ev else None

    # ---------- dataset ----------
    def set_area(self, name, box):
        with self.frame_cv:
            shape = self.raw.shape
        with self.lock:
            self.areas[name] = [int(v) for v in box]
            bi.save_areas(self.areas, shape)
            self.areas_frame = list(shape[:2])
        return f"bag place {name} saved"

    def set_empty(self, force=False):
        with self.frame_cv:
            img = self.raw.copy()
        if not force and self.empty is not None and self.areas:
            sc = bi.bag_scores(img, self.empty, self.areas)
            if sc and max(sc.values()) >= BAG_PRESENT:
                return self._msg("there seems to be a bag on the board - take it off first")
        os.makedirs(bi.DATA, exist_ok=True)
        cv2.imwrite(bi.EMPTY_FILE, img)
        self.empty = img
        # also teach the model what an empty place looks like (needed when there is only one place)
        os.makedirs(os.path.join(bi.CROPS, "empty"), exist_ok=True)
        stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}"
        for n, box in self.areas.items():
            cv2.imwrite(os.path.join(bi.CROPS, "empty", f"empty_board_place{n}_{stamp}.png"), bi.crop(img, box))
        return self._msg(f"empty board saved (+{len(self.areas)} empty photo{'s' if len(self.areas) != 1 else ''})")

    def remove_area(self, name):
        with self.frame_cv:
            shape = self.raw.shape
        with self.lock:
            self.areas.pop(name, None)
            bi.save_areas(self.areas, shape)
            self.areas_frame = list(shape[:2])
        return self._msg(f"bag place {name} removed")

    def capture(self, label, img=None):
        if img is None:
            with self.frame_cv:
                img = self.raw.copy()
        if not self.areas:
            return self._msg("mark the bag places first")
        if self.empty is None:
            return self._msg("take all bags off and save the empty board first")
        scores = bi.bag_scores(img, self.empty, self.areas)
        place = max(scores, key=scores.get)
        if scores[place] < bi.BAG_DIFF:
            return self._msg("no bag found in any place")
        stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}"
        for c in bi.LABELS:
            os.makedirs(os.path.join(bi.CROPS, c), exist_ok=True)
        os.makedirs(bi.FULL, exist_ok=True)
        saved = [os.path.join(bi.CROPS, label, f"{label}_place{place}_{stamp}.png")]
        cv2.imwrite(saved[0], bi.crop(img, self.areas[place]))
        for n, v in scores.items():
            if n != place and v < bi.EMPTY_REL * scores[place]:
                p = os.path.join(bi.CROPS, "empty", f"empty_place{n}_{stamp}.png")
                cv2.imwrite(p, bi.crop(img, self.areas[n]))
                saved.append(p)
        full = os.path.join(bi.FULL, f"{label}_place{place}_{stamp}.png")
        cv2.imwrite(full, img)
        saved.append(full)
        self.capture_history.append(saved)
        return self._msg(f"saved {label.upper()} bag in place {place}")

    def undo(self):
        if not self.capture_history:
            return self._msg("nothing to undo")
        for p in self.capture_history.pop():
            if os.path.exists(p):
                os.remove(p)
        return self._msg("last photo deleted")

    def _msg(self, m):
        self.capture_msg = f"{time.strftime('%H:%M:%S')}  {m}"
        return m

    def judge(self, seconds=1.5):
        """Look at the bag places for a moment: worst result wins (BAD > CHECK > GOOD > EMPTY)."""
        seen, end = [], time.time() + seconds
        while time.time() < end:
            for n, st in list(self.places.items()):
                if st["result"] != "EMPTY":
                    seen.append((st["result"], n, st.get("anom")))
            time.sleep(0.1)
        for v in ("BAD", "CHECK", "GOOD"):
            hits = [x for x in seen if x[0] == v]
            if hits:
                scores = [x[2] for x in hits if x[2] is not None]
                return v, hits[0][1], (max(scores) if scores else None)
        return "EMPTY", None, None

    def snapshot(self):
        with self.frame_cv:
            data = self.jpeg
        path = os.path.join(self.day, f"snapshot_{time.strftime('%H%M%S')}.jpg")
        with open(path, "wb") as f:
            f.write(data)
        return os.path.relpath(path, HERE)

    def save_photo(self, path):
        with self.frame_cv:
            img = self.raw.copy()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img)

    # ---------- bag types (anomaly profiles) ----------
    def load_anomaly(self):
        if anomaly is None:
            return
        prof = enroll.active_profile()
        path = os.path.join(enroll.PROFILES, prof, "anomaly.pt") if prof else anomaly.MODEL_FILE
        self.anom_rel = os.path.relpath(path, bi.MODEL_DIR)
        if self.remote is not None:
            self.remote.resync()                 # the IPC gets the same model file
        if not os.path.exists(path):
            self.anom, self.profile = None, None
            return
        try:
            m = anomaly.AnomalyModel(path)
            with self.lock:
                self.anom, self.profile = m, prof
            for st in self.places.values():
                st.pop("anom", None)
                st.pop("hm", None)
            print(f"defect model: {prof or 'default (brown bags)'} - limit {m.threshold:.2f}")
        except Exception:
            log_exc("anomaly")

    # ---------- where the AI runs: this laptop, or the inspection service on the IPC ----------
    def set_remote(self, url, token=""):
        old, self.remote = self.remote, None
        if old is not None:
            old.close()
        url = (url or "").strip()
        if url:
            self.remote = ai_client.RemoteAI(url, token, bi.MODEL_DIR, self.model_files)
        with open(AI_FILE, "w") as f:
            json.dump({"url": url, "token": token}, f)
        return f"AI runs on {url}" if url else "AI runs on this laptop"

    def model_files(self):
        files = [os.path.relpath(bi.BEST, bi.MODEL_DIR)]
        return files + ([self.anom_rel] if self.anom_rel else [])

    def ai_state(self):
        if self.remote is None:
            return {"where": "laptop"}
        return {"where": "IPC" if self.remote.ok else "laptop (IPC offline)", **self.remote.state()}

    def set_mode(self, mode):
        """Which model decides GOOD / BAD: the YOLO classifier alone, or the defect (anomaly) model."""
        self.mode = "classifier" if mode == "classifier" else "anomaly"
        with open(MODE_FILE, "w") as f:
            f.write(self.mode)
        for st in self.places.values():
            st.pop("anom", None)
            st.pop("hm", None)
            st["recent"] = []
        return ("now deciding with the classifier (YOLO, good/bad trained)" if self.mode == "classifier"
                else "now deciding with the defect model (learns from good bags)")

    def use_profile(self, name):
        enroll.set_active(name or "")
        self.load_anomaly()
        return f"now checking: {self.profile or 'default (brown bags)'}"

    def enroll_start(self, name):
        if not name.strip():
            return "type a name for the bag type first"
        if self.empty is None or not self.areas:
            return "set the bag places and the empty board first"
        self.enroll = {"name": enroll.safe_name(name), "side": None, "until": 0, "last": 0,
                       "real": {"front": [], "back": []}, "cuts": {"front": [], "back": []},
                       "building": False, "log": [f"new bag type '{enroll.safe_name(name)}' - record the FRONT"]}
        return self.enroll["log"][-1]

    def enroll_record(self, side):
        e = self.enroll
        if not e:
            return "start a new bag type first"
        if e["building"]:
            return "already building"
        e["real"][side], e["cuts"][side] = [], []
        e["side"], e["until"] = side, time.time() + RECORD_SECONDS
        return f"recording {side.upper()} for {RECORD_SECONDS} s - slowly slide and turn the bag"

    def enroll_step(self, img):
        """Called every frame while recording: keep a photo of the bag place every 0.3 s."""
        e = self.enroll
        if not e or not e["side"]:
            return
        if time.time() > e["until"]:
            n = len(e["real"][e["side"]])
            e["log"].append(f"{e['side'].upper()}: {n} photos" + (" - too few, record again with the bag in view" if n < 15 else ""))
            e["side"] = None
            return
        if time.time() - e["last"] < 0.3 or not self.scores:
            return
        place = max(self.scores, key=self.scores.get)
        if self.scores[place] < BAG_PRESENT:
            return
        e["last"] = time.time()
        box = self.areas[place]
        e["real"][e["side"]].append(bi.crop(img, box).copy())
        if len(e["real"][e["side"]]) % 8 == 1:                    # a few cut-outs for extra variations
            cut, _ = enroll.cut_out(img, self.empty, box)
            if cut is not None:
                e["cuts"][e["side"]].append(cut)

    def enroll_build(self):
        e = self.enroll
        if not e:
            return "start a new bag type first"
        if len(e["real"]["front"]) < 15 or len(e["real"]["back"]) < 15:
            return "record both FRONT and BACK first (at least 15 photos each)"
        if e["building"]:
            return "already building"
        e["building"] = True

        def run():
            try:
                log = lambda m: e["log"].append(m)
                enroll.build_from_recordings(e["name"], e["real"], e["cuts"], self.empty, dict(self.areas), log=log)
                self.use_profile(e["name"])
                e["log"].append(f"DONE - now checking bags as '{e['name']}'")
            except Exception as ex:
                e["log"].append(f"build failed: {ex}")
                log_exc("enroll")
            finally:
                e["building"] = False
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        threading.Thread(target=run, daemon=True).start()
        return "building the new bag type (about 2 minutes)"

    def enroll_state(self):
        e = self.enroll
        profiles = enroll.list_profiles() if enroll else []
        st = {"active": self.profile, "profiles": profiles, "has_default": anomaly is not None and os.path.exists(anomaly.MODEL_FILE)}
        if e:
            st.update(name=e["name"], side=e["side"], left=max(0, round(e["until"] - time.time())) if e["side"] else 0,
                      front=len(e["real"]["front"]), back=len(e["real"]["back"]), building=e["building"], log=e["log"][-8:])
        return st

    def reload_model(self):
        m = bi.load_model()
        with self.lock:
            self.model = m
        if self.remote is not None:
            self.remote.resync()
        for st in self.places.values():
            st["recent"] = []

    def dataset_state(self):
        return {"counts": {c: bi.count(c) for c in bi.LABELS}, "areas": self.areas,
                "empty_set": self.empty is not None, "scores": {k: round(v, 1) for k, v in self.scores.items()},
                "auto": self.auto_label, "msg": self.capture_msg,
                "frame": list(self.raw.shape[:2]) if self.raw is not None else [720, 1280]}

    def close(self):
        self.running = False
        time.sleep(0.3)
        self.cam.close()


# =============================================================================================
#  Robots
# =============================================================================================
class Robot:
    def __init__(self, key, title, kind, arm_name=None):
        self.key, self.title, self.kind, self.arm_name = key, title, kind, arm_name
        self.dev, self.lock = None, threading.Lock()
        self.status, self.error, self.busy = {}, "", None
        self.stop_flag = threading.Event()
        self.progress = ""
        self.stack_taken = {}

    def connect(self):
        with self.lock:
            if self.dev is not None:
                return "already connected"
            try:
                self.dev = ua.SwiftPro() if self.kind == "uarm" else so.Arm(arm=self.arm_name)
                self.error = ""
                return "connected"
            except (SystemExit, Exception) as e:
                self.dev = None
                self.error = str(e).splitlines()[0]
                return self.error

    def jobs(self):
        out = []
        try:
            if self.kind == "uarm":
                for n, p in ua.list_jobs():
                    g = next((q for q in p["points"] if q.get("action") == "grab"), None)
                    left = ua.stack_left.get((round(g["x"]), round(g["y"]))) if g else None
                    out.append({"job": n, "info": "touch-down pick" if p.get("touch") else
                                ("stack -%g mm/pick" % p.get("stack_step_mm", 0)) if p.get("stack_step_mm") else "same height",
                                "taught": p.get("taught", ""), "touch": bool(p.get("touch")), "bags_left": left,
                                "name": p.get("name", "")})
            elif self.dev is not None:
                folder = os.path.join(so.JOBS_DIR, str(self.dev.bus.serial))
                for f in sorted(glob.glob(os.path.join(folder, "job*.json")),
                                key=lambda s: int("".join(c for c in os.path.basename(s) if c.isdigit()) or 0)):
                    d = json.load(open(f))
                    out.append({"job": d.get("job"), "info": f"{d['points'][-1]['t']:.0f} s"
                                + (" (converted)" if d.get("copied_from") else ""), "taught": d.get("taught", ""),
                                "name": d.get("name", "")})
        except Exception:
            log_exc(self.key)
        return out

    def job_file(self, job):
        return ua.job_path(job) if self.kind == "uarm" else so.job_path(self.dev.bus.serial, job)

    def rename_job(self, job, name):
        """Give a job a name (shown next to its number); an empty name removes it."""
        if self.kind != "uarm" and self.dev is None:
            return "robot not connected"
        path = self.job_file(job)
        if not os.path.exists(path):
            return f"job {job} does not exist"
        with open(path) as f:
            d = json.load(f)
        name = " ".join(str(name or "").split())[:40]
        if name:
            d["name"] = name
        else:
            d.pop("name", None)
        with open(path, "w") as f:
            json.dump(d, f, indent=1)
        return f"job {job} is now called '{name}'" if name else f"job {job}: name removed"

    def poll(self):
        if self.dev is None or not self.lock.acquire(blocking=False):
            return
        try:
            if self.kind == "uarm":
                p = self.dev.position()
                self.status = {"pos": [round(v, 1) for v in p],
                               "switch": self.dev.cmd("P2233") == "V1",      # suction-cup touch switch
                               "pump": self.dev.cmd("P2231") != "V0"}
            else:
                b = self.dev.bus
                raw = self.dev.raw()
                temps = [b.read(i, so.ADDR["temp"], 1)[0] for i in so.IDS]
                self.status = {"joints": raw, "temp": max(temps), "volt": b.read(1, so.ADDR["volt"], 1)[0] / 10,
                               "torque": self.dev.torque_is_on(), "serial": b.serial}
        except Exception as e:
            self.error = f"lost connection: {e}"
            self.dev = None
        finally:
            self.lock.release()

    def check_stop(self):
        if self.stop_flag.is_set():
            raise Stopped()

    def info(self):
        return {"key": self.key, "title": self.title, "kind": self.kind, "connected": self.dev is not None,
                "busy": self.busy, "error": self.error, "status": self.status, "progress": self.progress,
                "jobs": self.jobs(), "stack_taken": self.stack_taken}


class Cell:
    """All robots + the background tasks (job runs, teaching, sequence)."""

    def __init__(self, vision):
        self.vision = vision
        self.robots = {"uarm": Robot("uarm", "uArm Swift Pro", "uarm"),
                       "arm1": Robot("arm1", "SO-100 Arm 1", "so", "1"),
                       "arm2": Robot("arm2", "SO-100 Arm 2", "so", "2"),
                       "buddy": Robot("buddy", "SO-100 Buddy", "so", "buddy")}
        self.teach = None
        self.seq = {"running": False, "cycle": 0, "cycles": 0, "step": "", "log": [], "stop": threading.Event()}
        self.alarms = []
        for r in self.robots.values():
            threading.Thread(target=r.connect, daemon=True).start()
        threading.Thread(target=self.poll_loop, daemon=True).start()

    def alarm(self, text):
        self.alarms = ([f"{time.strftime('%H:%M:%S')}  {text}"] + self.alarms)[:20]

    def poll_loop(self):
        while True:
            for r in self.robots.values():
                r.poll()
            time.sleep(1.0)

    # ---------- job runs ----------
    def run_job(self, key, job, count, speed):
        r = self.robots[key]
        if r.dev is None:
            return "not connected"
        if r.busy:
            return f"busy: {r.busy}"
        r.stop_flag.clear()
        r.busy = f"job {job}"
        threading.Thread(target=self._run_job, args=(r, job, count, speed), daemon=True).start()
        return "started"

    def _run_job(self, r, job, count, speed):
        try:
            with r.lock:
                if r.kind == "uarm":
                    prog = ua.load_job(job)
                    step = prog.get("stack_step_mm", 0)
                    r.dev.attach()
                    r.dev.move(*ua.SAFE_POSE, ua.feed(ua.TRAVEL_FEED, speed))
                    for i in range(count):
                        taken = r.stack_taken.get(job, 0)
                        r.progress = f"run {i + 1}/{count}" + (f" · sheet {taken + 1}" if step else "")
                        ua.replay(r.dev, prog["points"], speed, taken * step, stop=r.check_stop,
                                  touch=bool(prog.get("touch")), pick_bags=prog.get("pick_bags", 1))
                        r.dev.move(*ua.SAFE_POSE, ua.feed(ua.TRAVEL_FEED, speed))
                        r.stack_taken[job] = taken + 1
                else:
                    path = so.job_path(r.dev.bus.serial, job)
                    data = json.load(open(path))
                    if not r.dev.torque_is_on():
                        r.dev.torque_on()
                    for i in range(count):
                        r.progress = f"run {i + 1}/{count}"
                        so.replay_joint_job(r.dev, data, speed, stop=r.check_stop)
            r.progress = f"done ({count} runs)"
        except Stopped:
            r.progress = "stopped"
            self.halt(r)
        except SystemExit as e:
            r.progress, r.error = "error", str(e)
        except Exception as e:
            r.progress, r.error = "error", str(e)
            self.alarm(f"{r.title}: {e}")
            log_exc(r.key)
        finally:
            r.busy = None

    def halt(self, r):
        """Stop where it is: uArm pump off; SO-100 holds its current pose."""
        try:
            if r.kind == "uarm":
                r.dev.pump(False)
            else:
                here = r.dev.raw()
                r.dev.set_goal({i: v for i, v in zip(so.IDS, here)})
        except Exception:
            log_exc(r.key)

    def stop(self, key=None):
        keys = [key] if key else list(self.robots)
        for k in keys:
            self.robots[k].stop_flag.set()
        if not key:
            self.seq["stop"].set()
            self.alarm("E-STOP pressed - all robots stopped")
        return "stopping"

    def home(self, key):
        r = self.robots[key]
        if r.dev is None or r.busy:
            return "not connected" if r.dev is None else f"busy: {r.busy}"

        def go():
            r.busy = "home"
            try:
                with r.lock:
                    if r.kind == "uarm":
                        r.dev.attach()
                        r.dev.move(*ua.SAFE_POSE)
                        r.progress = "at home"
                    else:
                        so.cmd_park(r.dev, so.Kinematics())
                        r.progress = "parked (motors off)"
            except (Exception, SystemExit) as e:
                r.progress, r.error = "error", str(e)
            finally:
                r.busy = None
        threading.Thread(target=go, daemon=True).start()
        return "started"

    # ---------- teaching ----------
    def teach_start(self, key, job, leader=False):
        r = self.robots[key]
        if r.dev is None or r.busy:
            return "not connected" if r.dev is None else f"busy: {r.busy}"
        leader = leader and r.kind != "uarm"
        r.lock.acquire()
        r.busy = "teaching"
        tele = None
        try:
            if r.kind == "uarm":
                r.dev.detach()
            elif leader:                              # the follower copies the leader arm
                tele = so.Teleop(so.Arm(so.find_leader_port(r.dev.bus.serial)), r.dev)
            else:
                r.dev.torque_on([6])                  # gripper powered, arm limp
                r.dev.torque_off(so.IDS[:5])
        except Exception as e:
            if tele:
                tele.leader.bus.s.close()
            r.busy = None
            r.lock.release()
            return f"error: {e}"
        self.teach = {"robot": key, "job": job, "recording": False, "samples": [], "events": [],
                      "t0": None, "pos": [], "pump": False, "stop": threading.Event(), "tele": tele,
                      "leader": tele.leader.bus.serial if tele else None, "following": False}
        threading.Thread(target=self._teach_loop, daemon=True).start()
        return "follower moves slowly to the leader's pose, then copies it" if tele else "arm released - move it by hand"

    def _teach_loop(self):
        t = self.teach
        r = self.robots[t["robot"]]
        if t["tele"]:
            try:
                t["tele"].start()                    # slow move to the leader's pose
                t["following"] = True
            except Exception as e:
                self.alarm(f"leader arm: {e}")
                log_exc("leader")
                return
        while not t["stop"].is_set():
            try:
                if t["tele"]:
                    pos = t["tele"].step()
                else:
                    pos = r.dev.position() if r.kind == "uarm" else r.dev.raw()[:5]
            except Exception:
                time.sleep(0.05)
                continue
            t["pos"] = [round(float(v), 1) for v in pos]
            if t["recording"]:
                t["samples"].append((time.time(), *pos))
            time.sleep(0.0 if r.kind == "uarm" else 0.01)

    def teach_record(self):
        t = self.teach
        if not t:
            return "no teaching session"
        t["samples"], t["events"], t["recording"], t["t0"] = [], [], True, time.time()
        return "recording"

    def teach_event(self, kind):
        t = self.teach
        if not t or not t["recording"] or not t["samples"]:
            return "start recording first"
        r = self.robots[t["robot"]]
        on = kind == "grab"
        if r.kind == "uarm":
            r.dev.pump(on)
            t["events"].append((len(t["samples"]) - 1, "grab" if on else "release"))
        else:
            closed, opened = so.gripper_targets(r.dev)
            if on:
                so.grip(r.dev, closed)
            else:
                r.dev.set_goal({6: opened})
            t["events"].append((len(t["samples"]) - 1, "grip" if on else "release"))
        t["pump"] = on
        return "grab" if on else "release"

    def teach_finish(self, save, step_mm=0.0, pick_bags=None):
        t = self.teach
        if not t:
            return "no teaching session"
        t["stop"].set()
        time.sleep(0.1)
        r = self.robots[t["robot"]]
        msg = "cancelled - nothing saved"
        try:
            if save and len(t["samples"]) > 2:
                if r.kind == "uarm":
                    r.dev.pump(False)
                    prog = ua.build_program(t["samples"], t["events"])
                    prog.update(job=t["job"], stack_step_mm=float(step_mm), taught=time.strftime("%Y-%m-%d %H:%M"))
                    if pick_bags:                       # taught on top of a stack of this many bags
                        prog.update(pick_bags=int(pick_bags), touch=True)
                        g = next((q for q in prog["points"] if q.get("action") == "grab"), None)
                        if g:
                            ua.set_stack(g["x"], g["y"], int(pick_bags) - 1)    # the taught bag was taken
                    os.makedirs(ua.JOBS_DIR, exist_ok=True)
                    with open(ua.job_path(t["job"]), "w") as f:
                        json.dump(prog, f, indent=1)
                else:
                    closed, opened = so.gripper_targets(r.dev)
                    data = so.build_joint_job(t["samples"], t["events"])
                    data.update(job=t["job"], taught=time.strftime("%Y-%m-%d %H:%M"), closed=closed,
                                opened=opened, arm_serial=r.dev.bus.serial)
                    if t["leader"]:
                        data["taught_with_leader"] = t["leader"]
                    path = so.job_path(r.dev.bus.serial, t["job"])
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w") as f:
                        json.dump(data, f, indent=1)
                msg = f"job {t['job']} saved ({t['samples'][-1][0] - t['samples'][0][0]:.0f} s)"
        except Exception as e:
            msg = f"save failed: {e}"
            log_exc("teach")
        finally:
            try:
                if r.kind == "uarm":
                    r.dev.pump(False)
                    r.dev.attach()
                elif t["tele"]:
                    r.dev.set_goal(dict(zip(so.IDS[:5], r.dev.raw()[:5])))    # follower holds where it is
                    t["tele"].leader.bus.s.close()
                else:
                    r.dev.torque_on()
            except Exception:
                log_exc("teach lock motors")
            r.busy = None
            self.teach = None
            r.lock.release()
        return msg

    def teach_state(self):
        t = self.teach
        if not t:
            return None
        return {"robot": t["robot"], "job": t["job"], "recording": t["recording"], "pos": t["pos"],
                "leader": t["leader"], "following": t["following"],
                "seconds": round(time.time() - t["t0"], 1) if t["recording"] else 0,
                "samples": len(t["samples"]), "events": [e[1] for e in t["events"]], "pump": t["pump"]}

    # ---------- sequence ----------
    def last_run(self):
        """The last sequence (kept in a file, so 'Continue' also works after a restart)."""
        try:
            return json.load(open(LAST_RUN_FILE))
        except (OSError, ValueError):
            return None

    def _save_run(self, **kv):
        run = self.last_run() or {}
        run.update(kv)
        try:
            with open(LAST_RUN_FILE, "w") as f:
                json.dump(run, f, indent=1)
        except OSError:
            pass

    def sequence_start(self, cycles, ujob, a1job, a2job, uspeed, sspeed, a2bad=None, mode="new", bags=None):
        if self.seq["running"]:
            return "already running"
        for k in ("uarm", "arm1", "arm2"):
            r = self.robots[k]
            if r.dev is None or r.busy:
                return f"{r.title} is {'not connected' if r.dev is None else 'busy'}"
        last = self.last_run() if mode == "continue" else None
        if mode == "continue" and not last:
            return "no earlier sequence to continue - start a new one"
        if last:                                 # carry on with the next cycle, same stack, same counts
            first = last["done"] + 1
            end = last["total"] if last["done"] < last["total"] else last["done"] + cycles
            folder = last["folder"]
            msg = f"continuing at cycle {first} of {end}"
        else:                                    # new: refilled stack, fresh counts, cycle 1
            first, end = 1, cycles
            folder = os.path.join(HERE, "captures", time.strftime("sequence_%Y%m%d_%H%M%S"))
            prog = ua.load_job(ujob)
            g = next((q for q in prog["points"] if q.get("action") == "grab"), None)
            if prog.get("touch") and g:
                ua.set_stack(g["x"], g["y"], int(bags or prog.get("pick_bags", 1)))
            self.vision.reset_kpis()             # every new sequence starts with fresh detection counts
            self._save_run(total=end, done=0, folder=folder, sorted={"GOOD": 0, "BAD": 0})
            msg = "started"
        self.seq.update(running=True, cycle=first - 1, cycles=end, step="starting", log=[], stop=threading.Event(),
                        sorted=dict(last["sorted"]) if last else {"GOOD": 0, "BAD": 0},
                        flow={"cycle": 0, "steps": {}}, history=self.seq.get("history", []) if last else [],
                        robots={k: {"state": "idle", "task": ""} for k in ("uarm", "arm1", "arm2", "camera")},
                        started=time.time())
        if last:
            self._save_run(total=end)
        for r in self.robots.values():
            r.stop_flag.clear()
        threading.Thread(target=self._sequence, args=(first, end, folder, ujob, a1job, a2job, uspeed, sspeed, a2bad),
                         daemon=True).start()
        return msg

    STEPS = ("uarm", "photo1", "arm1", "photo2", "arm2")

    def _rs(self, who, state, task=""):
        """Robot/camera state for the status bar: working / waiting / idle / done."""
        self.seq.setdefault("robots", {})[who] = {"state": state, "task": task}

    def _flow(self, cycle, step, state):
        f = self.seq.setdefault("flow", {"cycle": 0, "steps": {}})
        if f["cycle"] != cycle:
            f["cycle"], f["steps"] = cycle, {k: "pending" for k in self.STEPS}
        f["steps"][step] = state

    def _hist(self, cycle, **kv):
        h = self.seq.setdefault("history", [])
        row = next((r for r in h if r["cycle"] == cycle), None)
        if row is None:
            row = {"cycle": cycle, "v1": "", "v2": "", "final": "", "job2": "", "state": "running",
                   "start": time.strftime("%H:%M:%S"), "t0": time.time(), "secs": None}
            h.append(row)
        row.update(kv)
        if kv.get("state") == "done":
            row["secs"] = round(time.time() - row["t0"])

    def _seq_log(self, text):
        self.seq["log"] = ([f"{time.strftime('%H:%M:%S')}  {text}"] + self.seq["log"])[:40]
        self.seq["step"] = text

    def _cycle_done(self, c):
        self._hist(c, state="done")
        run = self.last_run() or {}
        self._save_run(done=max(c, run.get("done", 0)), sorted=self.seq["sorted"])

    def _sequence(self, first, cycles, folder, ujob, a1job, a2job, uspeed, sspeed, a2bad=None):
        u, a1, a2 = self.robots["uarm"], self.robots["arm1"], self.robots["arm2"]
        prog = ua.load_job(ujob)
        step = prog.get("stack_step_mm", 0)
        d1 = json.load(open(so.job_path(a1.dev.bus.serial, a1job)))
        d2 = json.load(open(so.job_path(a2.dev.bus.serial, a2job)))              # Arm 2: GOOD bags
        d2bad = json.load(open(so.job_path(a2.dev.bus.serial, a2bad))) if a2bad else d2   # Arm 2: BAD bags
        errors = []

        def chk():
            if self.seq["stop"].is_set():
                raise Stopped()

        def uarm_run(c):
            try:
                with u.lock:
                    u.busy = "sequence"
                    taken = u.stack_taken.get(ujob, 0)
                    self._rs("uarm", "working", f"cycle {c} · picking bag (job {ujob})")
                    self._hist(c)
                    self._seq_log(f"cycle {c}: uArm job {ujob}" + (f" (sheet {taken + 1})" if step and not prog.get("touch") else ""))
                    ua.replay(u.dev, prog["points"], uspeed, taken * step, stop=chk, touch=bool(prog.get("touch")),
                              pick_bags=prog.get("pick_bags", 1))
                    u.dev.move(*ua.SAFE_POSE, ua.feed(ua.TRAVEL_FEED, uspeed))
                    u.stack_taken[ujob] = taken + 1
                    self._rs("uarm", "done", f"cycle {c} · bag placed")
            except Exception as e:
                errors.append(e)
                self._rs("uarm", "error", str(e)[:60])
            finally:
                u.busy = None

        def so_run(r, data, c, name):
            try:
                with r.lock:
                    r.busy = "sequence"
                    self._rs(r.key, "working", f"cycle {c} · {name}")
                    self._seq_log(f"cycle {c}: {name}")
                    if not r.dev.torque_is_on():
                        r.dev.torque_on()
                    so.replay_joint_job(r.dev, data, sspeed, stop=chk)
                    self._rs(r.key, "done", f"cycle {c} · finished")
                    if r.key == "arm2":
                        self._cycle_done(c)
                        if self.seq["flow"].get("cycle") == c:
                            self._flow(c, "arm2", "done")
            except Exception as e:
                errors.append(e)
                self._rs(r.key, "error", str(e)[:60])
            finally:
                r.busy = None

        def finish(t):
            t.join()
            if errors:
                raise errors[0]

        try:
            with u.lock:                         # hold the port: the status poller must not talk at the same time
                u.dev.attach()
                u.dev.move(*ua.SAFE_POSE, ua.feed(ua.TRAVEL_FEED, uspeed))
            t_u = t_a2 = None
            for c in range(first, cycles + 1):
                self.seq["cycle"] = c
                if t_u is None:
                    t_u = threading.Thread(target=uarm_run, args=(c,), daemon=True)
                    t_u.start()
                self._flow(c, "uarm", "active")
                finish(t_u)
                self._flow(c, "uarm", "done")
                if t_a2:                                   # Arm 2 must have cleared the last bag before the photo
                    self._rs("camera", "waiting", f"cycle {c} · waits for Arm 2")
                    finish(t_a2)
                    t_a2 = None
                self._flow(c, "photo1", "active")
                self._rs("camera", "working", f"cycle {c} · checking the bag")
                v1 = self.vision.judge()
                self._hist(c, v1=v1[0])
                self._flow(c, "photo1", "done")
                self._rs("camera", "done", f"cycle {c} · {v1[0]}")
                self.vision.save_photo(os.path.join(folder, f"cycle{c:02d}_1_after_uarm_{v1[0]}.png"))
                self._seq_log(f"cycle {c}: photo after uArm - camera says {v1[0]}"
                              + (f" (place {v1[1]}, defect score {v1[2]:.2f})" if v1[2] is not None else ""))
                chk()
                if v1[0] == "EMPTY":                       # the uArm brought no bag: skip the arms, try the next bag
                    self.alarm(f"cycle {c}: no bag after the uArm pick - arms skipped")
                    self.seq["sorted"]["MISSED"] = self.seq["sorted"].get("MISSED", 0) + 1
                    self._hist(c, final="MISSED")
                    self._cycle_done(c)
                    self._seq_log(f"cycle {c}: no bag after the uArm - Arm 1 / Arm 2 skipped")
                    t_u = None
                    if c < cycles:
                        t_u = threading.Thread(target=uarm_run, args=(c + 1,), daemon=True)
                        t_u.start()
                    continue
                self._flow(c, "arm1", "active")
                t1 = threading.Thread(target=so_run, args=(a1, d1, c, f"Arm 1 job {a1job}"), daemon=True)
                t1.start()
                finish(t1)
                self._flow(c, "arm1", "done")
                self._flow(c, "photo2", "active")
                self._rs("camera", "working", f"cycle {c} · checking the bag again")
                v2 = self.vision.judge()
                self._flow(c, "photo2", "done")
                self.vision.save_photo(os.path.join(folder, f"cycle{c:02d}_2_after_arm1_{v2[0]}.png"))
                order = ["BAD", "CHECK", "GOOD", "EMPTY"]
                verdict = min(v1[0], v2[0], key=order.index)      # the worse of the two inspections
                if verdict == "EMPTY":
                    self.alarm(f"cycle {c}: no bag seen by the camera - treated as BAD")
                good = verdict == "GOOD"                          # CHECK / no bag -> rejected (bad job)
                self.seq["sorted"]["GOOD" if good else "BAD"] += 1
                job2, data2 = (a2job, d2) if good else (a2bad or a2job, d2bad)
                self._hist(c, v2=v2[0], final="GOOD" if good else "BAD", job2=job2)
                self._rs("camera", "done", f"cycle {c} · {'GOOD' if good else 'BAD'} bag")
                self._flow(c, "arm2", "active")
                self._seq_log(f"cycle {c}: photo after Arm 1 - camera says {v2[0]} -> bag is "
                              f"{'GOOD' if good else 'BAD'} -> Arm 2 job {job2}")
                t_a2 = threading.Thread(target=so_run, args=(a2, data2, c,
                                        f"Arm 2 job {job2} ({'good' if good else 'bad'} bag, uArm in parallel)"),
                                        daemon=True)
                t_a2.start()
                t_u = None
                if c < cycles:
                    t_u = threading.Thread(target=uarm_run, args=(c + 1,), daemon=True)
                    t_u.start()
            if t_a2:
                finish(t_a2)
            for k in ("uarm", "arm1", "arm2", "camera"):
                self._rs(k, "idle", "")
            self._seq_log(f"ALL {cycles - first + 1} CYCLES DONE - photos in {os.path.relpath(folder, HERE)}")
        except Stopped:
            self._seq_log("sequence STOPPED")
            for r in (u, a1, a2):
                self.halt(r)
        except Exception as e:
            self._seq_log(f"sequence stopped: {e}")
            self.alarm(f"sequence: {e}")
            log_exc("sequence")
        finally:
            self.seq["running"] = False


# =============================================================================================
#  Training (separate process, so the live view keeps running)
# =============================================================================================
class Trainer:
    def __init__(self, vision):
        self.vision, self.proc, self.log, self.result, self.running = vision, None, [], "", False

    def start(self, epochs, model):
        if self.running:
            return "already training"
        self.log, self.result, self.running = [], "", True
        threading.Thread(target=self._run, args=(epochs, model), daemon=True).start()
        return "training started"

    def _run(self, epochs, model):
        cmd = [sys.executable, "-u", os.path.join(HERE, "bag_inspect.py"), "train", "--epochs", str(epochs), "--model", model]
        try:
            self.proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in self.proc.stdout:
                line = line.rstrip()
                if not line or "Axes3D" in line or "warnings.warn" in line:
                    continue
                for part in line.split("\r"):
                    if part.strip():
                        self.log = (self.log + [part.strip()[:160]])[-60:]
            self.proc.wait()
            if self.proc.returncode == 0 and os.path.exists(bi.BEST):
                self.vision.reload_model()
                test = [l for l in self.log if l.startswith("TEST") or "photos called" in l or "missed defects" in l]
                self.result = " | ".join(test) or "done"
            else:
                need = [l for l in self.log if "need at least" in l]
                self.result = need[0] if need else "training failed - see the log"
        except Exception as e:
            self.result = f"training failed: {e}"
        finally:
            self.running = False

    def state(self):
        return {"running": self.running, "log": self.log[-25:], "result": self.result,
                "model": os.path.exists(bi.BEST),
                "model_time": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(bi.BEST)))
                if os.path.exists(bi.BEST) else ""}


# =============================================================================================
#  Web server
# =============================================================================================
def make_handler(vision, cell, trainer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                with open(UI_FILE, "rb") as f:
                    self.send(200, f.read(), "text/html; charset=utf-8")
            elif p == "/api/state":
                self.send(200, {
                    "boot": BOOT, "time": time.strftime("%H:%M:%S"), "date": time.strftime("%a %d %b %Y"),
                    "vision": {"fps": round(vision.fps, 1), "model": vision.model is not None, "ai": vision.ai_state(),
                               "camera": vision.cam.error, "camera_source": vision.cam.source,
                               "threshold": vision.threshold,
                               "anom_threshold": round(vision.anom.threshold, 2) if vision.anom else None,
                               "places": {n: {"result": s["result"], "conf": round(s["conf"] * 100),
                                              "anom": round(s["anom"], 2) if s.get("anom") is not None else None}
                                          for n, s in vision.places.items()}},
                    "kpi": vision.kpis(), "events": vision.recent_events(), "review": vision.review_queue(),
                    "robots": {k: r.info() for k, r in cell.robots.items()},
                    "teach": cell.teach_state(),
                    "seq": {**{k: cell.seq[k] for k in ("running", "cycle", "cycles", "step", "log")},
                            "sorted": cell.seq.get("sorted", {"GOOD": 0, "BAD": 0}),
                            "flow": cell.seq.get("flow", {"cycle": 0, "steps": {}}),
                            "robots": cell.seq.get("robots", {}),
                            "history": [{k: v for k, v in h.items() if k != "t0"} for h in cell.seq.get("history", [])][-15:][::-1],
                            "elapsed": round(time.time() - cell.seq["started"]) if cell.seq.get("started") else 0,
                            "last": cell.last_run()},
                    "dataset": vision.dataset_state(), "train": trainer.state(), "alarms": cell.alarms,
                    "bagtype": vision.enroll_state(), "detector": vision.mode,
                })
            elif p.startswith("/thumb/"):
                path = vision.thumb(int(p.split("/")[-1]))
                if path and os.path.exists(path):
                    with open(path, "rb") as f:
                        self.send(200, f.read(), "image/jpeg")
                else:
                    self.send(404, b"")
            elif p == "/video":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while vision.running:
                        with vision.frame_cv:
                            vision.frame_cv.wait(2)
                            data = vision.jpeg
                        if data:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif p == "/raw.jpg":                       # the camera picture without overlays
                with vision.frame_cv:
                    img = vision.raw.copy()
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                self.send(200, buf.tobytes(), "image/jpeg")
            elif p == "/export.csv":
                rows = ["id,time,place,result,confidence,uncertain,review"]
                with vision.lock:
                    for e in vision.events:
                        rows.append(f"{e['id']},{e['time']},{e['place']},{e['result']},{e['conf']},{e['uncertain']},{e['review'] or ''}")
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", f"attachment; filename=inspections_{time.strftime('%Y%m%d')}.csv")
                self.end_headers()
                self.wfile.write("\n".join(rows).encode())
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            p, b = self.path, self.body()
            try:
                if p == "/api/estop":
                    r = cell.stop()
                elif p == "/api/stop":
                    r = cell.stop(b.get("robot"))
                elif p == "/api/connect":
                    r = cell.robots[b["robot"]].connect()
                elif p == "/api/run":
                    r = cell.run_job(b["robot"], int(b["job"]), int(b.get("count", 1)), float(b.get("speed", 1)))
                elif p == "/api/home":
                    r = cell.home(b["robot"])
                elif p == "/api/uarm/touch":
                    path = ua.job_path(int(b["job"]))
                    d = json.load(open(path))
                    d["touch"] = bool(b.get("on"))
                    with open(path, "w") as jf:
                        json.dump(d, jf, indent=1)
                    r = f"job {b['job']}: touch-down pick {'ON' if d['touch'] else 'off'}"
                elif p == "/api/uarm/bags":
                    d = json.load(open(ua.job_path(int(b["job"]))))
                    g = next(q for q in d["points"] if q.get("action") == "grab")
                    ua.set_stack(g["x"], g["y"], int(b["count"]))
                    r = f"job {b['job']}: {int(b['count'])} bags in the stack - top at about " \
                        f"{g['z'] + (int(b['count']) - d.get('pick_bags', 1)) * ua.SHEET_MM:.0f} mm"
                elif p == "/api/stack":
                    cell.robots["uarm"].stack_taken[int(b["job"])] = int(b.get("taken", 0))
                    r = "ok"
                elif p == "/api/sequence/start":
                    r = cell.sequence_start(int(b["cycles"]), int(b["ujob"]), int(b["a1job"]), int(b["a2job"]),
                                            float(b.get("uspeed", 1)), float(b.get("sspeed", 1)),
                                            a2bad=int(b["a2bad"]) if b.get("a2bad") else None,
                                            mode=b.get("mode", "new"), bags=int(b["bags"]) if b.get("bags") else None)
                elif p == "/api/ai":
                    r = vision.set_remote(b.get("url", ""), b.get("token", ""))
                elif p == "/api/job/rename":
                    r = cell.robots[b["robot"]].rename_job(int(b["job"]), b.get("name", ""))
                elif p == "/api/camera/list":
                    r = {"cameras": bi.list_cameras(), "current": vision.cam.source}
                elif p == "/api/camera":
                    r = vision.cam.set_source(b.get("source"))
                elif p == "/api/sequence/stop":
                    cell.seq["stop"].set()
                    r = "stopping"
                elif p == "/api/teach/start":
                    r = cell.teach_start(b["robot"], int(b["job"]), bool(b.get("leader")))
                elif p == "/api/teach/record":
                    r = cell.teach_record()
                elif p == "/api/teach/event":
                    r = cell.teach_event(b["kind"])
                elif p == "/api/teach/finish":
                    r = cell.teach_finish(bool(b.get("save")), float(b.get("step_mm", 0)), b.get("pick_bags"))
                elif p == "/api/threshold":
                    vision.threshold = max(0.05, min(0.95, float(b["value"])))
                    r = "ok"
                elif p == "/api/snapshot":
                    r = vision.snapshot()
                elif p == "/api/kpi/reset":
                    vision.reset_kpis()
                    r = "ok"
                elif p == "/api/review":
                    r = vision.review(int(b["id"]), b["label"])
                elif p == "/api/dataset/capture":
                    r = vision.capture(b["label"])
                elif p == "/api/dataset/auto":
                    vision.auto_label, vision.auto_last = b.get("label"), time.time()
                    r = "ok"
                elif p == "/api/dataset/undo":
                    r = vision.undo()
                elif p == "/api/dataset/empty":
                    r = vision.set_empty(force=bool(b.get("force")))
                elif p == "/api/dataset/area":
                    r = vision.set_area(str(b["name"]), b["box"])
                elif p == "/api/dataset/area/remove":
                    r = vision.remove_area(str(b["name"]))
                elif p == "/api/enroll/start":
                    r = vision.enroll_start(b.get("name", ""))
                elif p == "/api/enroll/record":
                    r = vision.enroll_record(b["side"])
                elif p == "/api/enroll/build":
                    r = vision.enroll_build()
                elif p == "/api/detector":
                    r = vision.set_mode(b.get("mode", "anomaly"))
                elif p == "/api/profile/use":
                    r = vision.use_profile(b.get("name") or "")
                elif p == "/api/train":
                    r = trainer.start(int(b.get("epochs", 50)), b.get("model", bi.BASE_MODEL))
                else:
                    return self.send(404, {"msg": "unknown"})
                self.send(200, {"msg": r})
            except Exception as e:
                log_exc(p)
                self.send(500, {"msg": f"error: {e}"})
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print("starting camera and model ...")
    vision = Vision()
    cell = Cell(vision)
    trainer = Trainer(vision)
    server = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(vision, cell, trainer))
    print(f"BagVision Control Center:  http://localhost:{a.port}    (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cell.stop()
        for r in cell.robots.values():
            try:
                if r.kind == "uarm" and r.dev:
                    r.dev.pump(False)
                    r.dev.close()
            except Exception:
                pass
        vision.close()
        server.server_close()
        print("stopped")


if __name__ == "__main__":
    main()
