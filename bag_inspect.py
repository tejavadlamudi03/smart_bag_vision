#!/usr/bin/env python3
"""
Paper bag inspection: EMPTY / GOOD / BAD for each bag place.
Camera: the RealSense D435 (default) or any webcam / stream:  --camera 0  /  --camera rtsp://...

  python3 bag_inspect.py capture          # 1. mark the bag places, take labelled photos
  python3 bag_inspect.py train            # 2. train the model on the GPU (a few minutes)
  python3 bag_inspect.py inspect          # 3. live: EMPTY / GOOD / BAD for every bag place
  python3 bag_inspect.py check FOLDER|IMAGE   # classify saved photos (e.g. captures/sequence_...)

Bags lie in one of several places (areas) on the board. Each area is marked once; the model
only ever looks at one area at a time, so small defects keep their detail and it learns what
a bag looks like - not where it lies. Model: YOLO11 image classifier, fine-tuned on your photos.

capture keys:
  1 / 2   mark bag place 1 (left) / 2 (right): drag a box around it, press Enter
  N       board is EMPTY (no bags) - press once after marking, and again if the light changes
  G / B   the bag lying on the board is GOOD / BAD   (the program finds which place it is in)
  A       auto: 1 photo per second with the last label - turn/move the bag meanwhile
  U       undo the last photo     E  lock/unlock exposure     Q  quit
inspect keys:  SPACE = save this result   Q = quit
"""
import argparse
import glob
import json
import os
import random
import shutil
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "bag_data")
CROPS = os.path.join(DATA, "crops")              # training photos: one bag place each, per class
FULL = os.path.join(DATA, "full")                # the whole camera picture, for reference
SPLIT = os.path.join(DATA, "split")              # train/val copy built by 'train'
AREAS_FILE = os.path.join(DATA, "areas.json")
EMPTY_FILE = os.path.join(DATA, "empty_board.png")
MODEL_DIR = os.path.join(HERE, "bag_models")
BEST = os.path.join(MODEL_DIR, "best.pt")
OUT = os.path.join(HERE, "inspections")
LABELS = ["good", "bad", "empty"]
BASE_MODEL = "yolo11s-cls.pt"                    # small + accurate; downloads once (~11 MB)
IMGSZ = 384                                      # bigger than the usual 224 -> small defects stay visible
MIN_PER_CLASS = 20
BAG_DIFF = 12.0                                  # how different from the empty board = "a bag is here"
EMPTY_REL = 0.4                                  # another place differing < 40 % as much = empty
# (relative, because the camera's auto-brightness shifts the empty places too when a bag is in view)
WIN = "Paper bag inspection"
CAMERA_SOURCE = "realsense"                      # set with --camera


# --- camera ---------------------------------------------------------------------------------
class Camera:
    def __init__(self):
        import pyrealsense2 as rs
        self.rs = rs
        if not rs.context().query_devices():
            raise SystemExit("RealSense camera not found - check its USB cable.")
        for w, h, fps in [(1920, 1080, 15), (1280, 720, 15), (640, 480, 30)]:
            pipe, cfg = rs.pipeline(), rs.config()
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
            try:
                self.profile = pipe.start(cfg)
                self.pipe = pipe
                break
            except RuntimeError:
                continue
        else:
            raise SystemExit("Could not start the RealSense - is it open in another program?")
        self.sensor = self.profile.get_device().first_color_sensor()
        for _ in range(20):                       # let the auto-exposure settle
            self.frame()

    def frame(self):
        return np.asanyarray(self.pipe.wait_for_frames(5000).get_color_frame().get_data())

    def toggle_exposure(self):
        auto = self.sensor.get_option(self.rs.option.enable_auto_exposure)
        self.sensor.set_option(self.rs.option.enable_auto_exposure, 0 if auto else 1)
        return "LOCKED" if auto else "auto"

    def close(self):
        self.sensor.set_option(self.rs.option.enable_auto_exposure, 1)
        self.pipe.stop()


class Webcam:
    """Any camera OpenCV can open: a number (0 = /dev/video0), a stream (rtsp://, http://) or a video file."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open camera '{source}'")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # most webcams give HD only as MJPG
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)       # the best the camera offers, up to 1080p
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if self.frame() is None:
            raise SystemExit(f"camera '{source}' sends no pictures")

    def frame(self):
        ok, img = self.cap.read()
        if not ok:
            raise RuntimeError("the camera stopped sending pictures")
        return img

    def toggle_exposure(self):
        return "not available for this camera"

    def close(self):
        self.cap.release()


def open_camera(source="realsense"):
    """'realsense' = the Intel RealSense, a number = /dev/videoN, anything else = stream URL or video file."""
    return Camera() if source in ("realsense", "", None) else Webcam(source)


def list_cameras():
    """Cameras that can be chosen: [{"id": ..., "name": ...}] - the RealSense and every webcam."""
    out = []
    try:
        import pyrealsense2 as rs
        for d in rs.context().query_devices():
            out.append({"id": "realsense", "name": f"{d.get_info(rs.camera_info.name)} (RealSense)"})
    except Exception:
        pass
    for path in sorted(glob.glob("/sys/class/video4linux/video*"), key=lambda p: int(p.rsplit("video", 1)[1])):
        try:
            name = open(os.path.join(path, "name")).read().strip()
            index = open(os.path.join(path, "index")).read().strip()
        except OSError:
            continue
        n = path.rsplit("video", 1)[1]
        if index == "0" and "realsense" not in name.lower():   # one entry per webcam (not its metadata node)
            out.append({"id": n, "name": f"{name.split(':')[0]} (/dev/video{n})"})
    return out


# --- bag places (areas) ------------------------------------------------------------------------
def load_areas():
    if os.path.exists(AREAS_FILE):
        with open(AREAS_FILE) as f:
            return {k: v for k, v in json.load(f)["areas"].items()}
    return {}


def save_areas(areas, shape):
    os.makedirs(DATA, exist_ok=True)
    with open(AREAS_FILE, "w") as f:
        json.dump({"areas": areas, "frame": list(shape[:2])}, f, indent=1)


def crop(img, box):
    x, y, w, h = box
    return img[y:y + h, x:x + w]


def bag_scores(img, empty, areas):
    """How different each area is from the empty board (0 = identical)."""
    if empty is None or empty.shape != img.shape:
        return {}
    g = lambda a: cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (9, 9), 0).astype(np.float32)
    return {n: float(np.mean(np.abs(g(crop(img, b)) - g(crop(empty, b))))) for n, b in areas.items()}


def draw_text(img, s, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def window_closed():
    try:
        return cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def count(label):
    return len(glob.glob(os.path.join(CROPS, label, "*.png")))


# --- 1. capture ------------------------------------------------------------------------------
def cmd_capture():
    for c in LABELS:
        os.makedirs(os.path.join(CROPS, c), exist_ok=True)
    os.makedirs(FULL, exist_ok=True)
    cam = open_camera(CAMERA_SOURCE)
    areas = load_areas()
    empty = cv2.imread(EMPTY_FILE) if os.path.exists(EMPTY_FILE) else None
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    history, auto, last_label, last_auto, msg, msg_t, flash, shown = [], False, "good", 0.0, "", 0.0, 0.0, False
    print(__doc__.split("capture keys:")[1].split("inspect keys")[0].rstrip())

    def say(text):
        nonlocal msg, msg_t
        msg, msg_t = text, time.time()
        print("  " + text)

    try:
        while True:
            img = cam.frame()
            now = time.time()
            scores = bag_scores(img, empty, areas)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27) or (shown and window_closed()):
                break
            label = None
            if k in (ord("1"), ord("2")):
                r = cv2.selectROI(WIN, img, showCrosshair=False)
                if r[2] > 50 and r[3] > 50:
                    areas[chr(k)] = [int(v) for v in r]
                    save_areas(areas, img.shape)
                    say(f"bag place {chr(k)} saved" + ("" if empty is not None else " - now clear the board and press N"))
            elif k == ord("n"):
                empty = img.copy()
                os.makedirs(DATA, exist_ok=True)
                cv2.imwrite(EMPTY_FILE, empty)
                say("empty board saved")
            elif k in (ord("g"), ord("b")):
                label = "good" if k == ord("g") else "bad"
            elif k == ord("a"):
                auto, last_auto = not auto, now
                say(f"AUTO {last_label.upper()} ON - move/turn the bag" if auto else "AUTO OFF")
            elif k == ord("u") and history:
                for p in history.pop():
                    if os.path.exists(p):
                        os.remove(p)
                say("last photo deleted")
            elif k == ord("e"):
                say("exposure " + cam.toggle_exposure())
            if auto and now - last_auto >= 1.0:
                label, last_auto = last_label, now

            if label:
                last_label = label
                if not areas:
                    say("mark the bag places first (keys 1 and 2)")
                elif empty is None:
                    say("clear the board and press N first")
                else:
                    place = max(scores, key=scores.get)
                    if scores[place] < BAG_DIFF:
                        say("no bag found in any marked place")
                    else:
                        if any(n != place and v >= 0.7 * scores[place] for n, v in scores.items()):
                            say("looks like bags in more than one place - using the clearest one")
                        stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{int(now * 1000) % 1000:03d}"
                        saved = [os.path.join(CROPS, label, f"{label}_place{place}_{stamp}.png")]
                        cv2.imwrite(saved[0], crop(img, areas[place]))
                        for n, s in scores.items():            # the empty place(s) = free "empty" photos
                            if n != place and s < EMPTY_REL * scores[place]:
                                p = os.path.join(CROPS, "empty", f"empty_place{n}_{stamp}.png")
                                cv2.imwrite(p, crop(img, areas[n]))
                                saved.append(p)
                        full = os.path.join(FULL, f"{label}_place{place}_{stamp}.png")
                        cv2.imwrite(full, img)
                        saved.append(full)
                        history.append(saved)
                        flash = now + 0.15
                        print(f"  saved {label.upper():4} bag in place {place}   "
                              f"(good {count('good')}, bad {count('bad')}, empty {count('empty')})")

            # ---- preview ----
            view = img.copy()
            s = view.shape[0] / 720
            top = max(scores.values()) if scores else 0
            for n, b in areas.items():
                x, y, w, h = b
                has_bag = scores.get(n, 0) >= BAG_DIFF and scores[n] >= EMPTY_REL * top
                col = (0, 255, 255) if has_bag else (160, 160, 160)
                cv2.rectangle(view, (x, y), (x + w, y + h), col, 3)
                info = f"place {n}" + (f": {'BAG' if has_bag else 'empty'} ({scores[n]:.0f})" if n in scores else "")
                draw_text(view, info, (x + 8, y + int(30 * s)), 0.7 * s, col)
            if now < flash:
                cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), (255, 255, 255), 14)
            draw_text(view, f"GOOD {count('good')}   BAD {count('bad')}   EMPTY {count('empty')}",
                      (15, int(40 * s)), 0.9 * s)
            if len(areas) < 2:
                step = f"STEP 1: press {'1' if '1' not in areas else '2'} and drag a box around bag place " \
                       f"{'1 (left)' if '1' not in areas else '2 (right)'}, then Enter"
            elif empty is None:
                step = "STEP 2: take ALL bags off the board, then press N"
            else:
                step = "STEP 3: put ONE bag in a place, press G (good) or B (bad)"
            draw_text(view, step, (15, int(80 * s)), 0.7 * s, (0, 200, 255))
            if now - msg_t < 1.5:
                draw_text(view, msg, (15, int(118 * s)), 0.75 * s, (0, 255, 0))
            draw_text(view, "1/2 places  N empty board  G good  B bad  A auto  U undo  E exposure  Q quit",
                      (15, view.shape[0] - int(15 * s)), 0.6 * s, (220, 220, 220), 1)
            if auto:
                cv2.circle(view, (view.shape[1] - int(30 * s), int(30 * s)), int(12 * s), (0, 0, 255), -1)
            cv2.imshow(WIN, view)
            shown = True
    finally:
        cam.close()
        cv2.destroyAllWindows()
    print(f"photos: good {count('good')}, bad {count('bad')}, empty {count('empty')}"
          "  ->  next: python3 bag_inspect.py train")


# --- 2. train ----------------------------------------------------------------------------------
def build_split(val_frac=0.2, seed=0):
    if os.path.isdir(SPLIT):
        shutil.rmtree(SPLIT)
    for c in LABELS:
        files = sorted(glob.glob(os.path.join(CROPS, c, "*.png")))
        if not files:
            continue
        random.Random(seed).shuffle(files)
        n_val = max(1, int(len(files) * val_frac))
        for k, src in enumerate(files):
            part = "val" if k < n_val else "train"
            os.makedirs(os.path.join(SPLIT, part, c), exist_ok=True)
            shutil.copy(src, os.path.join(SPLIT, part, c, os.path.basename(src)))


def cmd_train(epochs, model_name):
    counts = {c: count(c) for c in LABELS}
    print(f"photos: good {counts['good']}, bad {counts['bad']}, empty {counts['empty']}")
    if min(counts["good"], counts["bad"]) < MIN_PER_CLASS:
        raise SystemExit(f"need at least {MIN_PER_CLASS} GOOD and {MIN_PER_CLASS} BAD photos - "
                         "take more with:  python3 bag_inspect.py capture")
    build_split()
    from ultralytics import YOLO
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    YOLO(model_name).train(data=SPLIT, epochs=epochs, imgsz=IMGSZ, batch=32, device=device, patience=15,
                           project=MODEL_DIR, name="run", exist_ok=True, plots=False, verbose=False,
                           degrees=10, fliplr=0.5, flipud=0.5, hsv_v=0.3)   # bags can lie either way round
    if os.path.exists(BEST):                     # keep every older model in bag_models/history/
        old = os.path.join(MODEL_DIR, "history", time.strftime("model_%Y%m%d_%H%M.pt", time.localtime(os.path.getmtime(BEST))))
        os.makedirs(os.path.dirname(old), exist_ok=True)
        if not os.path.exists(old):
            shutil.copy2(BEST, old)
        print(f"previous model kept -> {os.path.relpath(old, HERE)}")
    shutil.copy(os.path.join(MODEL_DIR, "run", "weights", "best.pt"), BEST)
    # honest score: photos the model never trained on
    m = YOLO(BEST)
    labels = [c for c in LABELS if os.path.isdir(os.path.join(SPLIT, "val", c))]
    conf = {(t, p): 0 for t in labels for p in labels}
    for t in labels:
        for path in glob.glob(os.path.join(SPLIT, "val", t, "*.png")):
            r = m.predict(path, imgsz=IMGSZ, verbose=False)[0]
            conf[(t, r.names[int(r.probs.top1)])] += 1
    total = sum(conf.values())
    right = sum(conf[(c, c)] for c in labels)
    print(f"\nTEST on {total} unseen photos: {right}/{total} correct ({100 * right / max(total, 1):.0f} %)")
    for t in labels:
        print(f"   {t:<5} photos called: " + ", ".join(f"{p} {conf[(t, p)]}" for p in labels))
    print(f"   missed defects (BAD called GOOD): {conf.get(('bad', 'good'), 0)}")
    print(f"model saved -> {os.path.relpath(BEST, HERE)}     next: python3 bag_inspect.py inspect")


# --- 3./4. inspect ---------------------------------------------------------------------------
def load_model():
    if not os.path.exists(BEST):
        raise SystemExit("no trained model yet - run:  python3 bag_inspect.py train")
    from ultralytics import YOLO
    return YOLO(BEST)


def classify(model, img):
    r = model.predict(img, imgsz=IMGSZ, verbose=False)[0]
    return {r.names[i]: float(p) for i, p in enumerate(r.probs.data.tolist())}


def verdict(probs, bad_threshold):
    """'EMPTY', 'BAD' or 'GOOD' and the confidence in it."""
    if probs.get("empty", 0) > max(probs.get("good", 0), probs.get("bad", 0)):
        return "EMPTY", probs["empty"]
    bag = probs.get("good", 0) + probs.get("bad", 0)
    p_bad = probs.get("bad", 0) / bag if bag else 0
    return ("BAD", p_bad) if p_bad >= bad_threshold else ("GOOD", 1 - p_bad)


COLORS = {"GOOD": (0, 200, 0), "BAD": (0, 0, 255), "EMPTY": (160, 160, 160)}


def need_areas():
    areas = load_areas()
    if not areas:
        raise SystemExit("no bag places marked yet - run:  python3 bag_inspect.py capture  (keys 1 and 2)")
    return areas


def cmd_inspect(bad_threshold):
    model, areas, cam = load_model(), need_areas(), open_camera(CAMERA_SOURCE)
    day = os.path.join(OUT, time.strftime("%Y%m%d"))
    os.makedirs(day, exist_ok=True)
    log = os.path.join(day, "results.csv")
    if not os.path.exists(log):
        with open(log, "w") as f:
            f.write("time,place,result,confidence,image\n")
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    recent = {n: [] for n in areas}
    shown = False
    print("SPACE = save this result, Q = quit")
    try:
        while True:
            img = cam.frame()
            view = img.copy()
            s = view.shape[0] / 720
            results = {}
            for n, b in areas.items():
                probs = classify(model, crop(img, b))
                recent[n] = (recent[n] + [probs])[-5:]          # smooth over the last 5 frames
                avg = {c: float(np.mean([p.get(c, 0) for p in recent[n]])) for c in probs}
                res, conf = verdict(avg, bad_threshold)
                results[n] = (res, conf)
                x, y, w, h = b
                cv2.rectangle(view, (x, y), (x + w, y + h), COLORS[res], 5)
                draw_text(view, f"place {n}: {res} {100 * conf:.0f}%", (x + 8, y + int(40 * s)), 1.0 * s, COLORS[res], 3)
            draw_text(view, "SPACE save   Q quit", (15, view.shape[0] - int(15 * s)), 0.6 * s, (220, 220, 220), 1)
            cv2.imshow(WIN, view)
            shown = True
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27) or (shown and window_closed()):
                break
            if k == ord(" "):
                path = os.path.join(day, f"{time.strftime('%H%M%S')}.png")
                cv2.imwrite(path, img)
                with open(log, "a") as f:
                    for n, (res, conf) in results.items():
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{n},{res},{conf:.3f},{path}\n")
                print("  " + "   ".join(f"place {n}: {r} ({100 * c:.0f}%)" for n, (r, c) in results.items())
                      + f"   saved {os.path.relpath(path, HERE)}")
    finally:
        cam.close()
        cv2.destroyAllWindows()


def cmd_check(target, bad_threshold):
    model, areas = load_model(), need_areas()
    files = sorted(glob.glob(os.path.join(target, "*.png")) + glob.glob(os.path.join(target, "*.jpg"))) \
        if os.path.isdir(target) else [target]
    if not files:
        raise SystemExit(f"no images in {target}")
    out = os.path.join(target if os.path.isdir(target) else os.path.dirname(target), "bag_results.csv")
    with open(out, "w") as f:
        f.write("image,place,result,confidence\n")
        for path in files:
            img = cv2.imread(path)
            line = []
            for n, b in areas.items():
                res, conf = verdict(classify(model, crop(img, b)), bad_threshold)
                f.write(f"{os.path.basename(path)},{n},{res},{conf:.3f}\n")
                line.append(f"place {n}: {res:5} {100 * conf:3.0f}%")
            print(f"  {'   '.join(line)}   {os.path.basename(path)}")
    print(f"results -> {os.path.relpath(out, HERE)}")


def main():
    global CAMERA_SOURCE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="realsense",
                    help="realsense (default), a webcam number (0, 1, ...) or a stream / video address")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("capture")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--model", default=BASE_MODEL, help="e.g. yolo11n-cls.pt (faster) or yolo11m-cls.pt (bigger)")
    i = sub.add_parser("inspect")
    i.add_argument("--threshold", type=float, default=0.5, help="p(bad) needed to call a bag BAD (lower = stricter)")
    c = sub.add_parser("check")
    c.add_argument("target")
    c.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    CAMERA_SOURCE = a.camera
    if a.cmd == "capture":
        cmd_capture()
    elif a.cmd == "train":
        cmd_train(a.epochs, a.model)
    elif a.cmd == "inspect":
        cmd_inspect(a.threshold)
    else:
        cmd_check(a.target, a.threshold)


if __name__ == "__main__":
    main()
