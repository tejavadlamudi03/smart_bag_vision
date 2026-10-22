#!/usr/bin/env python3
"""
Capture a training dataset of paper bag photos with the RealSense D435 colour camera.

  python3 capture_dataset.py
  python3 capture_dataset.py --classes good,tear,hole,stain   # your own defect classes
  python3 capture_dataset.py --interval 0.5                   # faster auto-capture

Keys (click the preview window first):
  1-9    choose the class the next photos belong to
  SPACE  take one photo
  A      auto-capture on/off (one photo every --interval seconds)
  U      undo: delete the last photo
  E      lock / unlock exposure (lock it so all photos have the same brightness)
  Q/ESC  quit

Saved to dataset/ (counts carry over between sessions):
  dataset/<class>/<class>_<time>.png   colour photo, lossless
  dataset/captures.csv                 one row per photo (class, time, exposure, sharpness)
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLASSES = "good,tear,hole,stain,wrinkle,print_defect,glue_defect,handle_defect"
# Best first. 1920x1080 needs a USB 3 port; on USB 2 the colour camera tops out at 1280x720.
CONFIGS = [(1920, 1080, 30), (1920, 1080, 15), (1280, 720, 15), (640, 480, 30)]
WINDOW = "Dataset capture"
CSV_FIELDS = ["file", "class", "time", "width", "height", "exposure", "auto_exposure", "sharpness"]


def start_camera():
    if not rs.context().query_devices():
        raise SystemExit("No RealSense camera found - check its USB cable "
                         "(best: a USB 3 port directly on the laptop, not a hub).")
    for w, h, fps in CONFIGS:
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        try:
            return pipe, pipe.start(cfg)
        except RuntimeError:
            continue
    raise SystemExit("Could not start the camera - is it open in another program?")


def sharpness(img):
    """Higher = sharper. Very low values mean motion blur or out of focus."""
    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def text(img, s, org, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def window_closed():
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:          # Qt build raises instead of returning 0 after close
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default=DEFAULT_CLASSES, help="comma-separated, max 9 (keys 1-9)")
    ap.add_argument("--out", default=os.path.join(HERE, "dataset"))
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between auto-captures")
    a = ap.parse_args()

    classes = [c.strip() for c in a.classes.split(",") if c.strip()][:9]
    for c in classes:
        os.makedirs(os.path.join(a.out, c), exist_ok=True)
    counts = {c: len([f for f in os.listdir(os.path.join(a.out, c)) if f.endswith(".png")])
              for c in classes}
    csv_path = os.path.join(a.out, "captures.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_FIELDS)

    pipe, profile = start_camera()
    cprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
    W, H = cprof.width(), cprof.height()
    rgb = profile.get_device().first_color_sensor()
    print(f"Camera: colour {W}x{H}@{cprof.fps()}  ->  {a.out}")
    print(__doc__.split("Saved to")[0].split("Keys")[1].rstrip())

    cls, auto, last_auto = 0, False, 0.0
    history, msg, msg_until, flash_until = [], "", 0.0, 0.0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    def capture(color):
        c = classes[cls]
        name = f"{c}_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.png"
        path = os.path.join(a.out, c, name)
        cv2.imwrite(path, color)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([os.path.relpath(path, a.out), c, time.strftime("%Y-%m-%d %H:%M:%S"),
                                    W, H, int(rgb.get_option(rs.option.exposure)),
                                    int(rgb.get_option(rs.option.enable_auto_exposure)),
                                    f"{sharpness(color):.0f}"])
        counts[c] += 1
        history.append((path, c))
        print(f"  saved [{c}] #{counts[c]}  {os.path.relpath(path, HERE)}")

    def undo():
        path, c = history.pop()
        if os.path.exists(path):
            os.remove(path)
        rel = os.path.relpath(path, a.out)
        with open(csv_path) as f:
            rows = f.readlines()
        with open(csv_path, "w") as f:
            f.writelines(r for r in rows if not r.startswith(rel + ","))
        counts[c] -= 1
        print(f"  deleted {os.path.relpath(path, HERE)}")

    try:
        while True:
            cf = pipe.wait_for_frames(5000).get_color_frame()
            if not cf:
                continue
            color = np.asanyarray(cf.get_data())
            now = time.time()

            if auto and now - last_auto >= a.interval:
                capture(color)
                last_auto, flash_until = now, now + 0.15

            # ---- preview (overlays are drawn on a copy, never saved) ----
            view = color.copy()
            if now < flash_until:
                cv2.rectangle(view, (0, 0), (W - 1, H - 1), (255, 255, 255), 12)
            s = H / 720                       # keep text readable at any resolution
            y = int(30 * s)
            for i, c in enumerate(classes):
                sel = i == cls
                text(view, f"{'>' if sel else ' '} {i + 1} {c:<14} {counts[c]:>4}", (12, y),
                     (0.65 if sel else 0.55) * s, (0, 255, 255) if sel else (230, 230, 230))
                y += int(26 * s)
            text(view, f"total {sum(counts.values())}", (12, y + 6), 0.55 * s)
            locked = not rgb.get_option(rs.option.enable_auto_exposure)
            status = (f"class: {classes[cls].upper()}   "
                      f"{'AUTO every %.1fs' % a.interval if auto else 'SPACE = photo'}   "
                      f"exposure {'LOCKED' if locked else 'auto'}   sharpness {sharpness(color):.0f}")
            text(view, status, (12, H - int(40 * s)), 0.6 * s, (0, 255, 0) if auto else (255, 255, 255))
            text(view, "1-9 class  SPACE photo  A auto  U undo  E exposure  Q quit",
                 (12, H - int(14 * s)), 0.5 * s, (200, 200, 200))
            if now < msg_until:
                text(view, msg, (W // 2 - int(200 * s), int(40 * s)), 0.8 * s, (0, 200, 255))
            if auto:
                cv2.circle(view, (W - int(30 * s), int(30 * s)), int(12 * s), (0, 0, 255), -1)
            cv2.imshow(WINDOW, view)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27) or window_closed():
                break
            if ord("1") <= k <= ord("9") and k - ord("1") < len(classes):
                cls = k - ord("1")
                msg, msg_until = f"class -> {classes[cls]}", now + 1.0
            elif k == ord(" "):
                capture(color)
                flash_until = now + 0.15
            elif k == ord("a"):
                auto, last_auto = not auto, now
                msg, msg_until = ("AUTO capture ON" if auto else "AUTO capture OFF"), now + 1.0
            elif k == ord("u"):
                if history:
                    undo()
                    msg = "last photo deleted"
                else:
                    msg = "nothing to undo"
                msg_until = now + 1.0
            elif k == ord("e"):
                # Switching auto off keeps the current brightness (tested: ~6% change).
                rgb.set_option(rs.option.enable_auto_exposure, 1 if locked else 0)
                msg, msg_until = ("exposure: auto" if locked else "exposure: LOCKED"), now + 1.0
    finally:
        rgb.set_option(rs.option.enable_auto_exposure, 1)   # don't leave the camera locked
        pipe.stop()
        cv2.destroyAllWindows()
        print("\nPhotos per class:", ", ".join(f"{c} {counts[c]}" for c in classes),
              f"| total {sum(counts.values())}")


if __name__ == "__main__":
    main()
