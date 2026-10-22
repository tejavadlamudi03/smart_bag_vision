#!/usr/bin/env python3
"""
Live view of the Intel RealSense D435 colour camera.

  python3 realsense_view.py

Keys (click the window first):
  S      save a snapshot to captures/
  Q/ESC  quit (or close the window)
"""
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
# Best first. 1920x1080 needs a USB 3 port; on USB 2 the colour camera tops out at 1280x720.
CONFIGS = [(1920, 1080, 30), (1280, 720, 15), (640, 480, 30)]
WINDOW = "RealSense D435"


def window_closed():
    """True once the user closes the window with its X button."""
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:          # Qt build raises instead of returning 0 after close
        return True


def main():
    if not rs.context().query_devices():
        raise SystemExit("No RealSense camera found - check its USB cable "
                         "(best: a USB 3 port directly on the laptop, not a hub).")
    for w, h, fps in CONFIGS:
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        try:
            profile = pipe.start(cfg)
            break
        except RuntimeError:
            continue
    else:
        raise SystemExit("Could not start the camera - is it open in another program?")

    dev = profile.get_device()
    print(f"{dev.get_info(rs.camera_info.name)}  serial {dev.get_info(rs.camera_info.serial_number)}  "
          f"USB {dev.get_info(rs.camera_info.usb_type_descriptor)}  colour {w}x{h}@{fps}")
    print("Keys: S = save snapshot, Q = quit")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    fps_avg, last = 0.0, time.time()
    try:
        while True:
            cf = pipe.wait_for_frames(5000).get_color_frame()
            if not cf:
                continue
            color = np.asanyarray(cf.get_data())
            now = time.time()
            fps_avg = 0.9 * fps_avg + 0.1 / max(now - last, 1e-6)
            last = now

            view = color.copy()
            info = f"{w}x{h} | {fps_avg:4.1f} fps | S save  Q quit"
            cv2.putText(view, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, h / 900, (0, 0, 0), 4)
            cv2.putText(view, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, h / 900, (255, 255, 255), 1)
            cv2.imshow(WINDOW, view)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27) or window_closed():
                break
            if k == ord("s"):
                os.makedirs(CAPTURE_DIR, exist_ok=True)
                path = os.path.join(CAPTURE_DIR, time.strftime("%Y%m%d_%H%M%S") + ".png")
                cv2.imwrite(path, color)
                print("saved", path)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
