#!/usr/bin/env python3
"""
BagVision inspection service - runs the bag AI (YOLO classifier + PatchCore defect model) on the
Siemens IPC. The laptop keeps the robots and the camera; its dashboard sends the picture of each
bag place here and gets the classifier result and the defect score back.

  python3 inspect_server.py                      # port 8100, models in bag_models/
  python3 inspect_server.py --port 8100 --token SECRET

API (works under any path prefix, e.g. behind the Industrial Edge reverse proxy):
  GET  /health                     -> device (cuda/cpu) and the models loaded
  GET  /models                     -> {relative path: sha1} of the model files here
  PUT  /models/<relative path>     -> store a model file (the dashboard keeps them in sync)
  POST /inspect?cls=best.pt&anom=anomaly.pt   body = JPEG of one bag place
       -> {"probs": {...}, "anom": score, "thr": defect limit, "map": small defect map, "ms": time}
"""
import argparse
import hashlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

import anomaly
import bag_inspect as bi

MODEL_DIR = os.environ.get("BAG_MODELS", bi.MODEL_DIR)
MAX_UPLOAD = 500 * 1024 * 1024


def model_path(rel):
    """Model file inside MODEL_DIR (never outside it)."""
    path = os.path.realpath(os.path.join(MODEL_DIR, rel))
    if not path.startswith(os.path.realpath(MODEL_DIR) + os.sep) or not path.endswith(".pt"):
        raise ValueError(f"not a model file: {rel}")
    return path


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Models:
    """Loads each model file once; loads it again when the file is replaced."""

    def __init__(self):
        self.cache, self.hashes = {}, {}
        self.lock = threading.Lock()                 # one inference at a time on the GPU / CPU

    def get(self, rel, kind):
        path = model_path(rel)
        mtime = os.path.getmtime(path)
        hit = self.cache.get(rel)
        if hit and hit[0] == mtime:
            return hit[1]
        if kind == "cls":
            from ultralytics import YOLO
            m = YOLO(path)
        else:
            m = anomaly.AnomalyModel(path)
        self.cache[rel] = (mtime, m)
        print(f"loaded {rel}")
        return m

    def file_hashes(self):
        out = {}
        for root, _, files in os.walk(MODEL_DIR):
            for f in files:
                if f.endswith(".pt"):
                    p = os.path.join(root, f)
                    rel = os.path.relpath(p, MODEL_DIR)
                    key = (os.path.getmtime(p), os.path.getsize(p))
                    if self.hashes.get(rel, (None,))[0] != key:
                        self.hashes[rel] = (key, sha1(p))
                    out[rel] = self.hashes[rel][1]
        return out

    def inspect(self, img, cls, anom):
        out = {}
        with self.lock:
            t = time.time()
            if cls:
                out["probs"] = bi.classify(self.get(cls, "cls"), img)
            if anom:
                m = self.get(anom, "anom")
                small = m.patch_scores(img)
                full = cv2.GaussianBlur(cv2.resize(small, (img.shape[1], img.shape[0])), (0, 0), 6)
                out.update(anom=float(full.max()), thr=float(m.threshold), map=np.round(small, 3).tolist())
            out["ms"] = round(1000 * (time.time() - t))
        return out


def make_handler(models, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def allowed(self):
            if token and self.headers.get("X-Token") != token:
                self.send(403, {"error": "wrong token"})
                return False
            return True

        def body(self):
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_UPLOAD:
                raise ValueError("file too big")
            return self.rfile.read(n)

        def do_GET(self):
            if not self.allowed():
                return
            p = urlparse(self.path).path.rstrip("/")
            if p.endswith("/health") or p == "":
                import torch
                self.send(200, {"ok": True, "device": anomaly.DEVICE,
                                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                "loaded": sorted(models.cache)})
            elif p.endswith("/models"):
                self.send(200, models.file_hashes())
            else:
                self.send(404, {"error": "unknown"})

        def do_PUT(self):
            if not self.allowed():
                return
            p = unquote(urlparse(self.path).path)
            if "/models/" not in p:
                return self.send(404, {"error": "unknown"})
            try:
                rel = p.split("/models/", 1)[1]
                path = model_path(rel)
                data = self.body()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path + ".part", "wb") as f:
                    f.write(data)
                os.replace(path + ".part", path)          # a model in use is swapped in one step
                print(f"received {rel} ({len(data) / 1e6:.1f} MB)")
                self.send(200, {"ok": True})
            except ValueError as e:
                self.send(400, {"error": str(e)})

        def do_POST(self):
            if not self.allowed():
                return
            u = urlparse(self.path)
            if not u.path.rstrip("/").endswith("/inspect"):
                return self.send(404, {"error": "unknown"})
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                img = cv2.imdecode(np.frombuffer(self.body(), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("not a picture")
                self.send(200, models.inspect(img, q.get("cls"), q.get("anom")))
            except (ValueError, OSError) as e:
                self.send(400, {"error": str(e)})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8100)))
    ap.add_argument("--token", default=os.environ.get("AI_TOKEN", ""), help="shared secret the dashboard must send")
    a = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    models = Models()
    print(f"BagVision inspection service on port {a.port} - models in {MODEL_DIR} - running on {anomaly.DEVICE}")
    ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(models, a.token)).serve_forever()


if __name__ == "__main__":
    main()
