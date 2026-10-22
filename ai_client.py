#!/usr/bin/env python3
"""
Laptop side of the IPC inspection service (inspect_server.py).

The dashboard hands over the newest picture of each bag place; one background thread sends them
to the IPC and keeps the newest answer per place, so the live video never waits for the network.
Before use, the model files the dashboard uses are copied to the IPC if they differ there.
While the IPC cannot be reached, `ok` is False and the dashboard uses its own models.
"""
import hashlib
import json
import os
import ssl
import threading
import time
import urllib.request

import cv2
import numpy as np

TIMEOUT = 5.0
RETRY = 3.0                 # s between connection attempts while the IPC is offline
STALE = 3.0                 # s - an answer older than this is not used
HEALTH_EVERY = 10.0         # s - while idle, check this often that the IPC still answers


def sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class RemoteAI:
    def __init__(self, url, token="", model_dir=".", model_files=lambda: []):
        self.url, self.token = url.rstrip("/"), token
        self.model_dir, self.model_files = model_dir, model_files
        self.ssl = ssl.create_default_context()
        self.ssl.check_hostname = False              # the IPC uses a self-signed certificate
        self.ssl.verify_mode = ssl.CERT_NONE
        self.cv = threading.Condition()
        self.pending, self.results, self.forgot = {}, {}, {}
        self.ok, self.error, self.ms, self.device = False, "connecting ...", None, ""
        self.need_sync, self.syncing = True, ""
        self.running = True
        threading.Thread(target=self._worker, daemon=True, name="remote-ai").start()

    # ---------- used by the dashboard ----------
    def submit(self, place, crop, cls, anom):
        """Newest picture of a place (older unsent ones are replaced)."""
        with self.cv:
            self.pending[place] = (time.time(), crop, cls, anom)
            self.cv.notify()

    def forget(self, place):
        """The place is empty now: drop its answer and ignore answers still on the way."""
        with self.cv:
            self.pending.pop(place, None)
            self.results.pop(place, None)
            self.forgot[place] = time.time()

    def latest(self, place):
        r = self.results.get(place)
        return r if r and time.time() - r["t"] < STALE else None

    def resync(self):
        self.need_sync = True

    def close(self):
        self.running = False
        with self.cv:
            self.cv.notify()

    def state(self):
        return {"url": self.url, "ok": self.ok, "error": self.error, "ms": self.ms, "device": self.device,
                "syncing": self.syncing}

    # ---------- network ----------
    def _req(self, method, path, data=None, ctype="application/octet-stream", timeout=TIMEOUT):
        req = urllib.request.Request(self.url + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", ctype)
        if self.token:
            req.add_header("X-Token", self.token)
        with urllib.request.urlopen(req, timeout=timeout, context=self.ssl) as r:
            return json.loads(r.read())

    def _connect(self):
        h = self._req("GET", "/health")
        self.device = h.get("gpu") or h.get("device", "")
        if self.need_sync:
            self.need_sync = False
            there = self._req("GET", "/models")
            for rel in self.model_files():
                path = os.path.join(self.model_dir, rel)
                if os.path.exists(path) and there.get(rel) != sha1(path):
                    self.syncing = f"copying {rel} to the IPC ({os.path.getsize(path) / 1e6:.0f} MB) ..."
                    print(f"[remote AI] {self.syncing}")
                    with open(path, "rb") as f:
                        self._req("PUT", "/models/" + rel, f.read(), timeout=600)
            self.syncing = ""
        self.ok, self.error = True, ""

    def _worker(self):
        while self.running:
            if not self.ok or self.need_sync:
                try:
                    self._connect()
                    print(f"[remote AI] connected to {self.url} ({self.device})")
                except Exception as e:                      # noqa: BLE001 - offline: retry later
                    self.ok, self.error, self.syncing = False, str(e)[:120], ""
                    self.need_sync = True
                    time.sleep(RETRY)
                    continue
            with self.cv:
                idle_since = time.time()
                while self.running and not self.pending and not self.need_sync and time.time() - idle_since < HEALTH_EVERY:
                    self.cv.wait(1.0)
                job = self.pending and not self.need_sync
            if not job:
                if self.running and not self.need_sync:
                    try:                                    # nothing to inspect: still check the IPC is there
                        self._connect()
                    except Exception as e:                  # noqa: BLE001
                        self.ok, self.error = False, str(e)[:120]
                        print(f"[remote AI] lost the IPC: {self.error} - using the laptop models")
                continue
            with self.cv:
                if not self.pending:
                    continue
                place = min(self.pending, key=lambda k: self.pending[k][0])   # oldest first: places take turns
                t, crop, cls, anom = self.pending.pop(place)
            try:
                ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                q = "&".join(f"{k}={v}" for k, v in (("cls", cls), ("anom", anom)) if v)
                t0 = time.time()
                r = self._req("POST", f"/inspect?{q}", buf.tobytes(), "image/jpeg")
                self.ms = round(1000 * (time.time() - t0))
            except Exception as e:                          # noqa: BLE001
                self.ok, self.error = False, str(e)[:120]
                print(f"[remote AI] lost the IPC: {self.error} - using the laptop models")
                continue
            if t < self.forgot.get(place, 0):
                continue                                    # the bag was taken away meanwhile
            if r.get("map") is not None:                    # same smoothing as AnomalyModel.score
                small = np.array(r["map"], np.float32)
                r["hm"] = cv2.GaussianBlur(cv2.resize(small, (crop.shape[1], crop.shape[0])), (0, 0), 6)
            r["t"] = t
            self.results[place] = r
