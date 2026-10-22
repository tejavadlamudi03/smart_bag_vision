#!/usr/bin/env python3
"""
One-shot set-up of a new bag type ("enrolment").

Take ONE photo of a good bag front-up and ONE back-up (more is better). The bag is cut out
of each photo (by comparing with the empty board) and placed back onto the empty board in
many angles, positions, sizes and lighting conditions - in every bag place. From those
pictures the anomaly model (PatchCore / WideResNet-50) learns what a GOOD bag of this type
looks like; anything else is a defect.

Used by the dashboard (Dataset & AI -> Bag types). Profiles: bag_models/profiles/<name>/
"""
import json
import os
import re
import time

import cv2
import numpy as np

import anomaly

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.join(HERE, "bag_models", "profiles")
ACTIVE_FILE = os.path.join(HERE, "bag_models", "active_profile.txt")
PER_PLACE = 40            # pictures made per captured photo, per bag place (training)
VAL_PER_PLACE = 8         # extra pictures (other angles) to set the defect limit
MARGIN = 1.15             # limit = 15 % above the worst good check picture


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())[:40] or "bag"


def profile_dir(name):
    return os.path.join(PROFILES, safe_name(name))


def list_profiles():
    out = []
    if os.path.isdir(PROFILES):
        for n in sorted(os.listdir(PROFILES)):
            meta = os.path.join(PROFILES, n, "meta.json")
            if os.path.exists(meta):
                out.append(json.load(open(meta)))
    return out


def active_profile():
    if os.path.exists(ACTIVE_FILE):
        n = open(ACTIVE_FILE).read().strip()
        if n and os.path.exists(os.path.join(PROFILES, n, "anomaly.pt")):
            return n
    return None


def set_active(name):
    with open(ACTIVE_FILE, "w") as f:
        f.write(safe_name(name) if name else "")


# --- 1. cut the bag out ---------------------------------------------------------------------------
def cut_out(frame, empty, box, thr=35):
    """BGRA picture of just the bag (background transparent), or (None, reason)."""
    x, y, w, h = box
    a = cv2.GaussianBlur(frame[y:y + h, x:x + w], (5, 5), 0).astype(np.int16)
    b = cv2.GaussianBlur(empty[y:y + h, x:x + w], (5, 5), 0).astype(np.int16)
    mask = (np.abs(a - b).max(axis=2) > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, "no bag found - is the empty board picture up to date?"
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.15 * w * h:
        return None, ("bag hardly visible against the background - use a background with "
                      "more contrast (e.g. light bags on dark, dark bags on light)")
    full = np.zeros_like(mask)
    cv2.drawContours(full, [c], -1, 255, -1)                 # filled: holes inside the bag kept as bag
    bx, by, bw, bh = cv2.boundingRect(c)
    rgba = cv2.cvtColor(frame[y + by:y + by + bh, x + bx:x + bx + bw], cv2.COLOR_BGR2BGRA)
    alpha = cv2.GaussianBlur(full[by:by + bh, bx:bx + bw], (3, 3), 0)
    rgba[..., 3] = alpha
    return rgba, f"bag cut out ({bw}x{bh} px)"


# --- 2. make many realistic variations -------------------------------------------------------------
def rotate(rgba, angle, scale):
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def compose(bg, rgba, rng):
    """Put the bag on the empty-board crop at a random angle/position that still fits the place."""
    H, W = bg.shape[:2]
    for _ in range(30):
        base = 180.0 if rng.random() < 0.5 else 0.0          # bags may lie either way round
        angle = base + rng.uniform(-12, 12)
        r = rotate(rgba, angle, rng.uniform(0.96, 1.04))
        h, w = r.shape[:2]
        if w <= W - 4 and h <= H - 4:
            break
    else:                                                     # too big for this place: shrink to fit
        r = rotate(rgba, 0, min((W - 4) / rgba.shape[1], (H - 4) / rgba.shape[0]))
        h, w = r.shape[:2]
    ox, oy = rng.integers(0, W - w + 1), rng.integers(0, H - h + 1)
    out = bg.astype(np.float32).copy()
    a = r[..., 3:4].astype(np.float32) / 255
    out[oy:oy + h, ox:ox + w] = out[oy:oy + h, ox:ox + w] * (1 - a) + r[..., :3] * a
    out = out * rng.uniform(0.85, 1.15) + rng.uniform(-12, 12)          # lighting
    out += rng.normal(0, 2.0, out.shape)                                 # camera noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)                           # slight focus change
    return out


def make_set(cutouts, empty, areas, per_place, seed):
    rng = np.random.default_rng(seed)
    imgs = []
    for cut in cutouts:
        for n, (x, y, w, h) in sorted(areas.items()):
            bg = empty[y:y + h, x:x + w]
            imgs += [compose(bg, cut, rng) for _ in range(per_place)]
    return imgs


# --- 3. build the bag type from short recordings (the method that tested accurate) -----------------
def build_from_recordings(name, real, cuts, empty, areas, log=print):
    """real = {"front": [place crops...], "back": [...]} from ~20 s recordings while the bag is
    slid/turned; cuts = cut-out bags for extra synthetic angles/positions. 80 % of the real photos
    + variations train the model, the other 20 % (never trained on) set the defect limit."""
    rng = np.random.default_rng(0)
    train, val = [], []
    for side in ("front", "back"):
        imgs = list(real[side])
        rng.shuffle(imgs)
        n_val = max(2, len(imgs) // 5)
        val += imgs[:n_val]
        train += imgs[n_val:]
    all_cuts = cuts["front"] + cuts["back"]
    synth = make_set(all_cuts, empty, areas, 10, seed=3) if all_cuts else []
    d = profile_dir(name)
    os.makedirs(os.path.join(d, "samples"), exist_ok=True)
    for k in range(0, len(train + synth), max(1, len(train + synth) // 12)):
        cv2.imwrite(os.path.join(d, "samples", f"example_{k}.jpg"), (train + synth)[k])
    log(f"learning from {len(train)} real photos + {len(synth)} variations; {len(val)} photos kept back to set the limit")
    model = anomaly.build_from_images(train + synth, val, os.path.join(d, "anomaly.pt"), margin=1.10, coreset=0.06,
                                      log=log, extra={"profile": safe_name(name)})
    meta = {"name": safe_name(name), "created": time.strftime("%Y-%m-%d %H:%M"),
            "front": len(real["front"]), "back": len(real["back"]), "pictures": len(train) + len(synth),
            "threshold": round(model.threshold, 3)}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log(f"bag type '{meta['name']}' ready (defect limit {model.threshold:.2f})")
    return meta


# --- (one-photo version: kept for reference - it flags too many good bags, see tests) --------------
def build_profile(name, cutouts, real_crops, empty, areas, log=print):
    """cutouts = {"front": [rgba...], "back": [rgba...]}; real_crops = the captured place crops."""
    d = profile_dir(name)
    os.makedirs(os.path.join(d, "samples"), exist_ok=True)
    all_cuts = cutouts["front"] + cutouts["back"]
    for k, c in enumerate(all_cuts):
        cv2.imwrite(os.path.join(d, f"cutout_{k}.png"), c)
    log(f"making pictures: {len(all_cuts)} photo(s) x {len(areas)} place(s) x {PER_PLACE} variations ...")
    train = make_set(all_cuts, empty, areas, PER_PLACE, seed=1) + list(real_crops)
    val = make_set(all_cuts, empty, areas, VAL_PER_PLACE, seed=99)
    for k in range(0, len(train), max(1, len(train) // 12)):              # a few examples to look at
        cv2.imwrite(os.path.join(d, "samples", f"example_{k}.jpg"), train[k])
    model = anomaly.build_from_images(train, val, os.path.join(d, "anomaly.pt"), margin=MARGIN, coreset=0.06, log=log,
                                      extra={"profile": safe_name(name)})
    real = [model.score(c)[0] for c in real_crops]
    meta = {"name": safe_name(name), "created": time.strftime("%Y-%m-%d %H:%M"),
            "front": len(cutouts["front"]), "back": len(cutouts["back"]), "pictures": len(train),
            "threshold": round(model.threshold, 3), "real_scores": [round(s, 3) for s in real]}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    bad_real = [s for s in real if s > model.threshold]
    log(f"bag type '{meta['name']}' ready - limit {model.threshold:.2f}; the captured good bags score "
        f"{', '.join(f'{s:.2f}' for s in real)}" + ("  (WARNING: a captured good bag is above the limit)" if bad_real else ""))
    return meta
