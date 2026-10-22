#!/usr/bin/env python3
"""
Anomaly detection for paper bags (PatchCore with a WideResNet-50 backbone).

Learns ONLY from photos of GOOD bags and flags anything that does not look like a good bag:
wrong folds, damaged corners/edges, stains, holes, glue marks - including defect types that
were never photographed. Gives a score and a heatmap showing WHERE the problem is.

  python3 anomaly.py build                 # learn from bag_data/crops/good (+ calibrate the threshold)
  python3 anomaly.py test IMAGE [IMAGE...] # score images, writes *_heatmap.jpg next to each
"""
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
GOOD_DIR = os.path.join(HERE, "bag_data", "crops", "good")
MODEL_FILE = os.path.join(HERE, "bag_models", "anomaly.pt")
SIZE = (288, 384)                  # (width, height) the bag crop is resized to - portrait bags
CORESET = 0.10                     # keep 10 % of the good patches (greedy coreset)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Backbone(torch.nn.Module):
    """WideResNet-50 (ImageNet) - mid-level features (layer2 + layer3) describe texture and shape."""

    def __init__(self):
        super().__init__()
        net = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3

    @torch.no_grad()
    def forward(self, x):
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], 1)
        return F.avg_pool2d(f, 3, 1, 1)            # each patch also sees its neighbours


def prep(img_bgr):
    img = cv2.resize(img_bgr, SIZE, interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1)[None] / 255
    return ((t - MEAN) / STD).to(DEVICE)


def greedy_coreset(feats, n, proj_dim=128):
    """Pick n patches that cover the whole set of good patches as evenly as possible."""
    proj = torch.randn(feats.shape[1], proj_dim, device=feats.device) / proj_dim ** 0.5
    z = torch.cat([feats[i:i + 65536].float() @ proj for i in range(0, len(feats), 65536)])
    picked = [int(torch.randint(len(z), (1,)))]
    mind = torch.cdist(z, z[picked[-1]][None]).squeeze(1)
    for _ in range(n - 1):
        i = int(torch.argmax(mind))
        picked.append(i)
        mind = torch.minimum(mind, torch.cdist(z, z[i][None]).squeeze(1))
    return feats[picked]


class AnomalyModel:
    def __init__(self, path=MODEL_FILE):
        self.net = Backbone().to(DEVICE).eval()
        self.bank = self.threshold = None
        if os.path.exists(path):
            d = torch.load(path, map_location=DEVICE)
            self.bank, self.threshold = d["bank"].to(DEVICE), d["threshold"]
            self.good_scores = d.get("good_scores", [])

    @torch.no_grad()
    def patch_scores(self, img_bgr):
        f = self.net(prep(img_bgr))                                  # 1 x C x h x w
        h, w = f.shape[-2:]
        q = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]).half()
        d = torch.cdist(q.float(), self.bank.float()).min(1).values
        return d.reshape(h, w).cpu().numpy()

    def score(self, img_bgr):
        """(score, heatmap 0..1 at image size). score > threshold = not like a good bag."""
        m = self.patch_scores(img_bgr)
        m = cv2.GaussianBlur(cv2.resize(m, (img_bgr.shape[1], img_bgr.shape[0])), (0, 0), 6)
        return float(m.max()), m

    def heatmap(self, img_bgr, m):
        """Colour only the spots that are close to / above the defect limit."""
        lo, hi = self.threshold * 0.8, self.threshold * 1.4
        norm = np.clip((m - lo) / (hi - lo), 0, 1)
        col = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        alpha = (norm * 0.65)[..., None]
        return (img_bgr * (1 - alpha) + col * alpha).astype(np.uint8)


def build():
    files = sorted(glob.glob(os.path.join(GOOD_DIR, "*.png")))
    # only crops that match the current bag places (the two half-board places)
    import json
    areas = json.load(open(os.path.join(HERE, "bag_data", "areas.json")))["areas"]
    sizes = {(b[3], b[2]) for b in areas.values()}
    files = [f for f in files if any(abs(cv2.imread(f).shape[0] - h) < 25 and abs(cv2.imread(f).shape[1] - w) < 25 for h, w in sizes)]
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    n_val = max(5, len(files) // 5)
    val, train = files[:n_val], files[n_val:]
    print(f"good photos: {len(files)} (learn from {len(train)}, check on {len(val)})")
    model = AnomalyModel(path="/nonexistent")
    t = time.time()
    feats = []
    for f in train:
        x = model.net(prep(cv2.imread(f)))
        feats.append(x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]))
    feats = torch.cat(feats)
    n = max(1000, int(len(feats) * CORESET))
    print(f"{len(feats)} good patches -> keeping {n} (coreset) ...")
    model.bank = greedy_coreset(feats, n).half()
    good = [model.score(cv2.imread(f))[0] for f in val]
    # threshold: just above the worst GOOD bag the model has never seen
    model.threshold = float(max(good) * 1.10)
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    torch.save({"bank": model.bank.cpu(), "threshold": model.threshold, "good_scores": good,
                "size": SIZE, "built": time.strftime("%Y-%m-%d %H:%M")}, MODEL_FILE)
    print(f"built in {time.time() - t:.0f} s; unseen good bags score {min(good):.2f}..{max(good):.2f} "
          f"-> threshold {model.threshold:.2f};  saved {os.path.relpath(MODEL_FILE, HERE)}")


def build_from_images(train_imgs, val_imgs, out_path, margin=1.10, coreset=CORESET, log=print, extra=None):
    """Build a model from in-memory BGR images (used by the one-shot bag-type setup).
    The threshold is set just above the highest score of the GOOD validation images."""
    model = AnomalyModel(path="/nonexistent")
    t = time.time()
    feats = []
    keep = min(1.0, 250_000 / (len(train_imgs) * 1728))       # stay within GPU memory
    for img in train_imgs:
        x = model.net(prep(img))
        f = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        if keep < 1:
            f = f[torch.randperm(len(f), device=f.device)[:int(len(f) * keep)]]
        feats.append(f.half())
    feats = torch.cat(feats)
    n = max(1000, int(len(feats) * coreset))
    log(f"{len(train_imgs)} pictures, {len(feats)} patches -> keeping {n} ...")
    model.bank = greedy_coreset(feats, n).half()
    good = [model.score(img)[0] for img in val_imgs]
    model.threshold = float(max(good) * margin)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"bank": model.bank.cpu(), "threshold": model.threshold, "good_scores": good, "size": SIZE,
                "built": time.strftime("%Y-%m-%d %H:%M"), **(extra or {})}, out_path)
    log(f"built in {time.time() - t:.0f} s; good check pictures score {min(good):.2f}..{max(good):.2f} "
        f"-> limit {model.threshold:.2f}")
    return model


def test(paths):
    model = AnomalyModel()
    for p in paths:
        img = cv2.imread(p)
        t = time.time()
        s, m = model.score(img)
        dt = time.time() - t
        out = os.path.splitext(p)[0] + "_heatmap.jpg"
        cv2.imwrite(out, model.heatmap(img, m))
        print(f"{'DEFECT' if s > model.threshold else 'ok    '} score {s:.2f} (threshold {model.threshold:.2f}) "
              f"{dt * 1000:.0f} ms  {os.path.basename(p)}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        build()
    elif len(sys.argv) >= 3 and sys.argv[1] == "test":
        test(sys.argv[2:])
    else:
        print(__doc__)
