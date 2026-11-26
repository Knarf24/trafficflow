"""Camera car counter: finds vehicles in video frames or a photo and counts them.

* Detection: NanoDet-Plus (OpenCV Model Zoo, trained on the COCO dataset), run
  with OpenCV's dnn module. Only the vehicle classes are kept: car, motorcycle,
  bus and truck.
* Tracking: each new box is matched to the nearest track from the previous
  frames (centroid distance), so the same car is not counted twice.
* Counting: a car is counted once when its track crosses the counting line.
  Counts are grouped per minute of video, which can then be used as arrivals
  in the simulator and the forecast.

A pretrained detector is used as is; it was not trained or tuned for this app.
Top-down or night footage, heavy rain and cars hidden behind each other cause
missed or extra counts, so check the boxes drawn on the video.
"""

from __future__ import annotations

import json
import math

import cv2
import numpy as np

VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def _letterbox(img, size):
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    out = np.zeros((size, size, 3), np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, s, left, top


class NanoDet:
    """NanoDet-Plus (416x416). Output order differs between OpenCV builds, so
    the class-score and box outputs are matched up by their shapes."""

    size = 416
    reg_max = 7

    def __init__(self, model_path):
        self.net = cv2.dnn.readNet(model_path)
        self.names = self.net.getUnconnectedOutLayersNames()
        self.mean = np.array([103.53, 116.28, 123.675], np.float32)
        self.std = np.array([57.375, 57.12, 58.395], np.float32)
        self.project = np.arange(self.reg_max + 1, dtype=np.float32)
        self._anchors = {}

    def _anchor(self, n_cells):
        if n_cells not in self._anchors:
            side = int(round(math.sqrt(n_cells)))
            stride = self.size / side
            xv, yv = np.meshgrid(np.arange(side) * stride, np.arange(side) * stride)
            a = np.column_stack((xv.ravel(), yv.ravel())).astype(np.float32) + 0.5 * (stride - 1)
            self._anchors[n_cells] = (a, stride)
        return self._anchors[n_cells]

    def detect(self, bgr, min_score=0.35, classes=VEHICLES):
        img, s, left, top = _letterbox(bgr, self.size)
        blob = cv2.dnn.blobFromImage((img.astype(np.float32) - self.mean) / self.std)
        self.net.setInput(blob)
        outs = [o.reshape(o.shape[-2], o.shape[-1]) if o.ndim == 3 else o for o in self.net.forward(self.names)]
        cls_outs = sorted((o for o in outs if o.shape[-1] == 80), key=lambda o: -o.shape[0])
        reg_outs = sorted((o for o in outs if o.shape[-1] == 4 * (self.reg_max + 1)), key=lambda o: -o.shape[0])
        boxes, scores, ids = [], [], []
        wanted = np.array(sorted(classes))
        for cls, reg in zip(cls_outs, reg_outs):
            anc, stride = self._anchor(cls.shape[0])
            cid = cls.argmax(1)
            conf = cls[np.arange(len(cid)), cid]
            keep = (conf >= min_score) & np.isin(cid, wanted)
            if not keep.any():
                continue
            r = reg[keep].reshape(-1, self.reg_max + 1)
            e = np.exp(r - r.max(1, keepdims=True))
            dist = ((e / e.sum(1, keepdims=True)) @ self.project).reshape(-1, 4) * stride
            a = anc[keep]
            x1, y1 = a[:, 0] - dist[:, 0], a[:, 1] - dist[:, 1]
            x2, y2 = a[:, 0] + dist[:, 2], a[:, 1] + dist[:, 3]
            boxes += np.column_stack((x1, y1, x2 - x1, y2 - y1)).tolist()
            scores += conf[keep].tolist()
            ids += cid[keep].tolist()
        out = []
        if boxes:
            # class-agnostic suppression: a car is often also scored as a truck
            for i in np.array(cv2.dnn.NMSBoxes(boxes, scores, min_score, 0.45)).ravel():
                x, y, w, h = boxes[i]
                out.append({"kind": VEHICLES.get(int(ids[i]), str(ids[i])), "score": round(float(scores[i]), 3),
                            "box": [round((x - left) / s, 1), round((y - top) / s, 1),
                                    round(w / s, 1), round(h / s, 1)]})
        return out


class Counter:
    """Tracks vehicles across frames and counts line crossings."""

    def __init__(self, model_path="models/object_detection_nanodet_2022nov.onnx"):
        self.model_path = model_path
        self.det = None
        self.settings = {"line": 0.5, "direction": "vertical", "zone": [0.3, 1.0], "min_score": 0.35,
                         "max_jump": 0.12, "min_size": 0.04}
        self.reset()

    def detector(self):
        if self.det is None:
            self.det = NanoDet(self.model_path)
        return self.det

    def configure(self, settings_json):
        self.settings.update(json.loads(settings_json))
        return json.dumps(self.settings)

    def reset(self):
        self.tracks = {}          # id -> {"c": (x, y), "seen": t, "counted": bool, "side": -1/0/1}
        self.next_id = 1
        self.crossings = []       # seconds into the video when a vehicle crossed
        self.peak_visible = 0

    def _side(self, c, w, h):
        line = self.settings["line"]
        # "vertical" means a vertical line (cars move sideways across it)
        v = c[0] / w if self.settings["direction"] == "vertical" else c[1] / h
        return -1 if v < line else 1

    def process(self, bgr, t):
        """One frame at t seconds. Returns boxes, track ids and running counts."""
        h, w = bgr.shape[:2]
        st = self.settings
        dets = [d for d in self.detector().detect(bgr, st["min_score"])
                if max(d["box"][2] / w, d["box"][3] / h) >= st["min_size"]]
        max_jump = st["max_jump"] * math.hypot(w, h)
        zone_lo, zone_hi = st["zone"]
        # forget tracks not seen for 1.5 seconds
        self.tracks = {k: v for k, v in self.tracks.items() if t - v["seen"] <= 1.5}
        free = set(self.tracks)
        for d in sorted(dets, key=lambda d: -d["score"]):
            x, y, bw, bh = d["box"]
            c = (x + bw / 2, y + bh / 2)
            best, best_dist = None, max_jump
            for k in free:
                tr = self.tracks[k]
                dt = t - tr["seen"]
                guess = (tr["c"][0] + tr["v"][0] * dt, tr["c"][1] + tr["v"][1] * dt)
                dist = math.hypot(c[0] - guess[0], c[1] - guess[1])
                if dist < best_dist:
                    best, best_dist = k, dist
            side = self._side(c, w, h)
            across = c[1] / h if st["direction"] == "vertical" else c[0] / w
            in_zone = zone_lo <= across <= zone_hi
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = {"c": c, "v": (0.0, 0.0), "seen": t, "counted": False, "side": side}
            else:
                free.discard(best)
                tr = self.tracks[best]
                dt = max(t - tr["seen"], 1e-3)
                if not tr["counted"] and tr["side"] != side and in_zone:
                    tr["counted"] = True
                    self.crossings.append(round(t, 2))
                v = ((c[0] - tr["c"][0]) / dt, (c[1] - tr["c"][1]) / dt)
                tr.update(c=c, seen=t, side=side, v=(0.5 * tr["v"][0] + 0.5 * v[0], 0.5 * tr["v"][1] + 0.5 * v[1]))
            d["id"] = best
            d["counted"] = self.tracks[best]["counted"]
        self.peak_visible = max(self.peak_visible, len(dets))
        return {"t": t, "boxes": dets, "visible": len(dets), "count": len(self.crossings)}

    def process_rgba(self, data, width, height, t):
        rgba = np.frombuffer(bytes(data), np.uint8).reshape(height, width, 4)
        return json.dumps(self.process(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR), t))

    def photo_rgba(self, data, width, height):
        """Count the vehicles in one photo (e.g. the current queue)."""
        rgba = np.frombuffer(bytes(data), np.uint8).reshape(height, width, 4)
        dets = self.detector().detect(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR), self.settings["min_score"])
        return json.dumps({"boxes": dets, "visible": len(dets)})

    def summary(self, duration_s):
        """Crossings per minute of video and the implied arrival rate."""
        minutes = max(1, math.ceil(duration_s / 60.0))
        per_min = [0] * minutes
        for t in self.crossings:
            per_min[min(int(t // 60), minutes - 1)] += 1
        rate = len(self.crossings) / max(duration_s / 3600.0, 1e-9)
        return json.dumps({"count": len(self.crossings), "per_minute": per_min, "duration_s": duration_s,
                           "rate_per_hour": rate, "crossings": self.crossings, "peak_visible": self.peak_visible})
