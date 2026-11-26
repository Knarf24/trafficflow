"""Car counter checks (needs numpy + opencv). Run: python tests/test_vision.py"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import cv2  # noqa: E402

from trafficflow.vision import Counter  # noqa: E402

MODEL = os.path.join(ROOT, "models", "object_detection_nanodet_2022nov.onnx")
CLIP = os.path.join(ROOT, "web", "sample", "intersection.mp4")


def run_clip(step_s=0.25):
    c = Counter(MODEL)
    cap = cv2.VideoCapture(CLIP)
    fps = cap.get(cv2.CAP_PROP_FPS)
    every = max(1, round(fps * step_s))
    i = 0
    boxes_seen = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            r = c.process(frame, i / fps)
            boxes_seen += r["visible"]
            for b in r["boxes"]:
                assert b["kind"] in ("car", "motorcycle", "bus", "truck")
        i += 1
    return c, i / fps, boxes_seen


def test_counts_cars_on_the_sample_clip():
    c, duration, seen = run_clip()
    s = json.loads(c.summary(duration))
    assert seen > 50                      # vehicles are found in most frames
    assert 5 <= s["count"] <= 60          # a busy intersection, 30 seconds
    assert s["count"] == len(set(s["crossings"])) or s["count"] == len(s["crossings"])


def test_counter_reset_and_photo_mode():
    c = Counter(MODEL)
    cap = cv2.VideoCapture(CLIP)
    cap.set(cv2.CAP_PROP_POS_MSEC, 9000)
    ok, frame = cap.read()
    assert ok
    rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    h, w = rgba.shape[:2]
    out = json.loads(c.photo_rgba(rgba.tobytes(), w, h))
    assert out["visible"] >= 1
    c.reset()
    assert json.loads(c.summary(10))["count"] == 0


if __name__ == "__main__":
    test_counts_cars_on_the_sample_clip()
    test_counter_reset_and_photo_mode()
    print("vision tests passed")
