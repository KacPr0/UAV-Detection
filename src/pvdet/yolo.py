"""Detekcja hotspotów siecią YOLO na mapie temperatur.

Sieć nie dostaje kolorowego podglądu z palety, tylko obraz zbudowany z temperatur:
kanał 0 i 2 — temperatura rozciągnięta do 0–255 w obrębie kadru, kanał 1 — nadwyżka
nad lokalnym tłem (0–10 °C → 0–255), ta sama co w metodzie bazowej.
"""

import cv2
import numpy as np

from pvdet.hotspots import excess_over_background


def encode(temp: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(temp, [1, 99.5])
    t = np.clip((temp - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    e = np.clip(excess_over_background(temp) * 25.5, 0, 255).astype(np.uint8)
    return cv2.merge([t, e, t])


def detect(model, temp: np.ndarray, conf: float = 0.25) -> list[dict]:
    """Ten sam format wyniku co pvdet.hotspots.detect."""
    boxes = model.predict(encode(temp), conf=conf, verbose=False)[0].boxes
    spots = []
    for (x1, y1, x2, y2), score in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
        x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
        region = temp[y1:y2 + 1, x1:x2 + 1]
        spots.append({"type": "hotspot", "bbox": (x1, y1, x2, y2), "score": score,
                      "temp_c": float(region.max()), "delta_c": float(region.max() - np.median(temp))})
    return spots
