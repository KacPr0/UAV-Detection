#!/usr/bin/env python3
"""Ewaluacja detektora hotspotów (pvdet.hotspots) na adnotacjach LabelMe zbioru hf_thermal.

Użycie: .venv/bin/python scripts/eval_hotspots.py data/hf_thermal/raw [progi °C...]

Dopasowanie: ramki uznaje się za zgodne, gdy nachodzą na siebie po poszerzeniu ramki
z adnotacji o MARGIN px (hotspoty mają ~8 px, więc IoU jest zbyt surowe dla przesunięć o piksel).
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pvdet.hotspots import detect
from pvdet.io.radiometric import read_temperature

MARGIN = 2


def overlaps(a, b):
    return a[0] <= b[2] + MARGIN and b[0] <= a[2] + MARGIN and a[1] <= b[3] + MARGIN and b[1] <= a[3] + MARGIN


def gt_boxes(json_path):
    boxes = []
    for s in json.loads(json_path.read_text())["shapes"]:
        (x1, y1), (x2, y2) = s["points"]
        boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return boxes


def main(raw, thresholds):
    jsons = [j for j in Path(raw).rglob("*.json") if j.with_suffix(".JPG").exists()]
    with ThreadPoolExecutor(8) as ex:  # exiftool to osobny proces, więc wątki wystarczą
        temps = list(ex.map(lambda j: read_temperature(j.with_suffix(".JPG")), jsons))
    gts = [gt_boxes(j) for j in jsons]
    print(f"kadry={len(jsons)} hotspoty_w_adnotacjach={sum(map(len, gts))}")
    print("próg_°C  wykrycia  precision  recall  F1")
    for t in thresholds:
        tp_det = n_det = hit_gt = 0
        for temp, gt in zip(temps, gts):
            det = [s["bbox"] for s in detect(temp, delta_c=t) if s["type"] == "hotspot"]  # adnotacje F1 to tylko punktowe hotspoty
            n_det += len(det)
            tp_det += sum(any(overlaps(d, g) for g in gt) for d in det)
            hit_gt += sum(any(overlaps(d, g) for d in det) for g in gt)
        p, r = tp_det / max(n_det, 1), hit_gt / sum(map(len, gts))
        print(f"{t:7.1f}  {n_det:8d}  {p:9.3f}  {r:6.3f}  {2 * p * r / max(p + r, 1e-9):.3f}")


if __name__ == "__main__":
    main(sys.argv[1], [float(x) for x in sys.argv[2:]] or [2, 3, 4, 5, 7])
