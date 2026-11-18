#!/usr/bin/env python3
"""Zbiór hf_thermal → format YOLO (obrazy z pvdet.yolo.encode + etykiety), podział według farm.

Użycie: .venv/bin/python scripts/prepare_yolo.py data/hf_thermal/raw data/hf_thermal/yolo
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from pvdet.io.radiometric import read_temperature
from pvdet.yolo import encode

SPLITS = {"Plot_17_50MW": "train", "Plot_16_50MW": "val", "Plot_18_50MW": "test"}  # ocena na niewidzianej farmie


def convert(job):
    j, raw, out, split = job
    temp = read_temperature(j.with_suffix(".JPG"))
    h, w = temp.shape
    name = j.relative_to(raw).with_suffix("").as_posix().replace("/content/Dataset/", "/").replace("/", "__")  # nazwy DJI_XXXX powtarzają się między lotami
    lines = []
    for s in json.loads(j.read_text())["shapes"]:
        (x1, y1), (x2, y2) = s["points"]
        x1, x2, y1, y2 = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:  # ramki 1 px to szum adnotacji
            continue
        lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    cv2.imwrite(str(out / "images" / split / f"{name}.png"), encode(temp))
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
    return split, len(lines)


def main(raw, out):
    raw, out = Path(raw), Path(out)
    jobs = []
    for plot, split in SPLITS.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        jobs += [(j, raw, out, split) for j in (raw / plot).rglob("*.json") if j.with_suffix(".JPG").exists()]
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(convert, jobs))
    for split in SPLITS.values():
        r = [n for s, n in results if s == split]
        print(f"{split}: kadry={len(r)} ramki={sum(r)}")
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: hotspot\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
