#!/usr/bin/env python3
"""Statystyki adnotacji LabelMe zbioru hf_thermal (liczby do rozdziału o danych).

Użycie: python3 -I scripts/labelme_stats.py data/hf_thermal/raw
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def stats(jsons):
    w, h, labels = [], [], Counter()
    for j in jsons:
        for s in json.loads(j.read_text())["shapes"]:
            (x1, y1), (x2, y2) = s["points"]
            w.append(abs(x2 - x1))
            h.append(abs(y2 - y1))
            labels[s["label"]] += 1
    med = lambda v: f"{min(v):.0f}/{statistics.median(v):.0f}/{max(v):.0f}"
    paired = sum(j.with_suffix(".JPG").exists() for j in jsons)
    return f"kadry={len(jsons)} z_temperaturą={paired} ramki={len(w)} etykiety={dict(labels)} szer.min/med/max={med(w)} wys.={med(h)}"


raw = Path(sys.argv[1])
for plot in sorted(p for p in raw.iterdir() if p.is_dir()):
    print(plot.name, stats(list(plot.rglob("*.json"))))
print("RAZEM", stats([j for p in raw.iterdir() if p.is_dir() for j in p.rglob("*.json")]))
