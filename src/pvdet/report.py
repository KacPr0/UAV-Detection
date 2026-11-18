"""Raport HTML z wykrytych anomalii — jeden samodzielny plik z miniaturami w base64."""

import base64
import html
from datetime import datetime

import cv2
import numpy as np

COLORS = {"hotspot": (255, 255, 0), "przegrzany obszar": (0, 255, 255)}  # BGR


def thumbnail(temp: np.ndarray, spots: list[dict]) -> str:
    lo, hi = np.percentile(temp, [1, 99.9])
    img = cv2.applyColorMap(np.clip((temp - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    for n, s in enumerate(spots, 1):
        x1, y1, x2, y2 = s["bbox"]
        cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), COLORS[s["type"]], 1)
        cv2.putText(img, str(n), (x1, max(y1 - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS[s["type"]], 1)
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()


def write(path, frames: list[dict], n_analyzed: int, method: str) -> None:
    """frames: [{"name", "thumb", "spots"}] — tylko kadry z wykryciami."""
    counts = {t: sum(s["type"] == t for f in frames for s in f["spots"]) for t in COLORS}
    sections = []
    for f in frames:
        rows = "".join(
            f"<tr><td>{n}</td><td>{s['type']}</td><td>{s['temp_c']:.1f}</td><td>+{s['delta_c']:.1f}</td>"
            f"<td>{s['bbox']}</td></tr>" for n, s in enumerate(f["spots"], 1))
        sections.append(
            f"<section><h2>{html.escape(f['name'])}</h2><img src='data:image/jpeg;base64,{f['thumb']}'>"
            f"<table><tr><th>#</th><th>typ</th><th>maks. °C</th><th>nad tłem °C</th><th>bbox (x1, y1, x2, y2)</th></tr>"
            f"{rows}</table></section>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""<!doctype html><meta charset="utf-8"><title>Raport defektów PV</title>
<style>body{{font-family:sans-serif;max-width:1100px;margin:auto;padding:16px}}
section{{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start;border-top:1px solid #ccc;padding:12px 0}}
h2{{width:100%;font-size:1rem;margin:0}} img{{max-width:100%}} td,th{{padding:2px 8px;text-align:left}}</style>
<h1>Raport defektów modułów PV</h1>
<p>{datetime.now():%Y-%m-%d %H:%M} · kadrów: {n_analyzed} · z wykryciami: {len(frames)} ·
hotspotów: {counts['hotspot']} · przegrzanych obszarów: {counts['przegrzany obszar']} · metoda: {method}</p>
{''.join(sections)}""", encoding="utf-8")
