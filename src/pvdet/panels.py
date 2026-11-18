"""Detekcja modułów PV i przypisanie do nich defektów.

Model paneli uczono na szarych obrazach termo („biały = ciepło”, zbiór Zenodo), więc dostaje
temperaturę rozciągniętą do odcieni szarości. Moduły numeruje się wierszami, od lewej do prawej.
"""

import cv2
import numpy as np


def gray(temp: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(temp, [1, 99.5])
    g = np.clip((temp - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.merge([g, g, g])


def detect(model, temp: np.ndarray, conf: float = 0.25) -> list[tuple[int, int, int, int]]:
    boxes = [tuple(int(round(v)) for v in b) for b in model.predict(gray(temp), conf=conf, verbose=False)[0].boxes.xyxy.tolist()]
    if not boxes:
        return []
    row_h = np.median([b[3] - b[1] for b in boxes])
    return sorted(boxes, key=lambda b: (round(((b[1] + b[3]) / 2) / row_h), b[0]))  # wiersz, potem kolumna


def assign(spots: list[dict], panels: list[tuple]) -> list[dict]:
    """Dopisuje do każdego defektu numer panelu (od 1), w którym leży jego środek; None = poza panelami."""
    for s in spots:
        cx, cy = (s["bbox"][0] + s["bbox"][2]) / 2, (s["bbox"][1] + s["bbox"][3]) / 2
        s["panel"] = next((i for i, (x1, y1, x2, y2) in enumerate(panels, 1) if x1 <= cx <= x2 and y1 <= cy <= y2), None)
    return spots
