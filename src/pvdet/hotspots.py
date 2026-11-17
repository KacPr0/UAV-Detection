"""Bazowa (bez uczenia) detekcja hotspotów na mapie temperatur.

Nadwyżka = temperatura minus mediana z okna 31×31 px, czyli „o ile °C cieplej niż otoczenie”;
hotspot i zimne szczeliny między modułami są w oknie mniejszością, więc mediana daje temperaturę
panelu. Ziarnisty, nagrzany grunt między rzędami odcina maska paneli (panele są gładkie).
"""

import cv2
import numpy as np

MAX_HOTSPOT_AREA = 400  # px; większe plamy to przegrzane fragmenty modułów (np. dioda bocznikująca)


def panel_mask(temp: np.ndarray, max_std: float = 0.6, border: int = 5) -> np.ndarray:
    """Maska obszarów o gładkiej temperaturze (panele), z wypełnionymi dziurami i odciętą krawędzią.

    shortcut: heurystyka, przepuszcza gładki grunt i gubi część paneli — zastąpić detektorem paneli (etap 2).
    """
    mean = cv2.blur(temp, (9, 9))
    std = np.sqrt(np.maximum(cv2.blur(temp * temp, (9, 9)) - mean * mean, 0))
    mask = cv2.morphologyEx((std < max_std).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = cv2.drawContours(mask, contours, -1, 1, cv2.FILLED)  # dziury = hotspoty otoczone panelem
    return cv2.erode(mask, np.ones((2 * border + 1,) * 2, np.uint8))  # krawędź panel/grunt daje fałszywe wykrycia


def excess_over_background(temp: np.ndarray, window: int = 31) -> np.ndarray:
    # medianBlur dla okna > 5 działa tylko na uint8: kwantyzacja co 0,25 °C (zakres 64 °C od 0,5 percentyla)
    lo = np.percentile(temp, 0.5)
    q = np.clip((temp - lo) * 4, 0, 255).astype(np.uint8)
    return temp - (cv2.medianBlur(q, window).astype(np.float32) / 4 + lo)


def detect(temp: np.ndarray, delta_c: float = 4.0, min_area: int = 4) -> list[dict]:
    """Zwraca anomalie cieplne: typ (hotspot / przegrzany obszar), bbox (x1, y1, x2, y2), nadwyżka i maks. temperatura w °C."""
    excess = excess_over_background(temp)
    excess[panel_mask(temp) == 0] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats((excess > delta_c).astype(np.uint8))
    spots = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        region = labels[y:y + h, x:x + w] == i
        spots.append({
            "type": "hotspot" if area <= MAX_HOTSPOT_AREA else "przegrzany obszar",
            "bbox": (int(x), int(y), int(x + w - 1), int(y + h - 1)),
            "delta_c": float(excess[y:y + h, x:x + w][region].max()),
            "temp_c": float(temp[y:y + h, x:x + w][region].max()),
        })
    return sorted(spots, key=lambda s: -s["delta_c"])
