"""Odczyt temperatur z radiometrycznych R-JPEG (DJI Zenmuse XT2, rdzeń FLIR).

R-JPEG zawiera surowy obraz termiczny (16-bit TIFF) i stałe Plancka w segmentach FLIR,
wydobywane exiftoolem. Model FLIR, emisyjność = 1, bez korekty atmosfery:

    T [K] = B / ln( R1 / (R2 * (raw + O)) + F )
"""

import base64
import json
import subprocess

import cv2
import numpy as np


def raw_to_celsius(raw: np.ndarray, r1, r2, b, f, o) -> np.ndarray:
    return (b / np.log(r1 / (r2 * (raw.astype(np.float64) + o)) + f) - 273.15).astype(np.float32)


def read_temperature(path) -> np.ndarray:
    """Mapa temperatur w °C (float32, H×W) z radiometrycznego R-JPEG."""
    out = subprocess.run(
        ["exiftool", "-j", "-n", "-b", "-RawThermalImage",
         "-PlanckR1", "-PlanckR2", "-PlanckB", "-PlanckF", "-PlanckO", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    m = json.loads(out)[0]
    if "RawThermalImage" not in m:
        raise ValueError(f"{path}: to nie jest radiometryczny R-JPEG")
    blob = base64.b64decode(m["RawThermalImage"].removeprefix("base64:"))
    # shortcut: tylko TIFF (XT2); inne kamery FLIR zapisują PNG z odwróconymi bajtami — dodać przy nowej kamerze
    raw = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED)
    return raw_to_celsius(raw, m["PlanckR1"], m["PlanckR2"], m["PlanckB"], m["PlanckF"], m["PlanckO"])
