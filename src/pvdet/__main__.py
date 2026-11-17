"""Użycie: pvdet analizuj FOLDER [-o raport.html] [--prog 4.0]"""

import argparse
from pathlib import Path

from pvdet import hotspots, report
from pvdet.io.radiometric import read_temperature


def main():
    ap = argparse.ArgumentParser(prog="pvdet", description="Detekcja defektów modułów PV na kadrach termowizyjnych (R-JPEG).")
    ap.add_argument("polecenie", choices=["analizuj"])
    ap.add_argument("folder", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("raport.html"))
    ap.add_argument("--prog", type=float, default=4.0, help="nadwyżka nad tłem w °C (domyślnie 4)")
    a = ap.parse_args()

    images = sorted(a.folder.rglob("*_R.JPG"))
    if not images:
        ap.error(f"brak plików *_R.JPG w {a.folder}")
    frames = []
    for n, path in enumerate(images, 1):
        temp = read_temperature(path)
        spots = hotspots.detect(temp, delta_c=a.prog)
        if spots:
            frames.append({"name": str(path.relative_to(a.folder)), "thumb": report.thumbnail(temp, spots), "spots": spots})
        print(f"\r{n}/{len(images)}", end="", flush=True)
    report.write(a.out, frames, len(images), a.prog)
    print(f"\nRaport: {a.out.resolve()}")


if __name__ == "__main__":
    main()
