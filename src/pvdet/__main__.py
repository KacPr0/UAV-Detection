"""Użycie: pvdet analizuj FOLDER [-o raport.html] [--prog 4.0] [--model best.pt] [--panele panele.pt]"""

import argparse
from pathlib import Path

from pvdet import hotspots, report
from pvdet.io.radiometric import read_temperature


def main():
    ap = argparse.ArgumentParser(prog="pvdet", description="Detekcja defektów modułów PV na kadrach termowizyjnych (R-JPEG).")
    ap.add_argument("polecenie", choices=["analizuj"])
    ap.add_argument("folder", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("raport.html"))
    ap.add_argument("--prog", type=float, default=None, help="metoda bazowa: nadwyżka nad tłem w °C (domyślnie 4); z --model: pewność (domyślnie 0,25)")
    ap.add_argument("--model", type=Path, help="wagi YOLO (best.pt); bez tej opcji — metoda bazowa")
    ap.add_argument("--panele", type=Path, help="wagi YOLO detektora paneli — numery paneli w raporcie")
    a = ap.parse_args()

    if a.model:
        from ultralytics import YOLO  # ciężki import tylko gdy potrzebny

        from pvdet import yolo
        net, prog = YOLO(a.model), a.prog or 0.25
        find = lambda temp: yolo.detect(net, temp, conf=prog)
    else:
        prog = a.prog or 4.0
        find = lambda temp: hotspots.detect(temp, delta_c=prog)
    if a.panele:
        from ultralytics import YOLO

        from pvdet import panels
        panel_net = YOLO(a.panele)
    images = sorted(a.folder.rglob("*_R.JPG"))
    if not images:
        ap.error(f"brak plików *_R.JPG w {a.folder}")
    frames = []
    for n, path in enumerate(images, 1):
        temp = read_temperature(path)
        spots = find(temp)
        if spots:
            boxes = panels.detect(panel_net, temp) if a.panele else []
            if boxes:
                panels.assign(spots, boxes)
            frames.append({"name": str(path.relative_to(a.folder)), "thumb": report.thumbnail(temp, spots, boxes), "spots": spots})
        print(f"\r{n}/{len(images)}", end="", flush=True)
    report.write(a.out, frames, len(images), f"model {a.model.name}, conf {prog}" if a.model else f"{prog} °C nad tłem")
    print(f"\nRaport: {a.out.resolve()}")


if __name__ == "__main__":
    main()
