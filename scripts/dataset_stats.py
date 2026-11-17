#!/usr/bin/env python3
"""Statystyki datasetu (tylko biblioteka standardowa). Nie uruchamia niczego z plikow.

Uzycie: /usr/bin/python3 -I scripts/dataset_stats.py <sciezka>

Rozpoznaje (automatycznie, mozna ich kilka w jednym katalogu):
  - YOLO: katalogi */labels/*.txt obok */images/ (podzial wg nazw katalogow train/val/test)
  - JSON mapujacy obraz -> {anomaly_class}: (format InfraredSolarModules)
  - klasyfikacja w folderach: katalog-klasa/*.jpg
Dla obrazow czyta wymiary z naglowkow PNG/JPEG (bez PIL).
"""
import json
import os
import struct
import sys
from collections import Counter, defaultdict

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def image_size(path):
    """Zwraca (szerokosc, wysokosc) z naglowka PNG/JPEG albo None."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24])
                return w, h
            if head[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    b = f.read(1)
                    while b and b != b"\xff":
                        b = f.read(1)
                    m = f.read(1)
                    while m == b"\xff":
                        m = f.read(1)
                    if not m:
                        return None
                    code = m[0]
                    if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
                        f.read(3)
                        h, w = struct.unpack(">HH", f.read(4))
                        return w, h
                    seglen = struct.unpack(">H", f.read(2))[0]
                    f.seek(seglen - 2, 1)
    except OSError:
        return None
    return None


def split_of(path):
    parts = [p.lower() for p in os.path.normpath(path).split(os.sep)]
    for s in ("train", "val", "valid", "test"):
        if s in parts:
            return "val" if s == "valid" else s
    return "(brak podzialu)"


def report_sizes(paths):
    sizes = Counter()
    unknown = 0
    for p in paths:
        s = image_size(p)
        if s is None:
            unknown += 1
        else:
            sizes[s] += 1
    return sizes, unknown


def yolo(root):
    print("== YOLO (labels/*.txt) ==")
    per_split = defaultdict(lambda: {"images": 0, "labels_files": 0, "objects": 0, "classes": Counter(),
                                     "no_label_images": 0, "sizes": Counter()})
    img_paths = []
    for dp, _, fns in os.walk(root):
        if os.path.basename(dp) != "images":
            continue
        for fn in fns:
            if fn.lower().endswith(IMG_EXT):
                img_paths.append(os.path.join(dp, fn))
    for ip in img_paths:
        sp = split_of(ip)
        base = os.path.splitext(os.path.basename(ip))[0]
        lab = os.path.join(os.path.dirname(os.path.dirname(ip)), "labels", base + ".txt")
        st = per_split[sp]
        st["images"] += 1
        if not os.path.exists(lab):
            st["no_label_images"] += 1
            continue
        st["labels_files"] += 1
        with open(lab) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    st["objects"] += 1
                    st["classes"][parts[0]] += 1
        s = image_size(ip)
        st["sizes"][s] += 1 if s else 0
    if not per_split:
        print("  brak")
        return
    for sp in sorted(per_split):
        st = per_split[sp]
        print(f"  [{sp}] obrazy={st['images']} pliki_labels={st['labels_files']} "
              f"bez_labels={st['no_label_images']} obiekty={st['objects']}")
        print(f"    klasy (id: liczba): {dict(sorted(st['classes'].items()))}")
        sz = {k: v for k, v in st["sizes"].items() if k}
        if sz:
            print(f"    rozmiary (w x h: n): {dict(Counter(sz).most_common(5))}")



def anomaly_json(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict) or not d or not all(isinstance(v, dict) and "anomaly_class" in v for v in d.values()):
        return False
    print(f"== JSON (anomaly_class): {path} ==")
    cnt = Counter(v["anomaly_class"] for v in d.values())
    print(f"  obrazy={len(d)}")
    for k, v in cnt.most_common():
        print(f"  {k}: {v}")
    return True


def class_folders(root):
    per_class = defaultdict(list)
    for dp, _, fns in os.walk(root):
        imgs = [f for f in fns if f.lower().endswith(IMG_EXT)]
        if imgs and os.path.basename(dp) not in ("images", "labels"):
            per_class[os.path.basename(dp)] += [os.path.join(dp, f) for f in imgs]
    if len(per_class) < 2:
        return False
    print("== Klasyfikacja w folderach (klasa = nazwa katalogu) ==")
    tot = 0
    for k in sorted(per_class):
        print(f"  {k}: {len(per_class[k])}")
        tot += len(per_class[k])
    sizes, unk = report_sizes([p for v in per_class.values() for p in v][:2000])
    print(f"  razem obrazow: {tot}; rozmiary (probka): {dict(sizes.most_common(3))}, nieczytelne: {unk}")
    return True


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = os.path.abspath(sys.argv[1])
    print(f"Sciezka: {root}")
    found = False
    if os.path.isfile(root) and root.endswith(".json"):
        found = anomaly_json(root)
        return 0 if found else 1
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            if fn.lower().endswith(".json") and os.path.getsize(p) < 200 * 1024 * 1024:
                try:
                    found |= anomaly_json(p)
                except (ValueError, KeyError, TypeError):
                    pass
    if any(os.path.basename(dp) == "labels" for dp, _, _ in os.walk(root)):
        yolo(root)
        found = True
    found |= class_folders(root)
    if not found:
        print("Nie rozpoznano formatu adnotacji.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
