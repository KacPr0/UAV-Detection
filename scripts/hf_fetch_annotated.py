#!/usr/bin/env python3
"""Pobiera ze zbioru Hugging Face Manishsahu53/Solar-Panel-Thermal-Drone-UAV-Images tylko
adnotacje LabelMe (*.json) i ich pary *.JPG, czytając zip-y przez HTTP Range (bez pobierania 19 GB).

Użycie: .venv/bin/python -I scripts/hf_fetch_annotated.py data/hf_thermal/raw
Wznawialne: pliki o poprawnym rozmiarze są pomijane.
"""

import concurrent.futures as cf
import os
import re
import sys
import threading
import urllib.request
import zipfile

BASE = "https://huggingface.co/datasets/Manishsahu53/Solar-Panel-Thermal-Drone-UAV-Images/resolve/main/"
ZIPS = ["Plot_16_50MW.zip", "Plot_17_50MW.zip", "Plot_18_50MW.zip"]
CHUNK = 4 << 20  # zipfile czyta małymi kawałkami, a każdy odczyt to żądanie HTTP


class RangeFile:
    """Zdalny plik tylko do odczytu oparty o HTTP Range, z buforem — wystarcza dla zipfile."""

    def __init__(self, url):
        self.url, self.pos, self.buf, self.buf_start = url, 0, b"", 0
        cr = self._get(0, 0, meta=True)
        self.size = int(re.search(r"/(\d+)$", cr).group(1))

    def _get(self, start, end, meta=False):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.headers["Content-Range"] if meta else r.read()

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = [off, self.pos + off, self.size + off][whence]
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        out = []
        while n > 0:
            off = self.pos - self.buf_start
            if not 0 <= off < len(self.buf):
                self.buf, self.buf_start, off = self._get(self.pos, min(self.pos + CHUNK, self.size) - 1), self.pos, 0
            take = min(n, len(self.buf) - off)
            out.append(self.buf[off:off + take])
            self.pos += take
            n -= take
        return b"".join(out)


_local = threading.local()


def fetch(job):
    url, member, size, dst = job
    if os.path.isfile(dst) and os.path.getsize(dst) == size:
        return
    zips = _local.__dict__.setdefault("zips", {})  # ZipFile nie jest bezpieczny wątkowo
    zf = zips.get(url) or zips.setdefault(url, zipfile.ZipFile(RangeFile(url)))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst + ".part", "wb") as f:
        f.write(zf.read(member))
    os.replace(dst + ".part", dst)


def main(out):
    out = os.path.abspath(out)
    jobs = []
    for name in ZIPS:
        url = BASE + name
        files = [i for i in zipfile.ZipFile(RangeFile(url)).infolist() if not i.is_dir()]
        stems = {os.path.splitext(i.filename)[0] for i in files if i.filename.endswith(".json")}
        for i in files:
            if i.filename.endswith(".json") or (i.filename.endswith(".JPG") and os.path.splitext(i.filename)[0] in stems):
                dst = os.path.abspath(os.path.join(out, name[:-4], i.filename))
                if os.path.commonpath([out, dst]) != out:  # ścieżki z archiwum to dane z zewnątrz
                    raise ValueError(f"Niebezpieczna ścieżka w {name}: {i.filename}")
                jobs.append((url, i.filename, i.file_size, dst))
    print(f"{len(jobs)} plików", flush=True)
    with cf.ThreadPoolExecutor(8) as ex:
        for n, _ in enumerate(ex.map(fetch, jobs), 1):
            if n % 100 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
