# UAV-Detection

Detekcja defektów modułów fotowoltaicznych na skorelowanych obrazach RGB i termograficznych pozyskanych z platformy UAV — praca inżynierska, Informatyka Techniczna, AGH.

## Pipeline

```
wideo RGB + wideo termo
  → [1] ekstrakcja klatek i synchronizacja
  → [2] rejestracja RGB ↔ termo
  → [3] detekcja paneli
  → [4] detekcja defektów (termo / RGB)
  → [5] fuzja i grupowanie między klatkami
  → [6] raport
```

## Struktura

| Katalog | Zawartość |
|---|---|
| `src/pvdet/` | moduły pipeline'u i aplikacja desktopowa |
| `tests/` | testy |
| `scripts/` | pobieranie i przygotowanie danych, trening |
| `data/` | datasety (poza gitem) |
| `thesis/` | tekst pracy w LaTeX-u |

## Instalacja

```bash
python3.13 -m venv .venv.nosync && ln -s .venv.nosync .venv  # .nosync: iCloud nie synchronizuje środowiska
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Użycie

```bash
pvdet analizuj data/hf_thermal/raw/Plot_18_50MW -o outputs/raport.html   # raport HTML z wykrytymi anomaliami
python scripts/eval_hotspots.py data/hf_thermal/raw                      # precision/recall względem adnotacji
```

Wymaga `exiftool` (`brew install exiftool`) do odczytu temperatur z R-JPEG.
