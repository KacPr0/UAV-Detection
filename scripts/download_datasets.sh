#!/usr/bin/env bash
# Pobiera publiczne datasety PV do data/<nazwa>/raw/ (data/ jest poza gitem).
# Uzycie: scripts/download_datasets.sh [thermal_zenodo|pvf10|infrared_solar_modules|hf_thermal|all]
# Roboflow (defekty RGB) wymaga konta — pobierz ręcznie: https://universe.roboflow.com/solarpaneldataset/solarpanel-ggmtm
# Archiwa sa tylko rozpakowywane (unzip), nic z nich nie jest uruchamiane.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$REPO/data"

fetch() {  # fetch <url> <plik docelowy>; pomija, jesli plik juz istnieje
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ -s "$dest" ]]; then
    echo "  istnieje: $dest"
    return 0
  fi
  echo "  pobieram: $url"
  curl -fL --retry 5 --retry-delay 5 --connect-timeout 30 -C - -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

# 1. Zenodo 16420123 (CC-BY-4.0), ~15 MB, adnotacje paneli (format do ustalenia po rozpakowaniu)
thermal_zenodo() {
  echo "[thermal_zenodo]"
  local raw="$DATA/thermal_zenodo/raw"
  local zip="$raw/Thermal_PV_Panel_Detection_Dataset_for_UAV_Inspection.zip"
  fetch "https://zenodo.org/api/records/16420123/files/Thermal%20PV%20Panel%20Detection%20Dataset%20for%20UAV%20Inspection.zip/content" "$zip"
  unzip -q -n "$zip" -d "$raw"
}

# 2. PVF-10 (wangbobby1026/PVF-Dataset): PVF-10.zip to plik LFS; media.githubusercontent.com/media/ serwuje binarke.
#    Licencja: tylko niekomercyjne badania. Suma sha256 z pliku wskaznika LFS.
pvf10() {
  echo "[pvf10]"
  local raw="$DATA/pvf10/raw"
  local zip="$raw/PVF-10.zip"
  local sha="5bbedd95fd99b98198fcea84017c9243af8d4e8aeef307a3bd77200919195189"
  fetch "https://media.githubusercontent.com/media/wangbobby1026/PVF-Dataset/master/PVF-10.zip" "$zip"
  echo "$sha  $zip" | shasum -a 256 -c -
  fetch "https://raw.githubusercontent.com/wangbobby1026/PVF-Dataset/master/Readme.txt" "$raw/Readme.txt"
  fetch "https://raw.githubusercontent.com/wangbobby1026/PVF-Dataset/master/LICENSE.md" "$raw/LICENSE.md"
  unzip -q -n "$zip" -d "$raw"
}

# 3. InfraredSolarModules (RaptorMaps, MIT): zip z obrazami 24x40 i module_metadata.json
infrared_solar_modules() {
  echo "[infrared_solar_modules]"
  local raw="$DATA/infrared_solar_modules/raw"
  local zip="$raw/2020-02-14_InfraredSolarModules.zip"
  fetch "https://raw.githubusercontent.com/RaptorMaps/InfraredSolarModules/master/2020-02-14_InfraredSolarModules.zip" "$zip"
  unzip -q -n "$zip" -d "$raw"
  fetch "https://raw.githubusercontent.com/RaptorMaps/InfraredSolarModules/master/LICENSE" "$raw/LICENSE"
}

# 4. Hugging Face Manishsahu53/Solar-Panel-Thermal-Drone-UAV-Images (Apache-2.0): tylko kadry z adnotacjami, ~2,6 GB
hf_thermal() {
  echo "[hf_thermal]"
  "$REPO/.venv/bin/python" -I "$REPO/scripts/hf_fetch_annotated.py" "$DATA/hf_thermal/raw"
}

case "${1:-all}" in
  thermal_zenodo) thermal_zenodo ;;
  pvf10) pvf10 ;;
  infrared_solar_modules) infrared_solar_modules ;;
  hf_thermal) hf_thermal ;;
  all)
    thermal_zenodo
    pvf10
    infrared_solar_modules
    hf_thermal
    ;;
  *) echo "Nieznany zbior: $1" >&2; exit 2 ;;
esac
