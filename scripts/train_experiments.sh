#!/usr/bin/env bash
# Treningi do rozdziału z badaniami (kolejno, jedno GPU). Wyniki: runs/<nazwa>/, log: runs/<nazwa>.log
# Pełna ścieżka w project=..., bo globalne ustawienia Ultralytics mogą wskazywać inny katalog.
set -uo pipefail
cd "$(dirname "$0")/.."
R="$PWD/runs"; Y=.venv/bin/yolo
COMMON="epochs=100 patience=20 device=mps project=$R exist_ok=True seed=0"
HS="hsv_h=0 hsv_s=0"  # kanały to temperatury, nie barwy

train() { local name=$1; shift; echo "== $name $(date +%H:%M)"; $Y detect train name=$name $COMMON "$@" > "$R/$name.log" 2>&1; echo "   koniec $(date +%H:%M), kod $?"; }

train panele_y11n      data=data/thermal_zenodo/data.yaml   model=yolo11n.pt imgsz=640  batch=16
train hotspot_paleta   data=data/hf_thermal/yolo_paleta/data.yaml model=yolo11n.pt imgsz=640 batch=16 $HS
train hotspot_y11n_1024 data=data/hf_thermal/yolo/data.yaml  model=yolo11n.pt imgsz=1024 batch=8  $HS
train hotspot_y11s     data=data/hf_thermal/yolo/data.yaml   model=yolo11s.pt imgsz=640  batch=16 $HS
