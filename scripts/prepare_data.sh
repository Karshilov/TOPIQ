#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

NYX_URL="https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/NYX/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz"
SCALE_URL="https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/SCALE_LETKF/SDRBENCH-SCALE-98x1200x1200.tar.gz"
HURRICANE_URL="https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/Hurricane-ISABEL/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz"

# ── CESM-ATM (2D, 1800x3600) ──────────────────────────────────────────
# SDRBench updated this dataset after our experiments; the current tarball
# contains different data.  The original fields are bundled in the Docker
# image under /data/cesm_fields.tar.gz.
CESM_FIELDS="CLDTOT_1_1800_3600.f32 CLDHGH_1_1800_3600.f32 FLUT_1_1800_3600.f32 FLUTC_1_1800_3600.f32"
CESM_LOCAL="/data/cesm_fields.tar.gz"
if [ -f "$CESM_LOCAL" ]; then
    echo "[1/4] Extracting bundled CESM-ATM fields..."
    tar xzf "$CESM_LOCAL"
else
    echo "[1/4] ERROR: CESM fields not found at $CESM_LOCAL"
    echo "  The CESM-ATM dataset on SDRBench has been updated since our experiments."
    echo "  Please use the Docker image which bundles the original data."
    exit 1
fi
echo "  Done."

# ── NYX (3D, 512x512x512) ─────────────────────────────────────────────
echo "[2/4] Downloading NYX..."
wget -q --show-progress -N "$NYX_URL" -O nyx.tar.gz
echo "  Extracting needed fields..."
for f in dark_matter_density.f32 baryon_density.f32 velocity_x.f32; do
    tar xzf nyx.tar.gz --strip-components=1 --wildcards "*/$f"
done
rm -f nyx.tar.gz
echo "  Computing ln(1+x) for density fields..."
python3 -c "
import numpy as np
for name in ['dark_matter_density', 'baryon_density']:
    d = np.fromfile(f'{name}.f32', dtype=np.float32)
    np.log(1.0 + d.astype(np.float64)).astype(np.float32).tofile(f'{name}_log.f32')
    print(f'    {name}.f32 -> {name}_log.f32')
import os
os.remove('dark_matter_density.f32')
os.remove('baryon_density.f32')
"
echo "  Done."

# ── SCALE-LETKF (3D, 98x1200x1200) ───────────────────────────────────
SCALE_FIELDS="PRES-98x1200x1200.f32 T-98x1200x1200.f32"
echo "[3/4] Downloading SCALE-LETKF..."
wget -q --show-progress -N "$SCALE_URL" -O scale.tar.gz
echo "  Extracting needed fields..."
for f in $SCALE_FIELDS; do
    tar xzf scale.tar.gz --strip-components=1 --wildcards "*/$f"
done
rm -f scale.tar.gz
echo "  Done."

# ── Hurricane ISABEL (3D, 100x500x500) ────────────────────────────────
HURRICANE_FIELDS="TCf48.bin.f32 Uf48.bin.f32"
echo "[4/4] Downloading Hurricane ISABEL..."
wget -q --show-progress -N "$HURRICANE_URL" -O hurricane.tar.gz
echo "  Extracting needed fields..."
for f in $HURRICANE_FIELDS; do
    tar xzf hurricane.tar.gz --strip-components=1 --wildcards "*/$f"
done
rm -f hurricane.tar.gz
echo "  Done."

# ── Verify ─────────────────────────────────────────────────────────────
echo ""
echo "Verifying files..."
EXPECTED="CLDTOT_1_1800_3600.f32 CLDHGH_1_1800_3600.f32 FLUT_1_1800_3600.f32 FLUTC_1_1800_3600.f32 dark_matter_density_log.f32 baryon_density_log.f32 velocity_x.f32 PRES-98x1200x1200.f32 T-98x1200x1200.f32 TCf48.bin.f32 Uf48.bin.f32"
MISSING=0
for f in $EXPECTED; do
    if [ -f "$f" ]; then
        SZ=$(du -h "$f" | cut -f1)
        echo "  OK  $f  ($SZ)"
    else
        echo "  MISSING  $f"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -eq 0 ]; then
    echo ""
    echo "All 11 fields ready in $(pwd)"
    echo "Usage: python experiments/prediction_accuracy.py --data-dir $(pwd)"
else
    echo ""
    echo "WARNING: $MISSING file(s) missing."
    exit 1
fi
