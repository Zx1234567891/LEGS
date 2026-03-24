#!/usr/bin/env bash
# scripts/audit/legs_audit.sh
# Collect environment fingerprint + SBOM for baseline comparison between Dog & Server.
set -euo pipefail
OUT="_audit/$(hostname)-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

echo "[os]" > "$OUT/os.txt"
( uname -a && cat /etc/os-release ) >> "$OUT/os.txt" 2>&1 || true

echo "[gpu]" > "$OUT/gpu.txt"
nvidia-smi >> "$OUT/gpu.txt" 2>&1 || true

echo "[cuda]" > "$OUT/cuda.txt"
( command -v nvcc && nvcc --version ) >> "$OUT/cuda.txt" 2>&1 || true
( python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)' ) >> "$OUT/cuda.txt" 2>&1 || true

echo "[python]" > "$OUT/python.txt"
python3 -V >> "$OUT/python.txt" 2>&1 || true
python3 -m pip -V >> "$OUT/python.txt" 2>&1 || true
python3 -m pip freeze > "$OUT/pip_freeze.txt" 2>&1 || true

echo "[docker]" > "$OUT/docker.txt"
docker --version >> "$OUT/docker.txt" 2>&1 || true
docker compose version >> "$OUT/docker.txt" 2>&1 || true

echo "[sbom]" > "$OUT/sbom.txt"
if command -v syft >/dev/null 2>&1; then
  syft dir:. -o cyclonedx-json > "$OUT/sbom.cdx.json"
  echo "sbom: $OUT/sbom.cdx.json" >> "$OUT/sbom.txt"
else
  echo "syft not found — skipping SBOM generation" >> "$OUT/sbom.txt"
fi

echo "[done] Audit output: $OUT"
ls -la "$OUT/"
