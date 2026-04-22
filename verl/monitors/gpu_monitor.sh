#!/bin/bash
# GPU monitor: rocm-smi 를 INTERVAL 간격으로 실행하고 타임스탬프 포함 텍스트 로그에 누적.
# 끝날 때는 부모 프로세스 종료(HUP/TERM)에 자연 종료.
set -u
OUT="${1:-/data/flatminima/verl/logs/gpu_monitor.log}"
INTERVAL="${INTERVAL:-30}"
mkdir -p "$(dirname "$OUT")"
echo "[gpu_monitor] pid=$$ interval=${INTERVAL}s out=$OUT start=$(date -Iseconds)" >> "$OUT"
while :; do
    ts=$(date -Iseconds)
    {
        echo "=== $ts ==="
        rocm-smi --showtemp --showpower --showuse --showmeminfo vram 2>&1
    } >> "$OUT"
    sleep "$INTERVAL"
done
