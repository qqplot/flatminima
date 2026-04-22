#!/bin/bash
# 실험 진행상황 모니터: 체크포인트 상태, 디스크, 훈련 프로세스 요약을 INTERVAL 간격으로 기록.
set -u
OUT="${1:-/data/flatminima/verl/logs/progress_monitor.log}"
INTERVAL="${INTERVAL:-60}"
CKPT_ROOT="${CKPT_ROOT:-/data/flatminima/verl/checkpoints}"
mkdir -p "$(dirname "$OUT")"
echo "[progress_monitor] pid=$$ interval=${INTERVAL}s out=$OUT start=$(date -Iseconds)" >> "$OUT"
while :; do
    ts=$(date -Iseconds)
    {
        echo "=== $ts ==="
        echo "--- disk (ckpt volume) ---"
        df -h "$CKPT_ROOT" 2>/dev/null | tail -n 1
        echo "--- checkpoints per experiment ---"
        shopt -s nullglob
        for d in "$CKPT_ROOT"/*/; do
            steps=$(ls -d "$d"global_step_* 2>/dev/null | sort -V | sed 's|.*global_step_||' | paste -sd, -)
            size=$(du -sh "$d" 2>/dev/null | awk '{print $1}')
            exp=$(basename "$d")
            echo "  $exp  size=$size  steps=[$steps]"
        done
        shopt -u nullglob
        echo "--- training procs ---"
        ps -eo pid,etime,pcpu,pmem,cmd 2>/dev/null | grep -E "torch.distributed|fsdp_|run_experiment|run_all_experiments" | grep -v grep || echo "  (none)"
    } >> "$OUT"
    sleep "$INTERVAL"
done
