#!/bin/bash
# Disk pressure monitor: 짧은 주기로 ckpt 볼륨 여유량을 찍고, 임계치 이하로 떨어지면
# WARN/CRIT 태그와 함께 로그 남김. 훈련이 ckpt 저장 중에 순간적으로 디스크를 크게 먹을 수 있어
# progress_monitor(60s)보다 잦은 간격(기본 15s)으로 체크.
#
# 추가 기능 1: CRIT 상태에서 AUTO_ACTION=prune 이면 각 ckpt 디렉터리에서
# 최신 KEEP_RECENT(기본 2)개만 남기고 나머지 global_step_* 를 삭제.
#
# 추가 기능 2: /data/flatminima 전체 크기(FLAT_ROOT)가 FLAT_MAX_GB(기본 90)를 초과하면
# run_all_experiments.sh 및 하위 훈련 프로세스를 SIGTERM 으로 정지. 중복 호출 방지.
set -u
OUT="${1:-/data/flatminima/verl/logs/disk_monitor.log}"
INTERVAL="${INTERVAL:-15}"
CKPT_ROOT="${CKPT_ROOT:-/data/flatminima/verl/checkpoints}"
WARN_GB="${WARN_GB:-30}"
CRIT_GB="${CRIT_GB:-12}"
AUTO_ACTION="${AUTO_ACTION:-}"
KEEP_RECENT="${KEEP_RECENT:-2}"
FLAT_ROOT="${FLAT_ROOT:-/data/flatminima}"
FLAT_MAX_GB="${FLAT_MAX_GB:-90}"
RUN_ALL_PIDFILE="${RUN_ALL_PIDFILE:-/data/flatminima/verl/logs/run_all_experiments.pid}"
KILL_GUARD="${KILL_GUARD:-/data/flatminima/verl/logs/disk_monitor.killed}"
mkdir -p "$(dirname "$OUT")"
echo "[disk_monitor] pid=$$ interval=${INTERVAL}s warn=${WARN_GB}G crit=${CRIT_GB}G flat_max=${FLAT_MAX_GB}G flat_root=${FLAT_ROOT} auto_action='${AUTO_ACTION}' start=$(date -Iseconds)" >> "$OUT"

prune_old() {
    shopt -s nullglob
    for d in "$CKPT_ROOT"/*/; do
        mapfile -t all < <(ls -d "$d"global_step_* 2>/dev/null | sort -V)
        local n=${#all[@]}
        (( n <= KEEP_RECENT )) && continue
        local drop=$(( n - KEEP_RECENT ))
        for (( i=0; i<drop; i++ )); do
            echo "[disk_monitor]   AUTO-PRUNE rm -rf ${all[i]}"
            rm -rf "${all[i]}"
        done
    done
    shopt -u nullglob
}

stop_training() {
    # idempotent: guard 파일이 있으면 재호출 스킵
    [ -e "$KILL_GUARD" ] && { echo "[disk_monitor]   (kill already issued; skipping)"; return 0; }
    local run_all_pid=""
    [ -f "$RUN_ALL_PIDFILE" ] && run_all_pid=$(cat "$RUN_ALL_PIDFILE" 2>/dev/null | tr -dc '0-9')
    if [ -n "$run_all_pid" ] && kill -0 "$run_all_pid" 2>/dev/null; then
        echo "[disk_monitor]   SIGTERM run_all_experiments pid=$run_all_pid"
        kill -TERM "$run_all_pid" 2>/dev/null || true
        # run_all의 자식들(run_experiment.sh)도 같이
        pkill -TERM -P "$run_all_pid" 2>/dev/null || true
    else
        echo "[disk_monitor]   (no live run_all pid in $RUN_ALL_PIDFILE)"
    fi
    # 혹시 남은 trainer 프로세스들도 명시적으로 정리
    pkill -TERM -f "run_experiment.sh"        2>/dev/null || true
    pkill -TERM -f "torch.distributed.run"    2>/dev/null || true
    pkill -TERM -f "verl.trainer.fsdp_"       2>/dev/null || true
    date -Iseconds > "$KILL_GUARD"
    echo "[disk_monitor]   kill guard written: $KILL_GUARD (삭제하면 다시 작동)"
}

while :; do
    ts=$(date -Iseconds)
    line=$(df -BG --output=avail,used,size,pcent,target "$CKPT_ROOT" 2>/dev/null | tail -n 1)
    avail_gb=$(echo "$line" | awk '{print $1}' | tr -dc '0-9')
    pct=$(echo "$line" | awk '{print $4}')

    # /data/flatminima 자체 사용량 측정 (빠름, <50ms)
    flat_bytes=$(du -sb "$FLAT_ROOT" 2>/dev/null | awk '{print $1}')
    flat_gb=$(( (flat_bytes + 1073741823) / 1073741824 ))  # ceil GiB

    if [ -z "$avail_gb" ]; then
        echo "[$ts] WARN could not read df for $CKPT_ROOT (flat=${flat_gb}G)" >> "$OUT"
    elif (( avail_gb < CRIT_GB )); then
        echo "[$ts] CRIT avail=${avail_gb}G used=${pct} flat=${flat_gb}G (threshold avail<${CRIT_GB}G) | $line" >> "$OUT"
        if [ "$AUTO_ACTION" = "prune" ]; then
            echo "[$ts] CRIT AUTO_ACTION=prune — keeping last ${KEEP_RECENT} per ckpt dir" >> "$OUT"
            prune_old >> "$OUT" 2>&1
        fi
    elif (( avail_gb < WARN_GB )); then
        echo "[$ts] WARN avail=${avail_gb}G used=${pct} flat=${flat_gb}G (threshold avail<${WARN_GB}G)" >> "$OUT"
    else
        echo "[$ts] OK   avail=${avail_gb}G used=${pct} flat=${flat_gb}G" >> "$OUT"
    fi

    if (( flat_gb > FLAT_MAX_GB )); then
        echo "[$ts] CRIT flat=${flat_gb}G > FLAT_MAX_GB=${FLAT_MAX_GB}G — stopping training" >> "$OUT"
        stop_training >> "$OUT" 2>&1
    fi

    sleep "$INTERVAL"
done
