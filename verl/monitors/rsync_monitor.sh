#!/bin/bash
# Monitor rsync_transfer progress. 30초 주기로:
#   - transfer 프로세스 생존 여부
#   - local transfer log tail (rsync 진행률/속도)
#   - remote destination 크기 (목표 36G 대비 %)
#   - remote filesystem 여유
# 종료 조건: ALL DONE 라인 감지 시 1회 최종 요약 후 exit (persistent 가능하도록 FOREVER 지원)
set -u
OUT="${1:-/data/flatminima/verl/logs/rsync_monitor.log}"
TRANSFER_LOG="${TRANSFER_LOG:-/data/flatminima/verl/logs/rsync_transfer.log}"
TRANSFER_PID_FILE="${TRANSFER_PID_FILE:-/data/flatminima/verl/logs/rsync_transfer.pid}"
DEST_HOST="${DEST_HOST:-kyubyungchae@147.47.200.22}"
DEST_PATH="${DEST_PATH:-/shared/s2/lab01/qqplot/flatminima_ckpt}"
INTERVAL="${INTERVAL:-30}"
TOTAL_GB_TARGET="${TOTAL_GB_TARGET:-36}"
FOREVER="${FOREVER:-0}"
exec >> "$OUT" 2>&1

echo ""
echo "[$(date -Iseconds)] rsync_monitor started pid=$$ interval=${INTERVAL}s"

check_ssh_remote_size() {
    # single ssh call로 원격 du + df 받기
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$DEST_HOST" \
        "du -sBG $DEST_PATH 2>/dev/null | awk '{print \$1}'; df -BG /shared/s2/lab01/qqplot 2>/dev/null | tail -1 | awk '{print \$4}'" 2>/dev/null
}

while :; do
    ts=$(date -Iseconds)
    # transfer 프로세스 생존?
    alive="no"
    tpid=$(cat "$TRANSFER_PID_FILE" 2>/dev/null | tr -dc '0-9')
    if [ -n "$tpid" ] && kill -0 "$tpid" 2>/dev/null; then
        alive="yes"
        etime=$(ps -p "$tpid" -o etime= 2>/dev/null | tr -d ' ')
    fi

    # remote size + free
    mapfile -t REM < <(check_ssh_remote_size)
    rsize="${REM[0]:-?}"
    rfree="${REM[1]:-?}"

    # local log에서 rsync 최신 progress line 뽑기 (예: "10.2G 28% 142.6MB/s 0:03:01")
    last_prog=$(grep -oE '[0-9][0-9,]*[KMGT]? +[0-9]+% +[0-9.]+[KMGT]?B/s +[0-9:]+' "$TRANSFER_LOG" 2>/dev/null | tail -1)

    # ALL DONE 감지
    done_line=$(grep -c "^\[.*\] ALL DONE" "$TRANSFER_LOG" 2>/dev/null || echo 0)

    echo "=== $ts ==="
    echo "  transfer alive=$alive pid=$tpid ${etime:+etime=$etime}"
    echo "  remote_dest_size=$rsize / target_gb=${TOTAL_GB_TARGET}G  remote_fs_free=$rfree"
    [ -n "$last_prog" ] && echo "  last_progress: $last_prog"
    echo "  local_log_tail:"
    tail -3 "$TRANSFER_LOG" 2>/dev/null | sed 's/^/    /'

    # 완료 + transfer dead면 종료 (FOREVER=1이면 계속)
    if [ "$done_line" -gt 0 ] && [ "$alive" = "no" ] && [ "$FOREVER" != "1" ]; then
        echo "[$ts] ALL DONE detected + transfer dead → monitor exiting"
        break
    fi

    sleep "$INTERVAL"
done
