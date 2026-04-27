#!/bin/bash
# remote_backup_retry.sh — backup_method.sh 가 remote 백업 실패 시 남긴
# pending 마커(/data/flatminima/verl/logs/remote_pending/<name>) 를 주기적으로
# 재처리하는 best-effort 데몬. SSH 가 일시 차단된 동안에도 method 진행을 막지 않고,
# 연결이 복구되면 자동으로 백업을 따라잡는다.
#
# 환경변수:
#   INTERVAL              (기본: 600s) — pending 디렉토리 polling 주기
#   REMOTE_HOST           (기본: kyubyungchae@147.47.200.22)
#   REMOTE_BASE_PATH      (기본: /shared/s2/lab01/qqplot/flatminima_ckpt)
#   REMOTE_PENDING_DIR    (기본: /data/flatminima/verl/logs/remote_pending)
#   LOG                   (기본: /data/flatminima/verl/logs/remote_backup_retry.log)

set -u
INTERVAL="${INTERVAL:-600}"
REMOTE_HOST="${REMOTE_HOST:-kyubyungchae@147.47.200.22}"
REMOTE_BASE_PATH="${REMOTE_BASE_PATH:-/shared/s2/lab01/qqplot/flatminima_ckpt}"
REMOTE_PENDING_DIR="${REMOTE_PENDING_DIR:-/data/flatminima/verl/logs/remote_pending}"
LOG="${LOG:-/data/flatminima/verl/logs/remote_backup_retry.log}"

mkdir -p "$REMOTE_PENDING_DIR" "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

echo ""
echo "[$(date -Iseconds)] remote_backup_retry started pid=$$ interval=${INTERVAL}s pending_dir=$REMOTE_PENDING_DIR"

ssh_ok() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" "echo PING" >/dev/null 2>&1
}

retry_one() {
    local marker="$1"
    local name save_path remote_path
    name=$(basename "$marker")
    save_path=$(cat "$marker" 2>/dev/null)
    if [ -z "$save_path" ] || [ ! -d "$save_path" ]; then
        echo "[$(date -Iseconds)]   skip $name: src missing ($save_path)"
        rm -f "$marker"
        return 0
    fi
    remote_path="$REMOTE_BASE_PATH/$name"
    echo "[$(date -Iseconds)]   retry $name  src=$save_path  dest=$REMOTE_HOST:$remote_path"

    ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_HOST" "mkdir -p $remote_path" 2>&1
    if [ $? -ne 0 ]; then
        echo "[$(date -Iseconds)]   $name FAILED mkdir — leaving marker"
        return 1
    fi

    rsync -ah --partial --append-verify --info=stats2 \
          -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
          "$save_path/" "$REMOTE_HOST:$remote_path/"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date -Iseconds)]   $name rsync FAILED rc=$rc — leaving marker"
        return 1
    fi

    local LOCAL_BYTES REMOTE_BYTES DIFF ABSDIFF TOL
    LOCAL_BYTES=$(du -sb "$save_path" 2>/dev/null | awk '{print $1}')
    REMOTE_BYTES=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_HOST" \
        "du -sb $remote_path 2>/dev/null | awk '{print \$1}'")
    if [ -z "$REMOTE_BYTES" ] || [ "$REMOTE_BYTES" = "0" ]; then
        echo "[$(date -Iseconds)]   $name verify FAILED (cannot read remote size)"
        return 1
    fi
    DIFF=$(( REMOTE_BYTES - LOCAL_BYTES )); ABSDIFF=${DIFF#-}
    TOL=$(( LOCAL_BYTES / 50 ))
    if [ "$ABSDIFF" -gt "$TOL" ]; then
        echo "[$(date -Iseconds)]   $name verify FAILED (src=$LOCAL_BYTES dst=$REMOTE_BYTES diff=$DIFF tol=$TOL)"
        return 1
    fi
    echo "[$(date -Iseconds)]   $name OK (diff=$DIFF B, tol=$TOL B) — clearing marker"
    rm -f "$marker"
    return 0
}

while :; do
    shopt -s nullglob
    markers=("$REMOTE_PENDING_DIR"/*)
    shopt -u nullglob
    if [ ${#markers[@]} -gt 0 ]; then
        echo "[$(date -Iseconds)] tick: ${#markers[@]} pending"
        if ssh_ok; then
            for m in "${markers[@]}"; do
                retry_one "$m" || true
            done
        else
            echo "[$(date -Iseconds)]   SSH still down — sleeping"
        fi
    fi
    sleep "$INTERVAL"
done
