#!/bin/bash
# backup_method.sh — method 완료 후 ckpt 디렉터리 전체를 백업.
# 백업 대상: BACKUP_MODE 에 따라 분기.
#   local  : /data2 NVMe 로 cross-disk rsync (빠르고 안정).
#   remote : 원격 SSH 호스트로 rsync (네트워크 의존).
#   both   : local 먼저 (must succeed) → remote (best-effort, 실패해도 method 진행).
# 무결성 검증(local 합계 size ≈ backup 합계 size, ±2%) 통과 시 PURGE_AFTER_BACKUP 가
# true 면 로컬 ckpt 삭제. NVMe 마이그레이션 후 기본값은 false (디스크 여유 충분).
#
# 사용:
#   bash backup_method.sh <save_path>
#
# 환경변수:
#   BACKUP_MODE          (기본: both)         — local | remote | both
#   BACKUP_LOCAL_BASE    (기본: /data2/flatminima/backup)
#   REMOTE_HOST          (기본: kyubyungchae@147.47.200.22)
#   REMOTE_BASE_PATH     (기본: /shared/s2/lab01/qqplot/flatminima_ckpt)
#   BACKUP_LOG           (기본: /data/flatminima/verl/logs/backup.log)
#   PURGE_AFTER_BACKUP   (기본: false)        — true 면 백업 검증 후 로컬 삭제
#   REMOTE_PENDING_DIR   (기본: /data/flatminima/verl/logs/remote_pending) —
#                        remote 실패 시 method name 을 마커로 기록 → retry 데몬이 처리
#
# Exit codes:
#   0 = primary 백업 성공 (remote 실패는 warning)
#   2 = primary rsync 실패
#   3 = primary 무결성 검증 실패
#   4 = 입력 인자 잘못

set -u

SAVE_PATH="${1:-}"
if [ -z "$SAVE_PATH" ] || [ ! -d "$SAVE_PATH" ]; then
    echo "[backup_method] usage: $0 <save_path>" >&2
    echo "[backup_method] missing or invalid save_path: $SAVE_PATH" >&2
    exit 4
fi

BACKUP_MODE="${BACKUP_MODE:-both}"
BACKUP_LOCAL_BASE="${BACKUP_LOCAL_BASE:-/data2/flatminima/backup}"
REMOTE_HOST="${REMOTE_HOST:-kyubyungchae@147.47.200.22}"
REMOTE_BASE_PATH="${REMOTE_BASE_PATH:-/shared/s2/lab01/qqplot/flatminima_ckpt}"
BACKUP_LOG="${BACKUP_LOG:-/data/flatminima/verl/logs/backup.log}"
PURGE_AFTER_BACKUP="${PURGE_AFTER_BACKUP:-false}"
REMOTE_PENDING_DIR="${REMOTE_PENDING_DIR:-/data/flatminima/verl/logs/remote_pending}"

mkdir -p "$(dirname "$BACKUP_LOG")" "$REMOTE_PENDING_DIR"
exec >> "$BACKUP_LOG" 2>&1

NAME=$(basename "$SAVE_PATH")
TS=$(date -Iseconds)

echo ""
echo "================================================================"
echo "[$TS] backup_method START  name=$NAME mode=$BACKUP_MODE"
echo "[$TS]   src  = $SAVE_PATH"

LOCAL_BYTES=$(du -sb "$SAVE_PATH" 2>/dev/null | awk '{print $1}')
LOCAL_HUMAN=$(du -sh "$SAVE_PATH" 2>/dev/null | awk '{print $1}')
echo "[$TS]   local size: $LOCAL_HUMAN ($LOCAL_BYTES B)"

PRIMARY_OK=0

# ─────────────────────────── PRIMARY: LOCAL (cross-disk) ───────────────────────────
if [ "$BACKUP_MODE" = "local" ] || [ "$BACKUP_MODE" = "both" ]; then
    LOCAL_DEST="$BACKUP_LOCAL_BASE/$NAME"
    mkdir -p "$LOCAL_DEST"
    echo "[$(date -Iseconds)] LOCAL rsync  $SAVE_PATH/ -> $LOCAL_DEST/"
    rsync -ah --partial --info=stats2 "$SAVE_PATH/" "$LOCAL_DEST/"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date -Iseconds)] LOCAL rsync FAILED rc=$rc" >&2
        [ "$BACKUP_MODE" = "local" ] && exit 2
    else
        BK_BYTES=$(du -sb "$LOCAL_DEST" | awk '{print $1}')
        DIFF=$(( BK_BYTES - LOCAL_BYTES )); ABSDIFF=${DIFF#-}
        TOL=$(( LOCAL_BYTES / 50 ))
        if [ "$ABSDIFF" -gt "$TOL" ]; then
            echo "[$(date -Iseconds)] LOCAL integrity FAILED (src=$LOCAL_BYTES dst=$BK_BYTES diff=$DIFF tol=$TOL)" >&2
            [ "$BACKUP_MODE" = "local" ] && exit 3
        else
            echo "[$(date -Iseconds)] LOCAL OK (diff=$DIFF B, tol=$TOL B)"
            PRIMARY_OK=1
        fi
    fi
fi

# ─────────────────────────── SECONDARY: REMOTE (SSH) ───────────────────────────
REMOTE_OK=0
if [ "$BACKUP_MODE" = "remote" ] || [ "$BACKUP_MODE" = "both" ]; then
    REMOTE_PATH="$REMOTE_BASE_PATH/$NAME"
    echo "[$(date -Iseconds)] REMOTE attempt  $REMOTE_HOST:$REMOTE_PATH"
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_HOST" "mkdir -p $REMOTE_PATH" 2>&1
    if [ $? -ne 0 ]; then
        echo "[$(date -Iseconds)] REMOTE FAILED: cannot create remote dir (network/auth issue)"
        # remote 실패 마커 기록 → retry 데몬이 catch
        echo "$SAVE_PATH" > "$REMOTE_PENDING_DIR/$NAME"
    else
        rsync -ah --partial --append-verify --info=stats2 \
              -e "ssh -o BatchMode=yes -o ConnectTimeout=15" \
              "$SAVE_PATH/" "$REMOTE_HOST:$REMOTE_PATH/"
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "[$(date -Iseconds)] REMOTE rsync FAILED rc=$rc"
            echo "$SAVE_PATH" > "$REMOTE_PENDING_DIR/$NAME"
            [ "$BACKUP_MODE" = "remote" ] && exit 2
        else
            REMOTE_BYTES=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_HOST" \
                "du -sb $REMOTE_PATH 2>/dev/null | awk '{print \$1}'")
            if [ -n "$REMOTE_BYTES" ] && [ "$REMOTE_BYTES" != "0" ]; then
                DIFF=$(( REMOTE_BYTES - LOCAL_BYTES )); ABSDIFF=${DIFF#-}
                TOL=$(( LOCAL_BYTES / 50 ))
                if [ "$ABSDIFF" -gt "$TOL" ]; then
                    echo "[$(date -Iseconds)] REMOTE integrity FAILED (src=$LOCAL_BYTES dst=$REMOTE_BYTES diff=$DIFF tol=$TOL)"
                    echo "$SAVE_PATH" > "$REMOTE_PENDING_DIR/$NAME"
                    [ "$BACKUP_MODE" = "remote" ] && exit 3
                else
                    echo "[$(date -Iseconds)] REMOTE OK (diff=$DIFF B, tol=$TOL B)"
                    rm -f "$REMOTE_PENDING_DIR/$NAME"
                    REMOTE_OK=1
                    [ "$BACKUP_MODE" = "remote" ] && PRIMARY_OK=1
                fi
            else
                echo "[$(date -Iseconds)] REMOTE FAILED: cannot read remote size"
                echo "$SAVE_PATH" > "$REMOTE_PENDING_DIR/$NAME"
            fi
        fi
    fi
fi

# ─────────────────────────── PURGE (옵션) ───────────────────────────
if [ "$PURGE_AFTER_BACKUP" = "true" ] && [ "$PRIMARY_OK" = "1" ]; then
    echo "[$(date -Iseconds)] purging local: rm -rf $SAVE_PATH"
    rm -rf "$SAVE_PATH"
    echo "[$(date -Iseconds)] local purge done. freed $LOCAL_HUMAN"
else
    echo "[$(date -Iseconds)] PURGE_AFTER_BACKUP=$PURGE_AFTER_BACKUP primary_ok=$PRIMARY_OK — local 유지"
fi

echo "[$(date -Iseconds)] backup_method DONE  name=$NAME local_ok=$PRIMARY_OK remote_ok=$REMOTE_OK"
echo "================================================================"

exit 0
