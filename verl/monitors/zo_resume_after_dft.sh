#!/bin/bash
# zo_resume_after_dft.sh — dft 완료 후 fsdp_zo_trainer_dft.py holistic fix 자동 검증 데몬.
#
# 단계:
#   1) run_all_experiments 가 죽고 dft ckpt 가 step 1170 도달할 때까지 polling 대기.
#   2) METHOD=sft+zo CHUNK_STEPS=20 으로 fail-fast 검증 (별도 EXPERIMENT_NAME 으로 격리).
#   3) Error/Traceback 없고 step:10+ 도달 시 PASS.
#   4) PASS: run_all_experiments.sh experiments 배열에서 sft+zo, dft+zo 주석 해제,
#           검증 ckpt 정리, run_all 재시작.
#   5) FAIL: 마커 파일 기록 후 exit (사람이 fix 검토 필요).
#
# env:
#   POLL_INTERVAL    기본 60s
#   RUN_ALL_PID_FILE 기본 /data/flatminima/verl/logs/run_all_experiments.pid
#   DFT_CKPT_DIR     기본 /data1/flatminima/verl/checkpoints/numina-cot-dft-qwen-qwen2-5-1-5b-instruct
#   LOG              기본 /data/flatminima/verl/logs/zo_resume_after_dft.log

set -u
LOG="${LOG:-/data/flatminima/verl/logs/zo_resume_after_dft.log}"
RUN_ALL_PID_FILE="${RUN_ALL_PID_FILE:-/data/flatminima/verl/logs/run_all_experiments.pid}"
DFT_CKPT_DIR="${DFT_CKPT_DIR:-/data1/flatminima/verl/checkpoints/numina-cot-dft-qwen-qwen2-5-1-5b-instruct}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
VERL_DIR="/data/flatminima/verl"
LOG_DIR="$VERL_DIR/logs"
RUN_ALL_SH="$VERL_DIR/run_all_experiments.sh"
FAIL_MARKER="$LOG_DIR/zo_resume.validate.failed"

mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

echo ""
echo "[$(date -Iseconds)] zo_resume_after_dft started pid=$$ poll=${POLL_INTERVAL}s"
rm -f "$FAIL_MARKER"

# ─────────── 1) dft 완료 대기 ───────────
while :; do
    pid=$(cat "$RUN_ALL_PID_FILE" 2>/dev/null | tr -dc '0-9')
    alive="no"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive="yes"

    has_1170="no"
    if [ -d "$DFT_CKPT_DIR/global_step_1170" ] || [ -d "$DFT_CKPT_DIR/preserved/global_step_1170" ]; then
        has_1170="yes"
    fi

    if [ "$alive" = "no" ] && [ "$has_1170" = "yes" ]; then
        echo "[$(date -Iseconds)] dft completed (run_all dead + step_1170 present)"
        break
    fi
    if [ "$alive" = "no" ] && [ "$has_1170" = "no" ]; then
        echo "[$(date -Iseconds)] ⚠️ run_all dead but step_1170 missing — dft may have failed."
        echo "[$(date -Iseconds)]    inspect $LOG_DIR/run_all_experiments.out — aborting validation."
        date -Iseconds > "$FAIL_MARKER"
        exit 2
    fi
    sleep "$POLL_INTERVAL"
done

# ─────────── 2) sft+zo 검증 (CHUNK_STEPS=20) ───────────
TS=$(date +%Y%m%d-%H%M%S)
VAL_NAME="validate-zo-fix-${TS}"
VAL_LOG="$LOG_DIR/$VAL_NAME.log"
VAL_CKPT="/data1/flatminima/verl/checkpoints/$VAL_NAME"
echo "[$(date -Iseconds)] running validation: METHOD=sft+zo CHUNK_STEPS=20 NAME=$VAL_NAME"

cd "$VERL_DIR" || exit 3
WANDB_MODE=offline \
  CKPT_BASE_DIR="/data1/flatminima/verl" \
  CHUNK_STEPS=20 \
  TOTAL_EPOCHS=1 \
  SAVE_FREQ=20 \
  TEST_FREQ=20 \
  METHOD=sft+zo \
  MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct" \
  EXPERIMENT_NAME="$VAL_NAME" \
  AUTO_PRUNE=false \
  BACKUP_AND_PURGE=false \
  KEEP_RECENT_CKPTS=0 \
  bash run_experiment.sh > "$VAL_LOG" 2>&1
RC=$?

ERR_COUNT=$(grep -cE "KeyError|AttributeError|RuntimeError|Traceback" "$VAL_LOG" 2>/dev/null)
LAST_STEP=$(grep -oE "step:[0-9]+ - train/loss" "$VAL_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1)
LAST_STEP="${LAST_STEP:-0}"
echo "[$(date -Iseconds)] validation done rc=$RC errors=$ERR_COUNT last_step=$LAST_STEP"

# ─────────── 3) 결과 평가 ───────────
if [ "$RC" != "0" ] || [ "$ERR_COUNT" -gt "0" ] || [ "$LAST_STEP" -lt "10" ]; then
    echo "[$(date -Iseconds)] ❌ VALIDATION FAIL — sft+zo, dft+zo 재개하지 않음"
    echo "$(date -Iseconds) rc=$RC errors=$ERR_COUNT last_step=$LAST_STEP log=$VAL_LOG" > "$FAIL_MARKER"
    exit 4
fi
echo "[$(date -Iseconds)] ✅ VALIDATION PASS"

# ─────────── 4) experiments 배열 주석 해제 ───────────
echo "[$(date -Iseconds)] re-enabling sft+zo and dft+zo in $RUN_ALL_SH"
sed -i 's|^    # "METHOD=sft+zo".*|    "METHOD=sft+zo"           # holistic fix (2026-04-27) 검증 통과 후 재개|' "$RUN_ALL_SH"
sed -i 's|^    # "METHOD=dft+zo".*|    "METHOD=dft+zo"           # holistic fix (2026-04-27) 검증 통과 후 재개|' "$RUN_ALL_SH"
echo "  현재 experiments:"
grep -nE '^\s*"METHOD=' "$RUN_ALL_SH"

# ─────────── 5) 검증 ckpt 정리 + run_all 재시작 ───────────
[ -d "$VAL_CKPT" ] && { echo "[$(date -Iseconds)] removing validation ckpt $VAL_CKPT"; rm -rf "$VAL_CKPT"; }

NEW_TS=$(date +%Y%m%d-%H%M%S)
cd "$VERL_DIR"
nohup bash run_all_experiments.sh > "$LOG_DIR/run_all_experiments.out" 2>&1 &
NEW_PID=$!
echo $NEW_PID > "$LOG_DIR/run_all_experiments.pid"
echo "[$(date -Iseconds)] run_all restarted pid=$NEW_PID  ts=$NEW_TS"
echo "[$(date -Iseconds)]   dft 는 이미 step 1170 ckpt 있어 즉시 종료, sft+zo 부터 진행 예상"
echo "[$(date -Iseconds)] zo_resume_after_dft DONE"
