#!/bin/bash
# Workflow:
#   Phase 1: wait for live global_step_780 (epoch 2 경계) → mv to preserved/
#   Phase 2: wait for iter 4 compute end + sleep entry (no FSDP + run_experiment alive)
#   Phase 3: SIGTERM run_all + run_experiment → relaunch run_all_experiments.sh
# 목적: 현재 sft 런이 옛 rolling_prune 을 메모리에 들고 있어서 step 780 을 지우는 걸 막고,
# sleep 구간에 깨끗하게 재시작해서 이후 chunk 들은 새 rolling_prune(epoch 자동 보호)을 사용하게 함.
set -u
LOG=/data/flatminima/verl/logs/preserve_then_relaunch.log
CKPT=/data/flatminima/verl/checkpoints/numina-cot-sft-qwen-qwen2-5-1-5b-instruct
RUN_ALL_PID_FILE=/data/flatminima/verl/logs/run_all_experiments.pid
OUT=/data/flatminima/verl/logs/run_all_experiments.out
exec >> "$LOG" 2>&1

echo "[$(date -Iseconds)] preserve_then_relaunch started pid=$$"

# Phase 1: wait for step 780, preserve it
echo "[$(date -Iseconds)] PHASE 1: watching for $CKPT/global_step_780"
until [ -d "$CKPT/global_step_780" ]; do sleep 30; done
echo "[$(date -Iseconds)] step_780 appeared"
mkdir -p "$CKPT/preserved"
if [ -e "$CKPT/preserved/global_step_780" ]; then
    echo "[$(date -Iseconds)] preserved/global_step_780 already exists → removing live copy"
    rm -rf "$CKPT/global_step_780"
else
    mv "$CKPT/global_step_780" "$CKPT/preserved/"
    echo "[$(date -Iseconds)] mv global_step_780 → preserved/"
fi

# Phase 2: wait for iter 4 compute end (no FSDP python workers) + rolling_prune done + sleep begin
echo "[$(date -Iseconds)] PHASE 2: waiting for iter 4 trainer exit (no fsdp_sft_trainer procs)"
while pgrep -f "verl.trainer.fsdp_sft_trainer" > /dev/null 2>&1; do
    sleep 15
done
# trainer 전부 exit. rolling_prune(몇 초)이 끝나고 sleep 진입할 때까지 약간 대기.
sleep 10
if ! pgrep -f "run_experiment.sh" > /dev/null 2>&1; then
    echo "[$(date -Iseconds)] run_experiment already gone → skip relaunch (unexpected)"
    exit 0
fi
RUN_ALL_PID=$(cat "$RUN_ALL_PID_FILE" 2>/dev/null | tr -dc '0-9')
if [ -z "$RUN_ALL_PID" ] || ! kill -0 "$RUN_ALL_PID" 2>/dev/null; then
    echo "[$(date -Iseconds)] run_all pid file invalid or process gone → skip relaunch"
    exit 0
fi
echo "[$(date -Iseconds)] iter 4 done, run_all pid=$RUN_ALL_PID still alive in sleep → proceed"

# Phase 3: kill existing + relaunch
echo "[$(date -Iseconds)] PHASE 3: terminating run_all and run_experiment"
pkill -TERM -f "run_all_experiments.sh" 2>/dev/null || true
pkill -TERM -f "run_experiment.sh" 2>/dev/null || true
sleep 2
# 혹시 남으면 강제
pkill -KILL -f "run_all_experiments.sh" 2>/dev/null || true
pkill -KILL -f "run_experiment.sh" 2>/dev/null || true
until ! pgrep -f "run_all_experiments.sh|run_experiment.sh" > /dev/null 2>&1; do sleep 1; done
echo "[$(date -Iseconds)] processes terminated"

# 이전 out 파일 아카이브
if [ -f "$OUT" ]; then
    mv "$OUT" "${OUT%.out}-$(date +%Y%m%d-%H%M%S).out"
fi

# 관통 재실행 (WANDB offline + 기존 ENV 그대로)
cd /data/flatminima/verl
WANDB_MODE=offline nohup bash run_all_experiments.sh > "$OUT" 2>&1 &
NEW_PID=$!
disown
echo "$NEW_PID" > "$RUN_ALL_PID_FILE"
echo "[$(date -Iseconds)] relaunched pid=$NEW_PID  (new rolling_prune with epoch protection active)"
