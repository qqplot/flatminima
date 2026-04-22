#!/bin/bash
# Wait for sft method to complete (3 full epochs), then terminate run_all_experiments.sh
# so that the next method (sft+adazo) does NOT auto-start.
#
# Detection: master run_all log contains "[run_all] <<< METHOD=sft complete"
#            (run_all_experiments.sh prints this immediately after run_experiment.sh for sft exits,
#             and just before the next iteration of the methods loop begins.)
#
# 로그/파일:
#   /data/flatminima/verl/logs/pause_after_sft.log : 본 스크립트 로그
#   /data/flatminima/verl/logs/pause_after_sft.triggered : pause 발동 시 생성
set -u
LOG_OUT=/data/flatminima/verl/logs/pause_after_sft.log
RUN_ALL_OUT=/data/flatminima/verl/logs/run_all_experiments.out
RUN_ALL_PID_FILE=/data/flatminima/verl/logs/run_all_experiments.pid
MARKER_FILE=/data/flatminima/verl/logs/pause_after_sft.triggered
exec >> "$LOG_OUT" 2>&1

echo "[$(date -Iseconds)] pause_after_sft started pid=$$"
echo "[$(date -Iseconds)] watching $RUN_ALL_OUT for 'METHOD=sft complete' marker"

until grep -q "^\[run_all\] <<< METHOD=sft complete" "$RUN_ALL_OUT" 2>/dev/null; do
    sleep 10
done
echo "[$(date -Iseconds)] sft completion marker detected"

RUN_ALL_PID=$(cat "$RUN_ALL_PID_FILE" 2>/dev/null | tr -dc '0-9')
if [ -n "$RUN_ALL_PID" ] && kill -0 "$RUN_ALL_PID" 2>/dev/null; then
    echo "[$(date -Iseconds)] terminating run_all pid=$RUN_ALL_PID (user requested pause before sft+adazo)"
    kill -TERM "$RUN_ALL_PID" 2>/dev/null || true
    sleep 2
    # sft+adazo 가 이미 시작했을 가능성에 대비 — 하위도 정리
    pkill -TERM -f "run_experiment.sh"        2>/dev/null || true
    pkill -TERM -f "torch.distributed.run"    2>/dev/null || true
    pkill -TERM -f "verl.trainer.fsdp_"       2>/dev/null || true
    sleep 3
    pkill -KILL -f "run_all_experiments.sh|run_experiment.sh|torch.distributed.run|verl.trainer.fsdp_" 2>/dev/null || true
    date -Iseconds > "$MARKER_FILE"
    echo "[$(date -Iseconds)] pause complete — marker file written: $MARKER_FILE"
else
    echo "[$(date -Iseconds)] run_all pid not alive → nothing to kill"
    date -Iseconds > "$MARKER_FILE"
fi

echo "[$(date -Iseconds)] pause_after_sft exit"
