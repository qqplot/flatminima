#!/bin/bash
# 5 개 METHOD 를 순차 실행 (한 run 끝나야 다음 시작).
# 각 run 은 run_experiment.sh 에 정의된 chunked training (100 step + 15 min sleep) 방식.
# 모든 run 은 현재 활성화된 ROCm/PyTorch 환경에서 실행됨.
#
# 사용법:
#   bash run_all_experiments.sh
#
# 중단/재개:
#   - 중간에 Ctrl+C 로 끊어도, 각 method 의 체크포인트 기반 resume 이 동작
#   - 다시 실행하면 완료된 method 는 빠르게 재진입 후 "all epochs complete" 로 종료하고
#     미완료 method 부터 이어짐 (chunk 루프가 "no progress" 로 넘어가서 다음 method 로 진행)
#
# 환경변수 override 가능: MODEL_NAME, TOTAL_EPOCHS 등.

set -eo pipefail

export MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-1.5B-Instruct"}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
export CHUNK_STEPS=${CHUNK_STEPS:-100}
export SLEEP_SECONDS=${SLEEP_SECONDS:-600}
# 훈련 끝난 method 의 중간 step 체크포인트를 자동 삭제 (epoch 경계만 유지)
export AUTO_PRUNE=${AUTO_PRUNE:-true}

LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/run_all-$(date +%Y%m%d-%H%M%S).log"
echo "[run_all] master log -> $MASTER_LOG"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "[run_all] MODEL_NAME=$MODEL_NAME  TOTAL_EPOCHS=$TOTAL_EPOCHS  WEIGHT_DECAY=$WEIGHT_DECAY"
echo "[run_all] CHUNK_STEPS=$CHUNK_STEPS  SLEEP_SECONDS=$SLEEP_SECONDS"
date

# 실험 순서: SFT 계열 먼저, 그 다음 DFT 계열.
experiments=(
    "METHOD=sft"
    "METHOD=sft+adazo ZO_SIGMA=1e-3 SAM_RHO_MAX=20 SAM_RHO_MIN=2"
    "METHOD=sft+zo"
    "METHOD=dft"
    "METHOD=dft+adazo ZO_SIGMA=5e-3 SAM_RHO_MAX=10 SAM_RHO_MIN=1"
    "METHOD=dft+zo"
)

for args in "${experiments[@]}"; do
    echo ""
    echo "######################################################################"
    echo "[run_all] >>> $args  @  $(date)"
    echo "######################################################################"
    env $args bash run_experiment.sh
    echo "[run_all] <<< $args complete  @  $(date)"
done

echo ""
echo "[run_all] all experiments finished"
date
