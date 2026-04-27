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
# train_sft.sh 가 검증한 값 그대로: chunk 200 step + 15분 sleep
# (chunk 100 step 으로는 NCCL re-init 횟수 12회로 많음 → 적은 횟수 6회로 줄여 transient
# unhandled cuda error 노출 빈도 감소. SFT 메서드는 이 설정으로 1170 step 완주 검증됨.)
export CHUNK_STEPS=${CHUNK_STEPS:-200}
export SLEEP_SECONDS=${SLEEP_SECONDS:-600}
# 훈련 끝난 method 의 중간 step 체크포인트를 자동 삭제 (epoch 경계만 유지)
export AUTO_PRUNE=${AUTO_PRUNE:-true}
# ── 저장 경로 (NVMe 마이그레이션) ──────────────────────────────────────────
# CKPT_BASE_DIR 하위에 checkpoints/ 가 생성됨. 미지정 시 cwd 기반(레거시).
export CKPT_BASE_DIR=${CKPT_BASE_DIR:-/data1/flatminima/verl}
mkdir -p "$CKPT_BASE_DIR/checkpoints"
# ── 백업 (로컬 cross-disk 모드) ────────────────────────────────────────────
# remote SSH (147.47.200.22) 04-25 실패 사례 → /data2 NVMe 로컬 백업으로 전환.
# REMOTE_HOST/REMOTE_BASE_PATH 는 BACKUP_MODE=remote 일 때만 사용.
export BACKUP_AND_PURGE=${BACKUP_AND_PURGE:-false}   # NVMe 6.6T 여유: purge 불필요
export BACKUP_MODE=${BACKUP_MODE:-local}
export BACKUP_LOCAL_BASE=${BACKUP_LOCAL_BASE:-/data2/flatminima/backup}
mkdir -p "$BACKUP_LOCAL_BASE"
export REMOTE_HOST=${REMOTE_HOST:-kyubyungchae@147.47.200.22}
export REMOTE_BASE_PATH=${REMOTE_BASE_PATH:-/shared/s2/lab01/qqplot/flatminima_ckpt}

# WANDB offline (이 컨테이너에 API key 없음) — 미설정 시 trainer가 즉사하므로 명시적으로 강제
export WANDB_MODE=${WANDB_MODE:-offline}

# NCCL/Torch 안전장치
#   ASYNC_ERROR_HANDLING — 에러 시 graceful shutdown
#   NCCL_DEBUG=WARN      — 다음 실패 시 진단 가능
# (BLOCKING_WAIT=1 은 broadcast 실패 시 hang을 유발해서 제거함)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# FSDP1 사용 — verl 코드는 "fsdp" (1) / "fsdp2" 두 값만 인식. "fsdp1" 아님.
# FSDP2의 set_model_state_dict broadcast 에서 ROCm 'invalid device pointer' 발생 → "fsdp" 사용.
export MODEL_STRATEGY=${MODEL_STRATEGY:-fsdp}

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
    # "METHOD=sft"   # 2026-04-22 완료, 원격 백업됨, 로컬 purge → 재훈련 방지 위해 주석 처리
    # "METHOD=sft+adazo ZO_SIGMA=1e-3 SAM_RHO_MAX=20 SAM_RHO_MIN=2"  # 2026-04-25 완료 (단, fsdp_adazo_trainer.py 가 sam_rho_max/min 미적용 버그 — sam_rho 기본값 0.0005 사용)
    # "METHOD=sft+zo"   # fsdp_zo_trainer_dft.py 키-공간 불일치 버그 (state_dict vs named_parameters) — holistic fix 후 재개
    "METHOD=dft"              # fsdp_dft_trainer.py 사용 (ZO/SAM 없음, 안전)
    # "METHOD=dft+adazo ZO_SIGMA=5e-3 SAM_RHO_MAX=10 SAM_RHO_MIN=1"  # 사용자 요청으로 일단 보류 (adazo trainer 버그 검증 후 재개 결정)
    # "METHOD=dft+zo"   # 같은 fsdp_zo_trainer_dft.py 사용 — sft+zo와 함께 fix 후 재개
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
