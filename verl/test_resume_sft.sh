#!/bin/bash
# test_resume_sft.sh — SFT resume 동작 검증용
#   Phase 1: 3 epoch 설정으로 학습하되 100 step에서 조기 종료 (total_training_steps=100)
#   → 15분 sleep →
#   Phase 2: resume_path=auto 로 가장 최근 checkpoint 로드 후 3 epoch 끝까지 진행
#
# 사용법:
#   bash test_resume_sft.sh
#   NPROC_PER_NODE=2 bash test_resume_sft.sh   # GPU 개수 override
#   SLEEP_SECONDS=60 bash test_resume_sft.sh   # 빠른 테스트용 sleep 단축

set -eo pipefail

# ---------- 환경변수 (train_sft.sh 그대로) ----------
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export WANDB_MODE=${WANDB_MODE:-online}

PYBIN=${PYBIN:-$(command -v python)}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/sft-resume-test-$(date +%Y%m%d-%H%M%S).log}"
echo "[test_resume] logging to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

$PYBIN -c "import torch; print('[test_resume] torch', torch.__version__, 'hip', torch.version.hip, 'devs', torch.cuda.device_count())"

# ---------- 하이퍼파라미터 ----------
nproc_per_node=${NPROC_PER_NODE:-8}
project_name=numina-cot
model_name="Qwen/Qwen2.5-Math-1.5B"
experiment_name="numina-cot-sft-qwen-2.5-math-1.5b"
save_path=checkpoints/$experiment_name
lr=5e-5
phase1_steps=${PHASE1_STEPS:-100}
sleep_seconds=${SLEEP_SECONDS:-900}   # 15 min

# 두 phase 공통 인자 (experiment_name의 timestamp는 한 번만 평가되어 phase 1/2가 같은 run id 공유)
common_args=(
    data.train_files=data/numina_cot/train.parquet
    data.val_files=data/math500/test.parquet
    data.prompt_key=extra_info
    data.response_key=extra_info
    data.train_batch_size=256
    data.max_length=4096
    optim.lr=$lr
    "data.prompt_dict_keys=['question']"
    "data.response_dict_keys=['answer']"
    data.micro_batch_size_per_gpu=16
    model.partial_pretrain=$model_name
    model.use_liger=True
    model.fsdp_config.model_dtype=bf16
    trainer.default_local_dir=$save_path
    trainer.project_name=$project_name
    trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)"
    "trainer.logger=['console','wandb']"
    trainer.default_hdfs_dir=null
    trainer.test_freq=10
    trainer.save_freq=50
    trainer.total_epochs=3
    trainer.n_gpus_per_node=$nproc_per_node
    ulysses_sequence_parallel_size=1
    use_remove_padding=true
)

run_trainer() {
    $PYBIN -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m verl.trainer.fsdp_sft_trainer "$@"
}

# ---------- Phase 1 ----------
echo "======================================================================"
echo "[test_resume] Phase 1 시작: ${phase1_steps} step 까지 학습 후 종료"
echo "======================================================================"
date

run_trainer \
    "${common_args[@]}" \
    trainer.total_training_steps=$phase1_steps

echo "[test_resume] Phase 1 종료"
date

# checkpoint 확인
latest_ckpt=$(ls -d "$save_path"/global_step_* 2>/dev/null | sort -V | tail -n 1 || true)
if [ -z "$latest_ckpt" ]; then
    echo "[test_resume] ERROR: $save_path 에 global_step_* 가 없음 — resume 불가"
    exit 1
fi
echo "[test_resume] latest checkpoint: $latest_ckpt"
if [ ! -f "$latest_ckpt/trainer_state.pt" ]; then
    echo "[test_resume] WARN: $latest_ckpt/trainer_state.pt 없음 — 가중치만 로드됨(step=0부터 재시작)"
fi

# ---------- Sleep ----------
echo "======================================================================"
echo "[test_resume] ${sleep_seconds} 초 sleep …"
echo "======================================================================"
sleep $sleep_seconds

# ---------- Phase 2 ----------
echo "======================================================================"
echo "[test_resume] Phase 2 시작: resume_path=auto 로 이어학습 (총 3 epoch 완료까지)"
echo "======================================================================"
date

run_trainer \
    "${common_args[@]}" \
    trainer.resume_path=auto

echo "[test_resume] Phase 2 종료"
date
echo "[test_resume] 완료"
