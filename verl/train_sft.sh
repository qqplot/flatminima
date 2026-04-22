#!/bin/bash

# 1. 기본 자원 설정
#SBATCH --job-name=sft       # 작업 이름
#SBATCH --partition=laal_rtx6000         # 요청하신 파티션
#SBATCH --nodes=1                        # 사용할 노드 수
#SBATCH --ntasks-per-node=1              # 노드당 실행할 작업 수 (보통 1개로 잡고 torchrun 사용)
#SBATCH --gres=gpu:2                     # GPU 2개 요청
#SBATCH --cpus-per-task=16               # CPU 코어 수 (GPU당 8개 권장, 총 16개)
#SBATCH --mem=256G                       # 메모리 (시스템이 644G이므로 128~256G도 여유롭습니다)
#SBATCH --time=24:00:00                  # 최대 실행 시간 (1일)
#SBATCH --output=logs/numina-cot-sft-%j.out             # 표준 출력 로그 (logs 폴더 미리 생성 필요)
#SBATCH --error=logs/numina-cot-sft-%j.err              # 에러 로그

# 2. 환경 변수 설정 (ROCm / RCCL)
export NCCL_P2P_DISABLE=0                # RCCL은 NCCL_* 변수를 그대로 따름
export NCCL_IB_DISABLE=1                 # 단일 노드
# wandb: API 키 없을 때 학습이 죽지 않도록 기본을 offline으로. 로그인 돼 있으면 online으로 override 가능
export WANDB_MODE=${WANDB_MODE:-online}

# 시스템 rocm-torch Python 사용 (conda 불필요). 필요 시 PYBIN을 덮어쓰면 됨.
PYBIN=${PYBIN:-$(command -v python)}

# 로그 파일 경로 (stdout/stderr 모두 티) — 실시간 tail 가능
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/sft-$(date +%Y%m%d-%H%M%S).log}"
echo "[train_sft] logging to $LOG_FILE"

# 이 라인 이후 모든 stdout/stderr가 터미널과 로그 파일 양쪽으로 동시에 흐름
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[train_sft] python = $PYBIN"
$PYBIN -c "import torch; print('[train_sft] torch', torch.__version__, 'hip', torch.version.hip, 'devs', torch.cuda.device_count())"

nproc_per_node=${NPROC_PER_NODE:-8}
project_name=numina-cot

model_name="Qwen/Qwen2.5-Math-1.5B"
experiment_name="numina-cot-sft-qwen-2.5-math-1.5b"

save_path=checkpoints/$experiment_name
lr=5e-5

# Chunked training 설정: 100 step 마다 종료하고 15분 sleep 후 resume 반복.
chunk_steps=${CHUNK_STEPS:-200}          # 한 번에 학습할 step 수
sleep_seconds=${SLEEP_SECONDS:-900}      # 15분
run_timestamp=$(date +%Y%m%d-%H%M%S)     # phase 전체가 같은 experiment_name 공유

# 모든 chunk 가 같은 wandb run 에 기록되도록 run_id 고정 (점/공백 등은 하이픈으로 치환).
# 규칙:
#   - checkpoint 가 하나라도 있으면 기존 wandb_run_id.txt 를 재사용 (resume)
#   - checkpoint 가 없는데 id 파일만 있으면 이전 run 이 crash/abort 한 상태 → 새 id 재발급
#   - 아예 처음이면 새 id 생성
# WANDB_RESUME=allow: 동일 ID run 있으면 이어쓰기, 없으면 새로 생성.
wandb_id_file="$save_path/wandb_run_id.txt"
mkdir -p "$save_path"
has_ckpt=$(ls -d "$save_path"/global_step_* 2>/dev/null | head -n 1)
if [ -s "$wandb_id_file" ] && [ -n "$has_ckpt" ]; then
    export WANDB_RUN_ID="$(cat "$wandb_id_file")"
    echo "[train_sft] reusing existing wandb run id from $wandb_id_file (checkpoints present)"
else
    if [ -s "$wandb_id_file" ] && [ -z "$has_ckpt" ]; then
        echo "[train_sft] stale wandb_run_id.txt with no checkpoints → regenerating new run id"
    fi
    sanitized_name="${experiment_name//[. ]/-}"
    export WANDB_RUN_ID="${sanitized_name}-${run_timestamp}"
    echo "$WANDB_RUN_ID" > "$wandb_id_file"
    echo "[train_sft] created new wandb run id, saved to $wandb_id_file"
fi
export WANDB_RESUME=allow
echo "[train_sft] WANDB_RUN_ID=$WANDB_RUN_ID (resume=$WANDB_RESUME)"

# 두 phase 공통 인자 (단일 source of truth)
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
    "trainer.experiment_name=$experiment_name-$run_timestamp"
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

get_latest_step() {
    local latest
    latest=$(ls -d "$save_path"/global_step_* 2>/dev/null | sort -V | tail -n 1)
    if [ -n "$latest" ]; then
        basename "$latest" | sed 's/global_step_//'
    else
        echo 0
    fi
}

# Chunked training loop
iteration=0
while :; do
    iteration=$((iteration + 1))
    current_step=$(get_latest_step)
    target_step=$((current_step + chunk_steps))

    echo "======================================================================"
    echo "[train_sft] iteration $iteration: step $current_step → $target_step"
    echo "======================================================================"
    date

    if [ "$current_step" = "0" ]; then
        # 첫 chunk: 새로 시작
        run_trainer "${common_args[@]}" trainer.total_training_steps=$target_step
    else
        # 이후 chunk: 가장 최근 checkpoint 에서 resume
        run_trainer "${common_args[@]}" \
            trainer.total_training_steps=$target_step \
            trainer.resume_path=auto
    fi

    new_step=$(get_latest_step)
    echo "[train_sft] iteration $iteration: ended at step $new_step (target $target_step)"

    # 진척 없으면(전체 epoch 소진) 종료
    if [ "$new_step" -le "$current_step" ]; then
        echo "[train_sft] no progress — all epochs complete, exiting loop"
        break
    fi

    # 목표 step 도달 못했으면(학습이 비정상 종료됨) 종료
    if [ "$new_step" -lt "$target_step" ]; then
        echo "[train_sft] reached step $new_step < target $target_step — likely epochs exhausted, exiting loop"
        break
    fi

    echo "[train_sft] sleeping ${sleep_seconds}s before next chunk..."
    sleep $sleep_seconds
done

echo "[train_sft] done — final step $(get_latest_step)"
date
