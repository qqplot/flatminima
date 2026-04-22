#!/bin/bash
# 통합 학습 스크립트 — env 변수로 모델/METHOD/epoch/하이퍼파라미터를 스위치.
#
# 필수 env:
#   METHOD       : sft | dft | sft+adazo | dft+adazo | sft+zo | dft+zo
#   MODEL_NAME   : HF hub id, e.g. "Qwen/Qwen2.5-1.5B-Instruct"
#
# 자주 override:
#   TOTAL_EPOCHS=3   LR=5e-5   WEIGHT_DECAY=0.01
#   BATCH_SIZE=256   MAX_LENGTH=4096   MICRO_BS_PER_GPU=16
#   ZO_SIGMA  SAM_RHO_MAX  SAM_RHO_MIN   (AdaZO 계열에서만)
#
# 인프라:
#   NPROC_PER_NODE=8
#   CHUNK_STEPS=100     (100 step 마다 exit)
#   SLEEP_SECONDS=900   (15분 쿨링)
#   SAVE_FREQ=100  TEST_FREQ=50
#   KEEP_RECENT_CKPTS=2 (chunk 사이 rolling prune 으로 유지할 최신 ckpt 수, 0 이면 비활성)
#   MIN_FREE_GB=12      (chunk 시작 전 pre-flight 디스크 여유 최소값)
#   STEPS_PER_EPOCH     (rolling prune 시 epoch 경계 ckpt 보호용. 미지정 시 log 에서 자동 추출)
#   PROJECT_NAME=numina-cot
#   DATA_TRAIN=data/numina_cot/train.parquet
#   DATA_VAL=data/math500/test.parquet
#   EXPERIMENT_NAME                      (기본값은 METHOD+MODEL 로 자동 파생)
#
# 예:
#   METHOD=sft MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct" bash run_experiment.sh
#   METHOD=sft+adazo MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct" \
#       ZO_SIGMA=1e-3 SAM_RHO_MAX=20 SAM_RHO_MIN=2 bash run_experiment.sh

set -eo pipefail

# ---------- 필수 env 검증 ----------
: "${METHOD:?METHOD is required (sft | dft | sft+adazo | dft+adazo | dft+zo)}"
: "${MODEL_NAME:?MODEL_NAME is required (e.g. Qwen/Qwen2.5-1.5B-Instruct)}"

# ---------- method -> trainer module 매핑 ----------
declare -a method_extra_args=()
case "$METHOD" in
    sft)
        trainer_module="verl.trainer.fsdp_sft_trainer"
        ;;
    dft)
        trainer_module="verl.trainer.fsdp_dft_trainer"
        ;;
    sft+adazo)
        trainer_module="verl.trainer.fsdp_adazo_trainer"
        : "${ZO_SIGMA:?ZO_SIGMA required for sft+adazo}"
        : "${SAM_RHO_MAX:?SAM_RHO_MAX required for sft+adazo}"
        : "${SAM_RHO_MIN:?SAM_RHO_MIN required for sft+adazo}"
        method_extra_args+=(
            "optim.zo_sigma=$ZO_SIGMA"
            "optim.sam_rho_max=$SAM_RHO_MAX"
            "optim.sam_rho_min=$SAM_RHO_MIN"
        )
        ;;
    dft+adazo)
        trainer_module="verl.trainer.fsdp_adazo_trainer_dft"
        : "${ZO_SIGMA:?ZO_SIGMA required for dft+adazo}"
        : "${SAM_RHO_MAX:?SAM_RHO_MAX required for dft+adazo}"
        : "${SAM_RHO_MIN:?SAM_RHO_MIN required for dft+adazo}"
        method_extra_args+=(
            "optim.zo_sigma=$ZO_SIGMA"
            "optim.sam_rho_max=$SAM_RHO_MAX"
            "optim.sam_rho_min=$SAM_RHO_MIN"
            "trainer.use_dft=true"
        )
        ;;
    dft+zo)
        trainer_module="verl.trainer.fsdp_zo_trainer_dft"
        method_extra_args+=( "trainer.use_dft=true" )
        ;;
    sft+zo)
        # Pure ZO 최적화 + standard CE loss. fsdp_zo_trainer_dft 트레이너에서 use_dft=false 로
        # 설정하면 DFT 재가중치 로직이 꺼져서 결과적으로 SFT+ZO 가 됨.
        trainer_module="verl.trainer.fsdp_zo_trainer_dft"
        method_extra_args+=( "trainer.use_dft=false" )
        ;;
    *)
        echo "[run_experiment] unknown METHOD=$METHOD" >&2
        exit 1
        ;;
esac

# ---------- 하이퍼파라미터 defaults ----------
total_epochs=${TOTAL_EPOCHS:-3}
lr=${LR:-5e-5}
weight_decay=${WEIGHT_DECAY:-0.01}
batch_size=${BATCH_SIZE:-256}
max_length=${MAX_LENGTH:-4096}
micro_bs=${MICRO_BS_PER_GPU:-16}

nproc_per_node=${NPROC_PER_NODE:-8}
chunk_steps=${CHUNK_STEPS:-100}
sleep_seconds=${SLEEP_SECONDS:-900}
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-50}
# chunk 사이 rolling prune: 최신 K개만 유지 (resume 안전성 ↔ 디스크 균형).
# K<=0 이면 비활성. AUTO_PRUNE(종료 후 epoch 경계만 유지)과 독립으로 동작.
keep_recent=${KEEP_RECENT_CKPTS:-2}
# chunk 시작 전 free-space pre-flight 임계치 (GB). 이보다 작으면 실패 처리.
min_free_gb=${MIN_FREE_GB:-12}
project_name=${PROJECT_NAME:-numina-cot}
data_train=${DATA_TRAIN:-data/numina_cot/train.parquet}
data_val=${DATA_VAL:-data/math500/test.parquet}

# experiment_name auto-derive (METHOD+MODEL slug) — 사용자가 override 가능
method_slug=${METHOD//+/-}
model_slug=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr '/' '-' | tr '.' '-')
experiment_name=${EXPERIMENT_NAME:-"${project_name}-${method_slug}-${model_slug}"}
save_path="checkpoints/$experiment_name"
run_timestamp=$(date +%Y%m%d-%H%M%S)

# ---------- 환경변수 (train_sft.sh 와 동일) ----------
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export WANDB_MODE=${WANDB_MODE:-online}
PYBIN=${PYBIN:-$(command -v python)}

# ---------- logging ----------
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${experiment_name}-${run_timestamp}.log}"
echo "[run_experiment] logging to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[run_experiment] METHOD=$METHOD  model=$MODEL_NAME  epochs=$total_epochs"
echo "[run_experiment] trainer=$trainer_module  save_path=$save_path"
echo "[run_experiment] extra: ${method_extra_args[*]}"
$PYBIN -c "import torch; print('[run_experiment] torch', torch.__version__, 'hip', torch.version.hip, 'devs', torch.cuda.device_count())"

# ---------- wandb run id (train_sft.sh 의 규칙 그대로) ----------
wandb_id_file="$save_path/wandb_run_id.txt"
mkdir -p "$save_path"
has_ckpt=$(ls -d "$save_path"/global_step_* 2>/dev/null | head -n 1 || true)
if [ -s "$wandb_id_file" ] && [ -n "$has_ckpt" ]; then
    export WANDB_RUN_ID="$(cat "$wandb_id_file")"
    echo "[run_experiment] reusing wandb run id (checkpoints present)"
else
    if [ -s "$wandb_id_file" ] && [ -z "$has_ckpt" ]; then
        echo "[run_experiment] stale wandb_run_id.txt with no checkpoints → regenerating"
    fi
    sanitized_name="${experiment_name//[. ]/-}"
    export WANDB_RUN_ID="${sanitized_name}-${run_timestamp}"
    echo "$WANDB_RUN_ID" > "$wandb_id_file"
    echo "[run_experiment] created new wandb run id"
fi
export WANDB_RESUME=allow
echo "[run_experiment] WANDB_RUN_ID=$WANDB_RUN_ID (resume=$WANDB_RESUME)"

# ---------- 공통 trainer 인자 ----------
common_args=(
    "data.train_files=$data_train"
    "data.val_files=$data_val"
    data.prompt_key=extra_info
    data.response_key=extra_info
    "data.train_batch_size=$batch_size"
    "data.max_length=$max_length"
    "optim.lr=$lr"
    "optim.weight_decay=$weight_decay"
    "data.prompt_dict_keys=['question']"
    "data.response_dict_keys=['answer']"
    "data.micro_batch_size_per_gpu=$micro_bs"
    "model.partial_pretrain=$MODEL_NAME"
    model.use_liger=True
    model.fsdp_config.model_dtype=bf16
    "trainer.default_local_dir=$save_path"
    "trainer.project_name=$project_name"
    "trainer.experiment_name=${experiment_name}-${run_timestamp}"
    "trainer.logger=['console','wandb']"
    trainer.default_hdfs_dir=null
    "trainer.test_freq=$test_freq"
    "trainer.save_freq=$save_freq"
    "trainer.total_epochs=$total_epochs"
    "trainer.n_gpus_per_node=$nproc_per_node"
    ulysses_sequence_parallel_size=1
    use_remove_padding=true
)

run_trainer() {
    $PYBIN -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m "$trainer_module" "$@" "${method_extra_args[@]}"
}

get_latest_step() {
    local latest
    latest=$(ls -d "$save_path"/global_step_* 2>/dev/null | sort -V | tail -n 1 || true)
    if [ -n "$latest" ]; then
        basename "$latest" | sed 's/global_step_//'
    else
        echo 0
    fi
}

# epoch 당 step 수를 반환 (env STEPS_PER_EPOCH 우선, 없으면 exp log 의 tqdm 진행바에서 추출).
# 추출 실패 시 0 반환 → rolling_prune 은 epoch 경계 보호를 생략.
get_steps_per_epoch() {
    if [ -n "${STEPS_PER_EPOCH:-}" ] && [ "$STEPS_PER_EPOCH" -gt 0 ] 2>/dev/null; then
        echo "$STEPS_PER_EPOCH"; return 0
    fi
    if [ -n "${LOG_FILE:-}" ] && [ -f "$LOG_FILE" ]; then
        # [resume] 라인 "batch N/M" 패턴 우선
        local m
        m=$(grep -oE 'batch [0-9]+/[0-9]+' "$LOG_FILE" | tail -1 | sed 's|.*/||')
        [ -n "$m" ] && { echo "$m"; return 0; }
        # tqdm "Epoch X/Y: ...| N/M [..]" 패턴
        m=$(grep -oE '\| *[0-9]+/[0-9]+ \[' "$LOG_FILE" | tail -1 | sed 's|.*/||; s| \[||')
        [ -n "$m" ] && { echo "$m"; return 0; }
    fi
    echo 0
}

# 최신 keep_recent 개만 유지하고 나머지 global_step_* 디렉터리 삭제.
# 단 epoch 경계(step % steps_per_epoch == 0)인 ckpt 는 삭제 대신 preserved/ 로 이동.
# chunk 사이의 sleep 직전에 호출되므로 훈련 중에는 실행되지 않음 (동시성 이슈 없음).
rolling_prune() {
    [ "$keep_recent" -le 0 ] && return 0
    local all to_delete ckpt step spe preserve_dir
    mapfile -t all < <(ls -d "$save_path"/global_step_* 2>/dev/null | sort -V)
    local n=${#all[@]}
    (( n <= keep_recent )) && return 0
    local drop=$(( n - keep_recent ))
    to_delete=("${all[@]:0:$drop}")
    spe=$(get_steps_per_epoch)
    preserve_dir="$save_path/preserved"
    mkdir -p "$preserve_dir"
    local deleted=0 preserved=0
    echo "[run_experiment] rolling prune: keeping last $keep_recent of $n, candidates-to-drop=$drop, steps_per_epoch=$spe"
    for ckpt in "${to_delete[@]}"; do
        step=$(basename "$ckpt" | sed 's/global_step_//')
        if (( spe > 0 )) && (( step % spe == 0 )); then
            if [ -e "$preserve_dir/$(basename "$ckpt")" ]; then
                echo "  rm -rf $ckpt (duplicate in preserved/, keeping older copy)"
                rm -rf "$ckpt"
            else
                echo "  mv $ckpt -> preserved/ (epoch boundary step=$step)"
                mv "$ckpt" "$preserve_dir/"
                preserved=$((preserved + 1))
            fi
        else
            echo "  rm -rf $ckpt"
            rm -rf "$ckpt"
            deleted=$((deleted + 1))
        fi
    done
    echo "[run_experiment] rolling prune: deleted=$deleted preserved=$preserved (epoch boundaries moved to preserved/)"
}

# chunk 시작 전 디스크 여유 확인. 임계치 미만이면 실패로 종료 (컨테이너 죽는 것보다 빠른 중단이 안전).
preflight_disk() {
    local avail_gb
    avail_gb=$(df -BG --output=avail "$save_path" 2>/dev/null | tail -n 1 | tr -dc '0-9')
    if [ -z "$avail_gb" ]; then
        echo "[run_experiment] WARN: could not determine free space for $save_path"
        return 0
    fi
    echo "[run_experiment] free space on ckpt volume: ${avail_gb}G (min required: ${min_free_gb}G)"
    if (( avail_gb < min_free_gb )); then
        echo "[run_experiment] ERROR: free space ${avail_gb}G < ${min_free_gb}G — aborting before chunk to avoid disk-pressure kill" >&2
        exit 2
    fi
}

# ---------- Chunked training loop ----------
iteration=0
while :; do
    iteration=$((iteration + 1))
    current_step=$(get_latest_step)
    target_step=$((current_step + chunk_steps))

    echo "======================================================================"
    echo "[run_experiment] iter $iteration: $METHOD  step $current_step → $target_step"
    echo "======================================================================"
    date
    preflight_disk

    if [ "$current_step" = "0" ]; then
        run_trainer "${common_args[@]}" "trainer.total_training_steps=$target_step"
    else
        run_trainer "${common_args[@]}" \
            "trainer.total_training_steps=$target_step" \
            trainer.resume_path=auto
    fi

    new_step=$(get_latest_step)
    echo "[run_experiment] iter $iteration: ended at step $new_step (target $target_step)"

    if [ "$new_step" -le "$current_step" ]; then
        echo "[run_experiment] no progress — all epochs complete, exiting loop"
        break
    fi
    if [ "$new_step" -lt "$target_step" ]; then
        echo "[run_experiment] ended at $new_step < target $target_step — epochs exhausted, exiting"
        break
    fi

    rolling_prune
    echo "[run_experiment] sleeping ${sleep_seconds}s before next chunk..."
    sleep "$sleep_seconds"
done

echo "[run_experiment] $METHOD complete — final step $(get_latest_step)"
date

# AUTO_PRUNE=true 면 중간 step 체크포인트를 삭제하고 epoch 경계만 남김 (디스크 절약).
if [ "${AUTO_PRUNE:-false}" = "true" ]; then
    echo "[run_experiment] AUTO_PRUNE=true — pruning intermediate checkpoints..."
    bash "$(dirname "$0")/prune_epoch_ckpt.sh" "$save_path" --apply --epochs "$total_epochs"
fi
