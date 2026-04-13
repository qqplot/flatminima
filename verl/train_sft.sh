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

# 2. 환경 변수 설정 (에러 방지 및 최적화)
export CUDA_VISIBLE_DEVICES=0,1          # Slurm이 알아서 할당하지만 명시적으로 지정 가능
export NCCL_P2P_DISABLE=0                # Blackwell NVLink 성능 극대화 (P2P 허용)
export NCCL_IB_DISABLE=1                 # 단일 노드인 경우 인피니밴드 무시

source ~/anaconda3/etc/profile.d/conda.sh
conda activate DFT

nproc_per_node=2
project_name=numina-cot

model_name="Qwen/Qwen3-0.6B-Base"
experiment_name="numina-cot-sft-qwen-3-base-0.6b"


save_path=checkpoints/$experiment_name
lr=5e-5

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m verl.trainer.fsdp_sft_trainer \
    data.train_files=data/numina_cot/train.parquet \
    data.val_files=data/math500/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=256 \
    data.max_length=4096 \
    optim.lr=$lr \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=8 \
    model.partial_pretrain=$model_name \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=10 \
    trainer.save_freq=50 \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=$nproc_per_node \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true


# nohup bash train_sft.sh > train_sft_4096.log 2>&1 &