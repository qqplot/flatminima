
nproc_per_node=2
project_name=numina-cot


model_name=""
experiment_name=""
save_dir=""

save_path=$save_dir/checkpoints/$experiment_name

lr=5e-5
zo_sigma=1e-3
sam_rho=0.01
weight_decay=0.01


torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
        -m verl.trainer.fsdp_adazo_trainer \
    data.train_files=data/numina_cot/train.parquet \
    data.val_files=data/math500/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=512 \
    data.max_length=4096 \
    optim.lr=$lr \
    optim.weight_decay=$weight_decay \
    optim.zo_sigma=$zo_sigma \
    optim.sam_rho=$sam_rho \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=$model_name \
    model.use_liger=True \
    model.fsdp_config.model_dtype=bf16 \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name="$experiment_name-$(date +%Y%m%d-%H%M%S)" \
    trainer.logger=['console','tensorboard','wandb'] \
    trainer.default_hdfs_dir=null \
    trainer.test_freq=10 \
    trainer.save_freq=50 \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=$nproc_per_node \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true
