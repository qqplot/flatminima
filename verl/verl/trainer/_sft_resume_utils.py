# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Shared resume/save helpers for the FSDP SFT-family trainers (SFT, DFT, AdaZO, ZO).

Each of these trainers has its own FSDPSFTTrainer class but the checkpoint save/load
logic is identical. Rather than inlining the same ~100 lines in every file, they all
import these helpers and call them at the appropriate points.

Checkpoint layout:
    <default_local_dir>/global_step_<N>/
        config.json            (HF model config)
        model.safetensors      (HF weights)
        tokenizer*             (HF tokenizer)
        trainer_state.pt       (optimizer + lr_scheduler + global_step; added by save_trainer_state)
"""

import os
import re

import torch
import torch.distributed


def extract_step(path):
    """Return the integer N encoded in a ``global_step_N`` path fragment, or ``None``."""
    match = re.search(r"global_step_(\d+)", path)
    return int(match.group(1)) if match else None


def find_latest_checkpoint(base_dir):
    """Return the ``global_step_N`` sub-directory of ``base_dir`` with the largest N, or ``None``."""
    if not base_dir or not os.path.isdir(base_dir):
        return None
    best = None
    for name in os.listdir(base_dir):
        step = extract_step(name)
        if step is None:
            continue
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        if best is None or step > best[0]:
            best = (step, full)
    return best[1] if best else None


def save_trainer_state(fsdp_model, optimizer, lr_scheduler, step, path, fsdp_strategy, rank):
    """Save optimizer / lr_scheduler / global_step to ``path/trainer_state.pt``.

    Must be called on ALL ranks (the state dict gather is collective). Only rank 0 writes
    the file.
    """
    if fsdp_strategy == "fsdp":
        from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType

        sd_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        opt_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, sd_cfg, opt_cfg):
            optim_state = FSDP.optim_state_dict(fsdp_model, optimizer)
    elif fsdp_strategy == "fsdp2":
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        optim_state = get_optimizer_state_dict(fsdp_model, optimizer, options=options)
    else:
        raise NotImplementedError(f"save_trainer_state: unsupported fsdp_strategy={fsdp_strategy}")

    if rank == 0:
        torch.save(
            {
                "optimizer": optim_state,
                "lr_scheduler": lr_scheduler.state_dict(),
                "global_step": step,
            },
            os.path.join(path, "trainer_state.pt"),
        )


def load_trainer_state(fsdp_model, optimizer, lr_scheduler, path, fsdp_strategy, device_mesh):
    """Load model weights + optimizer + lr_scheduler from ``path``.

    Returns the global_step to resume from. If ``trainer_state.pt`` is missing, falls back
    to the step encoded in the directory name (legacy weight-only checkpoints) or 0.
    """
    from transformers import AutoModelForCausalLM

    rank = device_mesh.get_rank()

    if rank == 0:
        tmp_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
        model_state = tmp_model.state_dict()
        del tmp_model
    else:
        model_state = {}

    if fsdp_strategy == "fsdp":
        from torch.distributed.fsdp import FullStateDictConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType

        sd_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, sd_cfg):
            fsdp_model.load_state_dict(model_state if rank == 0 else {}, strict=False)
    elif fsdp_strategy == "fsdp2":
        from verl.utils.fsdp_utils import fsdp2_load_full_state_dict

        fsdp2_load_full_state_dict(fsdp_model, model_state, device_mesh, None)
    else:
        raise NotImplementedError(f"load_trainer_state: unsupported fsdp_strategy={fsdp_strategy}")

    trainer_state_path = os.path.join(path, "trainer_state.pt")
    if not os.path.exists(trainer_state_path):
        fallback_step = extract_step(os.path.basename(os.path.normpath(path))) or 0
        if rank == 0:
            print(
                f"[resume] {trainer_state_path} not found; weights loaded but optimizer/"
                f"lr_scheduler not restored — step={fallback_step}"
            )
        torch.distributed.barrier()
        return fallback_step

    trainer_state = torch.load(trainer_state_path, map_location="cpu", weights_only=False)

    if fsdp_strategy == "fsdp":
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        flat_osd = FSDP.optim_state_dict_to_load(
            model=fsdp_model, optim=optimizer, optim_state_dict=trainer_state["optimizer"]
        )
        optimizer.load_state_dict(flat_osd)
    elif fsdp_strategy == "fsdp2":
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict

        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        set_optimizer_state_dict(fsdp_model, optimizer, trainer_state["optimizer"], options=options)

    lr_scheduler.load_state_dict(trainer_state["lr_scheduler"])
    global_step = int(trainer_state["global_step"])

    if rank == 0:
        print(f"[resume] loaded checkpoint from {path}, resuming at global_step={global_step}")
    torch.distributed.barrier()
    return global_step


def resolve_resume_path(config_resume_path, default_local_dir, rank):
    """Translate ``trainer.resume_path`` into a concrete path.

    - ``None`` -> ``None`` (no resume)
    - ``"auto"`` -> ``find_latest_checkpoint(default_local_dir)``, logs result on rank 0
    - anything else -> returned as-is
    """
    if config_resume_path is None:
        return None
    if config_resume_path == "auto":
        path = find_latest_checkpoint(default_local_dir)
        if rank == 0:
            if path:
                print(f"[resume] auto-detected latest checkpoint: {path}")
            else:
                print(
                    f"[resume] auto mode: no checkpoint found under {default_local_dir}, "
                    "starting from scratch"
                )
        return path
    return config_resume_path
