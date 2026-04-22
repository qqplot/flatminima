# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
from contextlib import nullcontext

import hydra
import torch
import torch.distributed
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


class FSDPSFTTrainer:
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader(train_dataset, val_dataset)
        # build model
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)
        self.device_name = get_device_name()

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, (
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        self.val_sampler = DistributedSampler(
            self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if hasattr(self.model_config, "max_position_embeddings"):
            self.model_config.max_position_embeddings = max(
                self.model_config.max_position_embeddings, self.config.data.max_length
            )
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
            )

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        # betas must be a plain tuple; OmegaConf ListConfig breaks torch.distributed.checkpoint.
        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=tuple(self.config.optim.betas),
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/epoch {self.steps_per_epoch}, number of epochs "
                f"{self.config.trainer.total_epochs}, total number of steps {self.total_steps}"
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask = batch["loss_mask"][:, :-1].reshape(-1).to(self.device_name)
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # 💡 [추가] Config에서 DFT 사용 여부 확인 (기본값 False)
        # config.trainer.use_dft = true 로 설정하면 켜집니다.
        use_dft = getattr(self.config.trainer, "use_dft", False)

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                output = self.fsdp_model(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                logits = output.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                
                loss = loss_fct(shift_logits, shift_labels)

                # 💡 [DFT 적용] config가 켜져있을 때만 정답 토큰 예측 확률을 곱해줌
                if use_dft:
                    with torch.no_grad():
                        target_logits = shift_logits.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
                        log_sum_exp = torch.logsumexp(shift_logits, dim=-1)
                        prob_coefficients = (target_logits - log_sum_exp).exp()
                    loss = loss * prob_coefficients





                loss = loss * loss_mask.to(loss.device)
            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                batch_size, seqlen = input_ids.shape
                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size()
                )
                # For computing loss
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size()
                )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # Compute loss locally then aggregate
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # Gather and unpad for sequence parallelism
                loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(
                    hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                
                # SP 부분은 어차피 안 쓰신다 했으니 DFT 로직 생략하고 기본 CE loss 유지
                loss = full_loss * loss_mask

            valid_token_this_rank = torch.sum(loss_mask)

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            if do_backward:
                loss.backward()
            return loss

    # def _apply_noise_in_place(self, alpha, local_seed, global_seed):
    #     """In-place로 노이즈를 더해주고, 해당 시점의 Norm을 계산하여 반환합니다."""
    #     dp_size = self.device_mesh.size(0)
    #     local_norm_sq = 0.0

    #     for name, param in self.fsdp_model.named_parameters():
    #         if not param.requires_grad:
    #             continue

    #         # Sharded 여부 판별 (FSDP1 / FSDP2 혼용 대응)
    #         is_sharded = True
    #         if hasattr(param, "placements"): # FSDP2 (DTensor)
    #             from torch.distributed._tensor.placement_types import Replicate
    #             if all(isinstance(p, Replicate) for p in param.placements):
    #                 is_sharded = False
    #         elif "flat_param" not in name: # FSDP1 heuristic
    #             is_sharded = False

    #         seed_to_use = local_seed if is_sharded else global_seed

    #         # 파라미터별 독립된 동일 난수 생성 보장
    #         with torch.random.fork_rng(devices=[param.device]):
    #             torch.manual_seed(seed_to_use)
    #             z = torch.randn_like(param)

    #         # 1. Norm 계산 로직 (Replicated 파라미터 중복 합산 방지)
    #         z_local = z._local_tensor if hasattr(z, "_local_tensor") else z
    #         norm_val = float(z_local.pow(2).sum().item())
    #         if not is_sharded:
    #             norm_val /= float(dp_size) # 차후 all_reduce(SUM)에서 정상 값으로 복구됨
    #         local_norm_sq += norm_val

    #         # 2. In-place 노이즈 적용
    #         if hasattr(param, "_local_tensor") and hasattr(z, "_local_tensor"):
    #             param._local_tensor.add_(z._local_tensor, alpha=alpha)
    #         else:
    #             param.data.add_(z, alpha=alpha)
                
    #     return local_norm_sq


    # def training_step(self, batch: TensorDict, step: int = 0):
    #     self.fsdp_model.train()

    #     log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)
    #     self.optimizer.zero_grad()
    #     log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

    #     micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
    #     n_micro_batches = len(micro_batches)

    #     zo_sigma = self.config.optim.get("zo_sigma", 1e-3)
    #     sam_rho = self.config.optim.get("sam_rho", 0.0005)

    #     rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    #     # local_seed = step + rank * 100000 
    #     global_seed = step 
    #     local_seed = step + rank * 100000

    #     # =================================================================
    #     # 🛡️ [방어 1] PyTorch 공식 State Dict API를 통한 완벽한 상태 관리
    #     # (FSDP가 파라미터 크기를 몰래 바꿔도 알아서 안전하게 매핑해 줍니다!)
    #     # =================================================================
    #     if self.config.model.strategy == "fsdp2":
    #         from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
    #         options = StateDictOptions(full_state_dict=False, cpu_offload=False)
    #         orig_state = get_model_state_dict(self.fsdp_model, options=options)
    #         orig_state = {k: v.clone() for k, v in orig_state.items()}
            
    #         def apply_state(state):
    #             set_model_state_dict(self.fsdp_model, state, options=options)
    #     else:
    #         from torch.distributed.fsdp import StateDictType
    #         with FSDP.state_dict_type(self.fsdp_model, StateDictType.LOCAL_STATE_DICT):
    #             orig_state = {k: v.clone() for k, v in self.fsdp_model.state_dict().items()}
                
    #         def apply_state(state):
    #             with FSDP.state_dict_type(self.fsdp_model, StateDictType.LOCAL_STATE_DICT):
    #                 self.fsdp_model.load_state_dict(state, strict=False)

    #     # 업데이트가 필요한(requires_grad) 파라미터 이름만 추출
    #     grad_keys = {k for k, p in self.fsdp_model.named_parameters() if p.requires_grad}

    #     def apply_noise(alpha, current_local_seed):
    #         noisy_state = {}
    #         for k, v in orig_state.items():
    #             if k in grad_keys:
    #                 # 파라미터가 샤딩되었는지 확인 (DTensor의 경우 placements 검사)
    #                 is_sharded = True
    #                 if hasattr(v, "placements"): # FSDP2 (DTensor)
    #                     # Replicate 상태인지 Shard 상태인지 확인
    #                     from torch.distributed._tensor.placement_types import Replicate
    #                     if all(isinstance(p, Replicate) for p in v.placements):
    #                         is_sharded = False
    #                 elif getattr(v, "numel", lambda: 0)() == getattr(self.model.get_parameter(k), "numel", lambda: -1)():
    #                     # FSDP1 대략적 추론: 텐서 크기가 원본과 같으면 복제된 것 (예: LayerNorm)
    #                     is_sharded = False
                    
    #                 # 복제된 파라미터는 노드 간 값이 완전히 일치해야 하므로 global_seed 사용
    #                 seed_to_use = current_local_seed if is_sharded else global_seed
                    
    #                 # 🛠️ torch.random.fork_rng로 안전한 난수 제어
    #                 with torch.random.fork_rng(devices=[v.device]):
    #                     torch.manual_seed(seed_to_use)
    #                     z = torch.randn_like(v)
                    
    #                 noisy_v = v.clone()
                    
    #                 # 🛠️ [버그 4 수정] DTensor의 경우 In-place 연산 충돌을 피하기 위해 로컬 텐서에 직접 연산
    #                 if hasattr(noisy_v, "_local_tensor") and hasattr(z, "_local_tensor"):
    #                     noisy_v._local_tensor.add_(z._local_tensor, alpha=alpha)
    #                 else:
    #                     noisy_v.add_(z, alpha=alpha)
                        
    #                 noisy_state[k] = noisy_v
    #             else:
    #                 noisy_state[k] = v.clone()
            
    #         apply_state(noisy_state)


    #     # =================================================================
    #     # [AdaZo-SAM 단계 1] ZO 탐색 (eval 모드)
    #     # =================================================================
    #     self.fsdp_model.eval()

    #     with torch.no_grad():
    #         # 1. +Z 방향 탐색
    #         apply_noise(zo_sigma, local_seed)
    #         loss1_local = sum(
    #             self._compute_loss_and_backward(mb, do_backward=False).item() / n_micro_batches 
    #             for mb in micro_batches
    #         )
            
    #         # 2. -Z 방향 탐색
    #         apply_noise(-zo_sigma, local_seed)
    #         loss2_local = sum(
    #             self._compute_loss_and_backward(mb, do_backward=False).item() / n_micro_batches 
    #             for mb in micro_batches
    #         )

    #     # Loss 통신 및 방향 추정
    #     loss_tensor = torch.tensor([loss1_local, loss2_local], device=self.device_name)
    #     if torch.distributed.is_initialized():
    #         torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
    #     global_loss1, global_loss2 = loss_tensor[0].item(), loss_tensor[1].item()

    #     g_proj = (global_loss1 - global_loss2) / (2 * zo_sigma)

    #     # Norm Z 계산 (State Dict 기반으로 일관성 100% 보장)
    #     local_norm_sq = 0.0
    #     torch.manual_seed(local_seed)
    #     for k, v in orig_state.items():
    #         if k in grad_keys:
    #             z = torch.randn_like(v)
    #             z_local = z.to_local() if hasattr(z, "to_local") else z
    #             local_norm_sq += float(z_local.pow(2).sum().item())
                
    #     norm_tensor = torch.tensor([local_norm_sq], device=self.device_name)
    #     if torch.distributed.is_initialized():
    #         torch.distributed.all_reduce(norm_tensor, op=torch.distributed.ReduceOp.SUM)
    #     norm_z = (norm_tensor[0].item() + 1e-12) ** 0.5
        
    #     # 최악의 위치 방향 결정
    #     eps_scale = sam_rho * torch.sign(torch.tensor(g_proj)).item() / norm_z

    #     # =================================================================
    #     # [AdaZo-SAM 단계 2] 최악의 위치 세팅 후 훈련 모드 재개
    #     # =================================================================

    #     # apply_noise 호출 직전에 추가
    #     # before_norm = sum(p.sum().item() for p in self.fsdp_model.parameters() if p.requires_grad)

    #     # apply_noise(zo_sigma, local_seed) # 노이즈 주입!
    #     apply_noise(eps_scale, local_seed) # 노이즈 주입!

    #     # after_norm = sum(p.sum().item() for p in self.fsdp_model.parameters() if p.requires_grad)
    #     # if rank == 0:
    #     #     print(f"가중치 변화 확인 - Before: {before_norm:.4f}, After: {after_norm:.4f}")

    #     self.fsdp_model.train()

    #     # 실제 Backward Pass 1회 수행
    #     step_loss = 0
    #     for micro_batch in micro_batches:
    #         loss = self._compute_loss_and_backward(batch=micro_batch, do_backward=True) / n_micro_batches
    #         step_loss += loss.item()

    #     if self.config.model.strategy == "fsdp":
    #         grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
    #     elif self.config.model.strategy == "fsdp2":
    #         grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
    #     else:
    #         raise NotImplementedError(f"not implement {self.config.model.strategy}")

    #     # =================================================================
    #     # 🛡️ [방어 3] 옵티마이저 스텝 직전, 가중치를 '안전하게' 원본 완벽 복원
    #     # =================================================================
    #     apply_state(orig_state)

    #     # --- 옵티마이저 업데이트 ---
    #     log_gpu_memory_usage("Before optimizer step", logger=logger)

    #     if not torch.isfinite(grad_norm):
    #         print(f"WARN: grad_norm is not finite: {grad_norm}")
    #         self.optimizer.zero_grad()
    #     else:
    #         self.optimizer.step()

    #     log_gpu_memory_usage("After optimizer step", logger=logger)
    #     self.lr_scheduler.step()

    #     lr = self.lr_scheduler.get_last_lr()[0]
    #     step_loss = torch.tensor(step_loss).to(self.device_name)
        
    #     if is_cuda_available:
    #         torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
    #     elif is_npu_available:
    #         torch.distributed.all_reduce(step_loss)
    #         step_loss /= self.device_mesh.size(0)
            
    #     # return {"train/loss": step_loss.detach().item(), "train/lr(1e-3)": lr * 1e3}
    #     return {
    #                 "train/loss": step_loss.detach().item(), 
    #                 "train/lr(1e-3)": lr * 1e3,
    #                 "sam/eps_scale": eps_scale,  # 실제 이동 스텝
    #                 "sam/norm_z": norm_z,        # 노이즈 스케일
    #                 "sam/g_proj": g_proj         # 방향성(기울기) 추정치
    #             }        

    def training_step(self, batch: TensorDict, step: int = 0):
        self.fsdp_model.train()
        self.optimizer.zero_grad()

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)

        zo_sigma = self.config.optim.get("zo_sigma", 1e-3)
        # sam_rho = self.config.optim.get("sam_rho", 0.0005)
        # =================================================================
        # 🔥 GSAM 기반 sam_rho 동적 스케줄링 (bfloat16 안전장치 포함)
        # =================================================================
        current_lr = self.lr_scheduler.get_last_lr()[0]
        lr_max = self.config.optim.lr
        lr_min = 0.0
        
        # config에서 max, min 값을 받도록 수정 (기본값 세팅)
        rho_max = self.config.optim.get("sam_rho_max", 20.0) 
        rho_min = self.config.optim.get("sam_rho_min", 2.0)  
        
        if lr_max > lr_min:
            sam_rho = rho_min + (rho_max - rho_min) * ((current_lr - lr_min) / (lr_max - lr_min))
        else:
            sam_rho = rho_max

        sam_rho = max(rho_min, float(sam_rho))
        # =================================================================


        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        global_seed = step 
        local_seed = step + rank * 100000

        # 업데이트가 필요한 파라미터 키 추출
        grad_keys = {k for k, p in self.fsdp_model.named_parameters() if p.requires_grad}

        # =================================================================
        # 🛡️ [메모리 안전 백업] PyTorch 공식 State Dict API 활용 (질문자님 원본 구조)
        # =================================================================
        is_fsdp2 = self.config.model.strategy == "fsdp2"
        
        if is_fsdp2:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
            options = StateDictOptions(full_state_dict=False, cpu_offload=False)
            raw_state = get_model_state_dict(self.fsdp_model, options=options)
        else:
            from torch.distributed.fsdp import StateDictType
            with FSDP.state_dict_type(self.fsdp_model, StateDictType.LOCAL_STATE_DICT):
                raw_state = self.fsdp_model.state_dict()
                
        # 로컬 Shard 파라미터 복사 (GPU당 약 700MB 수준이므로 OOM 걱정 없음)
        orig_state = {k: v.clone() for k, v in raw_state.items()}
        
        def apply_state(state_dict):
            """FSDP 내부 로직을 안전하게 통과하는 공식 상태 업데이트"""
            if is_fsdp2:
                set_model_state_dict(self.fsdp_model, state_dict, options=options)
            else:
                with FSDP.state_dict_type(self.fsdp_model, StateDictType.LOCAL_STATE_DICT):
                    self.fsdp_model.load_state_dict(state_dict, strict=False)

        dp_size = self.device_mesh.size(0)

        def get_noisy_state_and_norm(alpha):
            """Drift(표류) 없는 노이즈 상태 및 완벽한 Norm 계산"""
            noisy_state = {}
            local_norm_sq = 0.0
            
            for k, v in orig_state.items():
                if k in grad_keys:
                    is_sharded = True
                    if hasattr(v, "placements"): # FSDP2
                        from torch.distributed._tensor.placement_types import Replicate
                        if all(isinstance(p, Replicate) for p in v.placements):
                            is_sharded = False
                    elif "flat_param" not in k: # FSDP1 휴리스틱
                        is_sharded = False

                    seed_to_use = local_seed if is_sharded else global_seed

                    with torch.random.fork_rng(devices=[v.device]):
                        torch.manual_seed(seed_to_use)
                        z = torch.randn_like(v)

                    # Norm 계산 (Replicated 파라미터의 중복 합산 방지)
                    z_local = z._local_tensor if hasattr(z, "_local_tensor") else z
                    norm_val = float(z_local.pow(2).sum().item())
                    if not is_sharded:
                        norm_val /= float(dp_size)
                    
                    local_norm_sq += norm_val
                    
                    # In-place 덧셈(add_) 대신 수학적으로 깨끗한 독립 텐서 생성
                    noisy_state[k] = v + z * alpha
                else:
                    noisy_state[k] = v.clone()

            return noisy_state, local_norm_sq

        # =================================================================
        # [AdaZo-SAM 단계 1] ZO 탐색 (eval 모드)
        # =================================================================
        self.fsdp_model.eval()

        with torch.no_grad():
            # 1. +Z 방향
            noisy_state_plus, local_norm_sq = get_noisy_state_and_norm(zo_sigma)
            apply_state(noisy_state_plus)
            loss1_local = sum(self._compute_loss_and_backward(mb, do_backward=False).item() / n_micro_batches for mb in micro_batches)
            
            # 2. -Z 방향
            noisy_state_minus, _ = get_noisy_state_and_norm(-zo_sigma)
            apply_state(noisy_state_minus)
            loss2_local = sum(self._compute_loss_and_backward(mb, do_backward=False).item() / n_micro_batches for mb in micro_batches)

        # Loss 통신 및 방향 추정
        loss_tensor = torch.tensor([loss1_local, loss2_local], device=self.device_name)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
        global_loss1, global_loss2 = loss_tensor[0].item(), loss_tensor[1].item()

        g_proj = (global_loss1 - global_loss2) / (2 * zo_sigma)

        norm_tensor = torch.tensor([local_norm_sq], device=self.device_name)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(norm_tensor, op=torch.distributed.ReduceOp.SUM)
        norm_z = (norm_tensor[0].item() + 1e-12) ** 0.5
        
        eps_scale = sam_rho * torch.sign(torch.tensor(g_proj)).item() / norm_z

        # =================================================================
        # [AdaZo-SAM 단계 2] 최악의 위치 세팅 후 훈련 모드 재개
        # =================================================================
        noisy_state_eps, _ = get_noisy_state_and_norm(eps_scale)
        apply_state(noisy_state_eps)
        self.fsdp_model.train()

        # 실제 Backward Pass 수행
        step_loss = 0
        for i, micro_batch in enumerate(micro_batches):
            is_last_microbatch = (i == len(micro_batches) - 1)
            context = self.fsdp_model.no_sync() if not is_last_microbatch and hasattr(self.fsdp_model, "no_sync") else nullcontext()
            with context:
                loss = self._compute_loss_and_backward(batch=micro_batch, do_backward=True)
                step_loss += (loss.item() / n_micro_batches)

        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)

        # =================================================================
        # 🛡️ 옵티마이저 스텝 직전, 가중치를 안전하게 원본 복원
        # =================================================================
        apply_state(orig_state)

        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]
        step_loss = torch.tensor(step_loss).to(self.device_name)
        
        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            step_loss /= self.device_mesh.size(0)

        base_loss_approx = (global_loss1 + global_loss2) / 2.0
        adv_loss = step_loss.detach().item()

        return {
            "train/loss": step_loss.detach().item(), 
            "train/lr(1e-3)": lr * 1e3,
            "sam/eps_scale": eps_scale,
            "sam/sam_rho": sam_rho,            
            "sam/norm_z": norm_z,
            "sam/g_proj": g_proj,
            "sam/base_loss_approx": base_loss_approx, # 원래 위치의 평화로운(?) Loss
            "sam/loss_diff": adv_loss - base_loss_approx # 얼마나 악화시켰는가! (클수록 SAM이 강하게 들어간 것)
        }

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.device_mesh.size(0)
        return loss

    def save_checkpoint(self, step):
        # save checkpoint
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            # FSDP1 checkpoint saving
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.fsdp_model.state_dict()

            # save huggingface model
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.tokenizer.save_pretrained(path)
        elif fsdp_strategy == "fsdp2":
            # FSDP2 checkpoint saving
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            # Get full state dict with FSDP2
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(self.fsdp_model, options=options)

            # save huggingface model
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.model_config.save_pretrained(path)
                self.tokenizer.save_pretrained(path)
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        from verl.trainer._sft_resume_utils import save_trainer_state

        save_trainer_state(
            self.fsdp_model, self.optimizer, self.lr_scheduler,
            step, path, fsdp_strategy, self.device_mesh.get_rank(),
        )

        # Copy to HDFS if configured
        if self.device_mesh.get_rank() == 0 and self.config.trainer.default_hdfs_dir:
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        global_step = 0
        last_valid_metric = None
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        from verl.trainer._sft_resume_utils import load_trainer_state, resolve_resume_path

        resume_path = resolve_resume_path(
            getattr(self.config.trainer, "resume_path", None),
            self.config.trainer.default_local_dir,
            rank,
        )
        if resume_path:
            global_step = load_trainer_state(
                self.fsdp_model, self.optimizer, self.lr_scheduler,
                resume_path, self.config.model.strategy, self.device_mesh,
            )
        start_epoch = global_step // self.steps_per_epoch
        skip_batches = global_step % self.steps_per_epoch
        if rank == 0 and global_step > 0:
            print(
                f"[resume] global_step={global_step} / total={self.total_training_steps} | "
                f"start at epoch {start_epoch + 1}/{self.config.trainer.total_epochs}, "
                f"batch {skip_batches + 1}/{self.steps_per_epoch}"
            )

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for i, data in enumerate(tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                disable=rank != 0,
            )):
                if epoch == start_epoch and i < skip_batches:
                    continue
                global_step += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                # metric = self.training_step(data)
                metric = self.training_step(data, step=global_step)

                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0
                is_epoch_end = (i + 1) == self.steps_per_epoch

                # early exit or validation step
                if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                    # Perform validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).to(
                            self.device_name
                        )
                        val_loss = self.validation_step(val_data)
                        val_losses.append(val_loss)
                    if rank == 0:
                        val_loss = torch.mean(torch.stack(val_losses))
                        metric = {"val/loss": val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step) or is_epoch_end:
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.fit()

    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    else:
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    return dataset


if __name__ == "__main__":
    main()
