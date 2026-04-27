# flatminima · Claude context

이 저장소는 `verl` (volcano-engine RL) 포크 기반의 **SFT vs DFT + ZO/AdaZO 변종 비교 실험**입니다. Qwen2.5-1.5B-Instruct를 numina-cot 데이터로 파인튜닝하고 MATH500으로 검증합니다.

**실행 환경**: ROCm 7.12 + torch 2.9.1 + MI300X 8장. vLLM 포함 rocm/vllm 컨테이너.

**실시간 현황판**: [`EXPERIMENT_STATUS.md`](./EXPERIMENT_STATUS.md) — 진행 상황·ckpt·데몬·복구 절차 일체.

---

## 핵심 컨벤션

### 훈련 실행
- **진입점**: `verl/run_all_experiments.sh` → 6개 METHOD(sft, sft+adazo, sft+zo, dft, dft+adazo, dft+zo) 순차 실행
- 각 method는 `verl/run_experiment.sh`가 chunked 방식으로 돌림 (100 step → exit → 10분 sleep → resume)
- **wandb는 offline 모드로만 작동** (컨테이너에 API 키 없음). `WANDB_MODE=offline` env 필수. 사용자가 `wandb login`하기 전까지 online 시도하면 즉사.

### 체크포인트 관리
- **steps_per_epoch = 390** (이 dataset + batch_size=256 기준)
- 1-epoch 경계 = step 390, 2-epoch = step 780, 3-epoch = step 1170
- `run_experiment.sh`의 `rolling_prune`이 chunk 사이에 최신 K=2 개만 유지, 나머지 삭제. **단 step % 390 == 0인 ckpt는 `$save_path/preserved/`로 이동해 영구 보존**.
- 이미 실행 중인 bash 프로세스는 함수 정의를 메모리에 들고 있어 파일 수정으로는 반영 안 됨 — 재실행 필요.
- **hardcoded 정리 스크립트 `prune_epoch_ckpt.sh`는 `max_step/epochs`를 경계로 가정**. 실제 경계(390)와 다를 수 있으니 dry-run 결과를 그대로 믿지 말고 사용자 확인 받을 것.

### 디스크 안전
- `/data/flatminima` 전체가 단일 볼륨 (overlay와 공유). 한도 **90G** 초과 시 `disk_monitor.sh`가 훈련 전체 kill.
- chunk당 ckpt 1개 (~9G), `KEEP_RECENT_CKPTS=2` + `preserved/` 분리로 안정 운영 시 **flat ~55-64G** 수준 유지.
- 새 ckpt 디렉터리 추가 전에는 반드시 예상 최대 사이즈 계산 후 90G 한도 확인.

### 파괴적 작업
- ckpt 대량 삭제는 사용자 명시 승인 없이 절대 금지 (rolling_prune 제외).
- `prune_epoch_ckpt.sh`는 반드시 **dry-run 먼저**, 리스트 보여준 후 `--apply`.
- 사용자가 step_390처럼 특정 ckpt를 보존하라고 지정하면 **그 뜻을 기억하고 이후에도 지킬 것** (수동 mv 필요 시 미리 계획).

---

## 자주 쓰는 명령

```bash
# 현재 훈련 상태 한눈에
tail -20 /data/flatminima/verl/logs/run_all_experiments.out
grep -E "step:[0-9]+" $(ls -t /data/flatminima/verl/logs/numina-cot-*.log | head -1) | tail -3
ls /data/flatminima/verl/checkpoints/*/ | head

# 디스크
tail -3 /data/flatminima/verl/logs/disk_monitor.log
du -sh /data/flatminima

# GPU
rocm-smi --showtemp --showpower --showuse

# 프로세스
pgrep -af "run_all_experiments|run_experiment.sh|torch.distributed.run|fsdp_sft_trainer" | head

# 개별 method 실행 (run_all 우회)
cd /data/flatminima/verl
WANDB_MODE=offline METHOD=sft+adazo MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct" \
  ZO_SIGMA=1e-3 SAM_RHO_MAX=20 SAM_RHO_MIN=2 \
  bash run_experiment.sh
```

---

## 중요한 파일

```
/data/flatminima/
├── EXPERIMENT_STATUS.md               # 실시간 현황판 (Section 5 = 현재 상태)
├── CLAUDE.md                          # 이 파일
├── verl/
│   ├── run_all_experiments.sh         # 6 method 오케스트레이션
│   ├── run_experiment.sh              # chunked trainer + rolling_prune
│   ├── prune_epoch_ckpt.sh            # method 완료 후 정리 (주의: max_step/epochs 가정)
│   ├── monitors/                      # 백그라운드 데몬들
│   │   ├── gpu_monitor.sh             # rocm-smi 30s 샘플
│   │   ├── progress_monitor.sh        # ckpt/proc/disk 60s 요약
│   │   ├── disk_monitor.sh            # 15s 디스크 감시 + 90G 초과 시 자동 kill
│   │   ├── preserve_then_relaunch.sh  # step 보존 + 재시작 (조건부)
│   │   └── pause_after_sft.sh         # sft 완료 후 run_all 자동 중단
│   ├── logs/                          # 모든 로그와 PID 파일
│   │   └── *.pid                      # 각 데몬 pid (kill 시 참조)
│   └── checkpoints/
│       └── numina-cot-*/
│           ├── global_step_N/         # rolling ckpt (prune 대상)
│           └── preserved/             # 영구 보존 (epoch 경계)
└── math_evaluation/                   # 평가용 (MATH500 등)
```

---

## 현재 진행 중 (2026-04-27 기준 · `EXPERIMENT_STATUS.md`는 04-22로 stale)

### 전체 상태
- **모든 훈련/데몬 정지**. 마지막 활동 04-26 00:12 (모니터 로그 멈춤).
- 6 method 중 **2개 완료**: sft (04-22, 원격 백업+purge), sft+adazo (04-25 17:40, val_loss 0.700).
- **sft+adazo 백업 실패**: 04-25T17:40 SSH `kex_exchange_identification: read: Connection reset by peer` → 28G ckpt가 `/data/flatminima/verl/checkpoints/numina-cot-sft-adazo-qwen-qwen2-5-1-5b-instruct/`에 그대로 남음 (preserved=[390,780], live=[1170]).
- 남은 4 method: **sft+zo → dft → dft+adazo → dft+zo** (이 순서로 `run_all_experiments.sh`의 experiments 배열, sft+adazo는 주석 처리됨).

### NVMe 마이그레이션 계획 (검토 완료, 미적용)
- 새 7T NVMe 3개 마운트됨: `/data1`, `/data2`, `/data3` (각 6.6T 여유).
- 합의된 설계:
  - **save_path**: `/data1/flatminima/verl/checkpoints/...`
  - **백업**: `/data2/flatminima/backup/...` (cross-disk local rsync — 원격 SSH 의존성 제거)
  - **로그**: `/data/flatminima/verl/logs/` 유지 (PID 호환성)
  - `BACKUP_AND_PURGE=true → false` (NVMe 여유로 method별 ckpt 모두 보존)
  - `disk_monitor.sh` `FLAT_MAX_GB`: 90 → 6000, `CKPT_ROOT`/`FLAT_ROOT`도 `/data1`로
- 디렉토리만 생성됨: `/data1/flatminima/verl/checkpoints`, `/data2/flatminima/backup`. 스크립트 수정은 미적용.
- 영향 받는 파일 4개: `run_all_experiments.sh`, `run_experiment.sh`, `monitors/backup_method.sh`, `monitors/disk_monitor.sh`.

### 미결정 항목
1. **기존 28G sft+adazo ckpt 처리** (a) `/data1`로 mv 후 백업 / (b) 그대로 두고 새 method만 NVMe.
2. **SAM 메트릭 0 조사**: sft+adazo wandb에서 `sam/eps_scale`, `sam/g_proj`, `sam/norm_z` 전부 0 — AdaZO ramp 미활성 의심. dft+adazo 시작 전 확인 권장.
3. **데몬 재시작 시점**: config 적용 후.

### 남은 실험 wall-clock 예상 (실측 기반)
- 기준: sft 4.5h, sft+adazo 11.5h
- sft+zo 6–8h, dft 5–6h, dft+adazo 11–13h, dft+zo 7–9h
- **합계 29–36h**, 운영 오버헤드(NCCL transient/chunk sleep) 포함 시 **36–44h (1.5–2일)**

---

## 이 프로젝트에서 피해야 할 것

- `run_all_experiments.sh`의 `experiments` 배열을 중간에 편집하지 말 것 (이미 실행 중인 bash에는 반영 안 됨).
- `WANDB_MODE=online` 강제하지 말 것 (키 없어서 즉사).
- 마운트별 공간 차이 주의: **`/data`는 LVM 295G(공유), `/data1/2/3`은 각 7T NVMe(독립)**. 새 워크로드는 NVMe 우선.
- 기존 `preserved/` 디렉터리 건드리지 말 것 — 사용자가 명시 보존한 step들.
- 원격 SSH 백업(`kyubyungchae@147.47.200.22`)은 04-25 실패 사례 있음 — 새 작업은 local cross-disk 백업 우선.
