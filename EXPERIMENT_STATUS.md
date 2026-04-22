# flatminima 실험 현황 · 통합 상황판

_이 문서는 현재 진행 중인 numina-cot SFT/DFT 비교 실험의 **단일 진실 출처(SSoT)** 입니다. 로그 · 체크포인트 · 데몬 · 정지/재시작 절차를 한 자리에 모아 두어, 누구든 이 파일만 보고 진행 상황과 다음 단계를 파악할 수 있도록 설계했습니다._

**마지막 업데이트**: 2026-04-22 09:16 UTC (**sft 완주 + pause 발동 완료 🛑**)

---

## 1. 실험 개요

| 항목 | 값 |
|---|---|
| 프로젝트 | `numina-cot` |
| 모델 | `Qwen/Qwen2.5-1.5B-Instruct` |
| 학습 데이터 | `data/numina_cot/train.parquet` |
| 검증 데이터 | `data/math500/test.parquet` |
| batch size | 256 (micro 16 × 8 GPUs) |
| max_length | 4096 |
| optim | AdamW, lr 5e-5, weight_decay 0.01, cosine scheduler, warmup 10% |
| epochs | 3 |
| steps_per_epoch | **390** (= dataset size / batch_size, log 에서 자동 추출됨) |
| total_steps | 1170 (= 3 × 390) |
| GPU | 8× AMD MI300X (ROCm 7.12.0rc0 / torch 2.9.1) |
| wandb | **offline 모드** (컨테이너에 API 키 없음, 나중에 `wandb sync`로 업로드) |

## 2. Method 시퀀스 (6개, 순차 실행)

`run_all_experiments.sh`가 순서대로 실행:
1. **sft** — 표준 SFT (진행 중)
2. sft+adazo (ZO_SIGMA=1e-3, SAM_RHO_MAX=20, SAM_RHO_MIN=2) — ⏸ 사용자 요청으로 sft 완료 후 pause
3. sft+zo
4. dft
5. dft+adazo (ZO_SIGMA=5e-3, SAM_RHO_MAX=10, SAM_RHO_MIN=1)
6. dft+zo

**현재 사용자 요청**: sft method의 3 epoch 완료 후 **자동 pause** (sft+adazo 시작 전). 이후 이어서 실행할지 별도 결정.

## 3. Chunked training 인프라

| 파라미터 | 값 | 의미 |
|---|---|---|
| `CHUNK_STEPS` | 100 | 100 step마다 trainer exit → sleep → resume |
| `SLEEP_SECONDS` | 600 (10 min) | chunk 사이 GPU/노드 쿨링 간격 |
| `SAVE_FREQ` | 100 | 100 step마다 ckpt 저장 (+epoch 경계 자동) |
| `TEST_FREQ` | 50 | 50 step마다 validation |
| `KEEP_RECENT_CKPTS` | 2 | rolling_prune이 유지할 최신 ckpt 수 |
| `MIN_FREE_GB` | 12 | chunk 시작 전 여유 디스크 임계 |
| `STEPS_PER_EPOCH` | 자동(390) | 로그에서 추출, epoch 경계 보호에 사용 |
| `AUTO_PRUNE` | true | method 완료 시 epoch 경계만 남기고 정리 |

## 4. 디스크 안전 장치

- **disk_monitor**: 15초 주기로 `/data/flatminima` 크기 관측 → `FLAT_MAX_GB=90` 초과 시 `SIGTERM` run_all + 하위 트레이너 전부 정리
- **preflight**: chunk 시작 전 `avail < 12GB`면 즉시 fail
- **rolling_prune**: chunk 종료마다 최신 2개 유지, 나머지 삭제. 단 **step % 390 == 0** (epoch 경계)이면 `preserved/`로 이동해 영구 보존

## 5. 현재 상태 (09:16 UTC · **sft 완료, pause 상태**)

### 진행 상황
- **METHOD=sft** ✅ **3 epoch 완주** (step 0→1170), final step 1170 저장됨
- **pause_after_sft** 정상 발동 (08:59:38) → `run_all_experiments.sh` + `run_experiment.sh` SIGTERM/SIGKILL 완료 → sft+adazo 시작 5초 만에 정지됨
- **훈련 프로세스**: 전부 종료 (run_all/run_experiment/torch.distributed/fsdp 모두 없음)
- **데몬**: gpu_monitor, progress_monitor, disk_monitor는 계속 가동 중

### val_loss 추이 (과적합 없음, 완전 안정)
step 750 → 800 → 900 → 1000 → 1050 → 1150 : **0.724 → 0.737 → 0.736 → 0.737 → 0.737 → 0.737**

### ckpt 상태 (2026-04-22 09:16)
```
/data/flatminima/verl/checkpoints/
├── numina-cot-sft-qwen-qwen2-5-1-5b-instruct/  (sft 완료, 28G)
│   ├── live: [1170]                          🏆 최종 모델 (AUTO_PRUNE 결과)
│   └── preserved: [390, 780]                 🛡️ epoch 1, 2 경계
├── numina-cot-sft-adazo-qwen-qwen2-5-1-5b-instruct/  (8K, 빈 dir — pause 직후 kill된 흔적)
└── numina-cot-sft-qwen-2.5-math-1.5b/  (이전 실험 잔해, 19G)
    └── [390, 400]                            (삭제 여부 미결정)
```

### 디스크
- 볼륨: 295G, Used 183G, **Avail 112G (61%)**
- `/data/flatminima` 실사용: **46G** (한도 90G, 여유 44G)
- `disk_monitor` 연속 OK

### wandb offline 로그 (나중에 sync)
```
/data/flatminima/verl/wandb/offline-run-20260422_044840-*    (첫 run, resume 이전)
/data/flatminima/verl/wandb/offline-run-20260422_070845-*    (2차 run, 패치 적용 후)
/data/flatminima/verl/wandb/offline-run-20260422_085318-*    (iter 4 구간)
```

## 6. 데몬 · 자동화 스크립트

모든 백그라운드 스크립트는 `/data/flatminima/verl/logs/<name>.pid`로 PID 추적.

| 스크립트 | 기능 | 상태 |
|---|---|---|
| `monitors/gpu_monitor.sh` | rocm-smi 30초 주기 | ✅ 가동 중 |
| `monitors/progress_monitor.sh` | 체크포인트/디스크/프로세스 60초 요약 | ✅ 가동 중 |
| `monitors/disk_monitor.sh` | 15초 디스크 감시 + 90G 초과시 자동 kill | ✅ 가동 중 |
| `monitors/preserve_then_relaunch.sh` | step 780 보존 + 재시작 (1회성) | ✅ 임무 완료 후 exit |
| `monitors/pause_after_sft.sh` | sft 완료 감지 → run_all 종료 | ✅ 가동 중 (sft 완주 대기) |
| 5분 cron (job `6ac864b3`) | Claude 정기 보고 | 세션 종료 시 만료 |

## 7. 코드 패치 이력

| 파일 | 변경 | 이유 |
|---|---|---|
| `run_experiment.sh` | SAVE_FREQ 50→100 기본값 | chunk당 ckpt 2개→1개로 줄여 디스크 압박 완화 |
| `run_experiment.sh` | `KEEP_RECENT_CKPTS`, `MIN_FREE_GB`, `STEPS_PER_EPOCH` env 추가 | rolling prune + preflight + epoch 보호 |
| `run_experiment.sh` | `rolling_prune` 함수 신규 + epoch 경계 자동 `mv preserved/` | 훈련 중단 누적 방지 + epoch ckpt 보존 |
| `run_experiment.sh` | `preflight_disk` 함수 | chunk 시작 전 디스크 점검, 부족시 즉시 fail |
| `monitors/disk_monitor.sh` | `FLAT_MAX_GB=90` 초과 시 자동 kill | 컨테이너 OOM/kill 회피 |

## 8. 주요 파일 경로

```
훈련 스크립트
  /data/flatminima/verl/run_all_experiments.sh   # 6개 method 순차 실행
  /data/flatminima/verl/run_experiment.sh        # chunked training wrapper
  /data/flatminima/verl/prune_epoch_ckpt.sh      # method 완료 후 epoch-only 정리

로그
  /data/flatminima/verl/logs/run_all_experiments.out             # 전체 out
  /data/flatminima/verl/logs/run_all-<ts>.log                    # master log (timestamped)
  /data/flatminima/verl/logs/numina-cot-sft-*.log                # method별 exp log
  /data/flatminima/verl/logs/gpu_monitor.log                     # rocm-smi 샘플
  /data/flatminima/verl/logs/progress_monitor.log                # ckpt/disk/proc 요약
  /data/flatminima/verl/logs/disk_monitor.log                    # 디스크 15s 관측
  /data/flatminima/verl/logs/preserve_then_relaunch.log          # step 780 보존 이력
  /data/flatminima/verl/logs/pause_after_sft.log                 # pause watcher
  /data/flatminima/verl/logs/pause_after_sft.triggered           # pause 발동 표식

체크포인트
  /data/flatminima/verl/checkpoints/numina-cot-sft-qwen-qwen2-5-1-5b-instruct/
    global_step_N/                                   # rolling pruned live ckpts
    preserved/global_step_N/                         # epoch 경계 보존 (영구)
    wandb_run_id.txt                                 # resume 용 wandb id

wandb offline 데이터
  /data/flatminima/verl/wandb/offline-run-*/         # 나중에 `wandb sync <path>` 로 업로드
```

## 9. Pause 상태 이후 재개 방법

sft가 3 epoch 완주하고 `pause_after_sft.sh`가 발동해 run_all을 종료하면:

```bash
# 1) pause 발동 확인
cat /data/flatminima/verl/logs/pause_after_sft.triggered
ls -la /data/flatminima/verl/checkpoints/numina-cot-sft-qwen-qwen2-5-1-5b-instruct/

# 2) 다음 method(sft+adazo)부터 이어서 돌리려면:
#    run_all_experiments.sh 의 experiments 배열을 sft 제외로 편집하거나,
#    아래처럼 method 개별 실행:
cd /data/flatminima/verl
WANDB_MODE=offline METHOD=sft+adazo MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct" \
  ZO_SIGMA=1e-3 SAM_RHO_MAX=20 SAM_RHO_MIN=2 \
  AUTO_PRUNE=true KEEP_RECENT_CKPTS=2 \
  bash run_experiment.sh

# 3) 혹은 원래 대로 run_all_experiments.sh 를 다시 실행하면 sft는 skip(체크포인트 있음→resume 후 즉시 완료)되고 다음 method 로 진행
cd /data/flatminima/verl
WANDB_MODE=offline nohup bash run_all_experiments.sh > logs/run_all_experiments.out 2>&1 &
echo $! > logs/run_all_experiments.pid
```

## 10. 한 눈에 보는 다음 행동 (checklist)

- [x] sft 완주 대기 ✅ (2026-04-22 08:59:33 `[run_all] <<< METHOD=sft complete`)
- [x] `pause_after_sft.triggered` 생성 확인 ✅ (08:59:38)
- [x] sft의 최종 체크포인트 검증 ✅ `preserved=[390, 780]` + `live=[1170]`
- [ ] (선택) wandb offline 업로드: `wandb sync /data/flatminima/verl/wandb/offline-run-*`
- [ ] 이전 실험 잔해 `numina-cot-sft-qwen-2.5-math-1.5b/` (19G) 삭제 여부 결정
- [ ] 빈 `numina-cot-sft-adazo-qwen-qwen2-5-1-5b-instruct/` 디렉터리(8K, pause 흔적) 정리 여부
- [ ] **이어서 sft+adazo 실행할지 결정** (Section 9 참고) — pause 상태이므로 명시적 시작 필요
- [ ] (선택) MATH500 평가로 sft 결과 검증

---

_현황 업데이트는 파일 맨 위의 '마지막 업데이트' 줄과 Section 5 '현재 상태'만 갱신하면 됩니다._
