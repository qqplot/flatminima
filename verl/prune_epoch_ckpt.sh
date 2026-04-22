#!/bin/bash
# prune_epoch_ckpt.sh — epoch 경계 체크포인트만 남기고 나머지 global_step_N 은 삭제.
#
# 사용법:
#   bash prune_epoch_ckpt.sh <save_path> [--apply] [--epochs N]
#
#   --apply       실제 삭제 (기본은 dry-run; 뭐가 지워질지 보여주기만 함)
#   --epochs N    전체 epoch 수 (미지정 시 TOTAL_EPOCHS env, 기본 3)
#
# 예:
#   bash prune_epoch_ckpt.sh checkpoints/numina-cot-sft-qwen-2.5-math-1.5b
#       ↑ dry-run 으로 뭐가 지워질지 확인
#
#   bash prune_epoch_ckpt.sh checkpoints/numina-cot-sft-qwen-2.5-math-1.5b --apply
#       ↑ 실제 삭제
#
# 동작:
#   max_step = max(global_step_*) 로 가정. 훈련 완료 직후에만 의미 있음.
#   steps_per_epoch = max_step / epochs
#   유지: epoch 1,2,...,N 에 해당하는 step × 1..N 체크포인트
#   삭제: 그 외 모든 global_step_* 디렉터리
#   wandb_run_id.txt 및 기타 파일은 건드리지 않음.

set -euo pipefail

save_path="${1:-}"
if [ -z "$save_path" ]; then
    echo "usage: $0 <save_path> [--apply] [--epochs N]" >&2
    exit 1
fi
shift

apply=false
epochs="${TOTAL_EPOCHS:-3}"

while (( $# > 0 )); do
    case "$1" in
        --apply)  apply=true ;;
        --epochs) shift; epochs="$1" ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
    shift || true
done

if [ ! -d "$save_path" ]; then
    echo "[prune] not a directory: $save_path" >&2
    exit 1
fi

mapfile -t ckpts < <(ls -d "$save_path"/global_step_* 2>/dev/null | sort -V)
if [ ${#ckpts[@]} -eq 0 ]; then
    echo "[prune] no checkpoints found under $save_path"
    exit 0
fi

max_step=0
for ckpt in "${ckpts[@]}"; do
    step=$(basename "$ckpt" | sed 's/global_step_//')
    if [ "$step" -gt "$max_step" ]; then
        max_step=$step
    fi
done

if (( epochs <= 0 )); then
    echo "[prune] invalid --epochs=$epochs" >&2
    exit 1
fi

if (( max_step % epochs != 0 )); then
    echo "[prune] WARN: max_step=$max_step is not divisible by epochs=$epochs"
    echo "[prune] WARN: epoch 경계 추정이 부정확할 수 있음. floor division 으로 계속 진행."
fi
steps_per_epoch=$(( max_step / epochs ))

declare -A keep=()
for (( e = 1; e <= epochs; e++ )); do
    target=$(( e * steps_per_epoch ))
    keep["$save_path/global_step_$target"]=1
done

echo "[prune] save_path        : $save_path"
echo "[prune] total epochs     : $epochs"
echo "[prune] max_step         : $max_step"
echo "[prune] steps_per_epoch  : $steps_per_epoch"
echo "[prune] epoch boundaries : ${!keep[*]}"
echo ""

to_delete=()
for ckpt in "${ckpts[@]}"; do
    if [ -n "${keep[$ckpt]:-}" ]; then
        echo "  KEEP   $ckpt"
    else
        to_delete+=("$ckpt")
        echo "  DELETE $ckpt"
    fi
done

if [ ${#to_delete[@]} -eq 0 ]; then
    echo ""
    echo "[prune] nothing to delete (모든 체크포인트가 epoch 경계)"
    exit 0
fi

total=0
for ckpt in "${to_delete[@]}"; do
    size=$(du -sb "$ckpt" 2>/dev/null | awk '{print $1}')
    total=$(( total + size ))
done
human=$(numfmt --to=iec "$total" 2>/dev/null || echo "${total} B")

echo ""
echo "[prune] would free: ${human} across ${#to_delete[@]} checkpoint(s)"

if $apply; then
    echo "[prune] deleting..."
    for ckpt in "${to_delete[@]}"; do
        rm -rf "$ckpt"
        echo "  rm -rf $ckpt"
    done
    echo "[prune] done."
else
    echo "[prune] dry-run — 실제 삭제하려면 --apply 추가하여 다시 실행"
fi
