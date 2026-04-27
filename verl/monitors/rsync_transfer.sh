#!/bin/bash
# Transfer epoch checkpoints (sft + math) to login2:/shared/s2/lab01/qqplot/flatminima_ckpt
# Option 1: full ckpts (model + trainer_state.pt ~ 9.1G each × 4 = ~36G)
set -u
LOG="${1:-/data/flatminima/verl/logs/rsync_transfer.log}"
DEST_HOST="kyubyungchae@147.47.200.22"
DEST_PATH="/shared/s2/lab01/qqplot/flatminima_ckpt"
SRC="/data/flatminima/verl/checkpoints"
exec >> "$LOG" 2>&1

echo ""
echo "[$(date -Iseconds)] ========================================"
echo "[$(date -Iseconds)] rsync_transfer started pid=$$"
echo "[$(date -Iseconds)] DEST=$DEST_HOST:$DEST_PATH"
echo "[$(date -Iseconds)] SRC=$SRC"

# remote dir 준비
ssh -o BatchMode=yes "$DEST_HOST" \
    "mkdir -p $DEST_PATH/numina-cot-sft-qwen-qwen2-5-1-5b-instruct \
            $DEST_PATH/numina-cot-sft-qwen-2.5-math-1.5b"
echo "[$(date -Iseconds)] remote dirs ready"

# 1) sft (epoch 1,2,3) 3 ckpts
echo "[$(date -Iseconds)] === PHASE 1: sft (step 390, 780, 1170) ==="
rsync -ah --partial --append-verify --info=progress2,stats2 \
      -e "ssh -o BatchMode=yes" \
    "$SRC/numina-cot-sft-qwen-qwen2-5-1-5b-instruct/global_step_390" \
    "$SRC/numina-cot-sft-qwen-qwen2-5-1-5b-instruct/global_step_780" \
    "$SRC/numina-cot-sft-qwen-qwen2-5-1-5b-instruct/global_step_1170" \
    "$DEST_HOST:$DEST_PATH/numina-cot-sft-qwen-qwen2-5-1-5b-instruct/"
SFTRC=$?
echo "[$(date -Iseconds)] PHASE 1 done exit=$SFTRC"

# 2) math-1.5b (epoch 1)
echo "[$(date -Iseconds)] === PHASE 2: math-1.5b (step 390) ==="
rsync -ah --partial --append-verify --info=progress2,stats2 \
      -e "ssh -o BatchMode=yes" \
    "$SRC/numina-cot-sft-qwen-2.5-math-1.5b/global_step_390" \
    "$DEST_HOST:$DEST_PATH/numina-cot-sft-qwen-2.5-math-1.5b/"
MATHRC=$?
echo "[$(date -Iseconds)] PHASE 2 done exit=$MATHRC"

echo "[$(date -Iseconds)] ALL DONE. sft=$SFTRC math=$MATHRC"
echo "[$(date -Iseconds)] ========================================"
