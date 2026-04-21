#!/bin/bash
# ROCm GPU junction 온도를 주기적으로 폴링해서 임계값 초과 시 alert 파일에 기록.
#   logs/rocm_temp.log         - 모든 샘플 (max junction temp)
#   logs/rocm_temp_alerts.log  - threshold 초과한 GPU/온도 (여기에 찍히면 '알림')

INTERVAL=${INTERVAL:-30}         # polling 주기 (sec)
THRESHOLD=${THRESHOLD:-90}       # alert 임계 (°C)
LOG=${LOG:-logs/rocm_temp.log}
ALERT_LOG=${ALERT_LOG:-logs/rocm_temp_alerts.log}

mkdir -p "$(dirname "$LOG")"
echo "[rocm_temp_monitor] started; interval=${INTERVAL}s threshold=${THRESHOLD}C log=$LOG alerts=$ALERT_LOG" | tee -a "$LOG"

while :; do
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    output=$(rocm-smi --showtemp 2>&1)

    max_temp=0
    hot=""
    while IFS= read -r line; do
        # 'GPU[N]   : Temperature (Sensor junction) (C): <value>' 만 파싱
        case "$line" in
            *"Sensor junction"*)
                gpu=$(printf '%s\n' "$line" | grep -oE 'GPU\[[0-9]+\]')
                temp=$(printf '%s\n' "$line" | awk -F': ' 'END{print $NF}')
                temp_int=${temp%.*}
                # 비교
                if [ "$temp_int" -gt "$max_temp" ] 2>/dev/null; then
                    max_temp=$temp_int
                fi
                if [ "$temp_int" -gt "$THRESHOLD" ] 2>/dev/null; then
                    hot="$hot $gpu=${temp}C"
                fi
                ;;
        esac
    done <<< "$output"

    echo "[$ts] max_junction=${max_temp}C" >> "$LOG"

    if [ -n "$hot" ]; then
        echo "[$ts] ALERT threshold=${THRESHOLD}C →$hot" | tee -a "$ALERT_LOG"
    fi

    sleep "$INTERVAL"
done
