#!/bin/bash
cd "$(dirname "$0")" || exit 1
while true; do
    echo "[Loop] Submitting distillation job..."
    JID=$(sbatch run_distillation.sbatch | awk '{print $4}')
    
    if [ -z "$JID" ]; then
        echo "[Loop] sbatch failed. Retrying in 60s..."
        sleep 60
        continue
    fi
    
    echo "[Loop] Job $JID submitted. Waiting for it to finish..."
    
    while squeue -j $JID 2>/dev/null | grep -q $JID; do
        sleep 60
    done
    
    echo "[Loop] Job $JID finished. Checking if there are entities left..."
    sleep 30
    
    # Check the latest python output log for "Por processar agora: 0"
    LATEST_LOG=$(ls -t distill_*.out 2>/dev/null | head -n 1)
    if [ -n "$LATEST_LOG" ]; then
        if grep -q "Por processar agora: 0" "$LATEST_LOG"; then
            echo "[Loop] All entities processed! Exiting loop."
            break
        fi

        REMAINING=$(grep -oP "Por processar agora: \K\d+" "$LATEST_LOG" | tail -n 1)
        if [ "$REMAINING" == "$PREV_REMAINING" ] && [ -n "$REMAINING" ]; then
            echo "[Loop] Deadloop detected! Remaining entities stuck at $REMAINING. Breaking loop to advance pipeline."
            break
        fi
        PREV_REMAINING="$REMAINING"
    fi
    
    # Fallback checking 0 output to avoid infinite loops if it crashed immediately
    if [ -n "$LATEST_LOG" ] && grep -qE "Traceback|\[FATAL\]" "$LATEST_LOG"; then
        echo "[Loop] Error found in the Python script. Exiting to avoid infinite crash loops."
        break
    fi
done
