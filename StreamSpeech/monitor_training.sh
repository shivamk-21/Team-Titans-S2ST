#!/bin/bash

# Training monitor script
# Monitors the progress of all three language trainings

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check training status
check_training_status() {
    local lang=$1
    local lang_name=$2
    local model="streamspeech.offline-s2st.${lang}-en"
    local checkpoint_dir="checkpoints/$model"
    
    echo -e "\n${BLUE}=== $lang_name Training Status ===${NC}"
    
    if [ -d "$checkpoint_dir" ]; then
        # Count checkpoints
        local checkpoint_count=$(ls -1 "$checkpoint_dir"/checkpoint*.pt 2>/dev/null | wc -l)
        local total_size=$(du -sh "$checkpoint_dir" 2>/dev/null | cut -f1)
        
        # Get latest checkpoint
        local latest_checkpoint=$(ls -t "$checkpoint_dir"/checkpoint[0-9]*.pt 2>/dev/null | head -1)
        
        if [ -n "$latest_checkpoint" ]; then
            local latest_epoch=$(basename "$latest_checkpoint" | sed 's/checkpoint\([0-9]*\)\.pt/\1/')
            local checkpoint_time=$(stat -c %y "$latest_checkpoint" 2>/dev/null | cut -d' ' -f1-2)
            
            echo -e "  📁 Checkpoint Directory: $checkpoint_dir"
            echo -e "  📊 Total Checkpoints: $checkpoint_count"
            echo -e "  📈 Latest Epoch: $latest_epoch"
            echo -e "  ⏰ Last Updated: $checkpoint_time"
            echo -e "  💾 Total Size: $total_size"
            
            # Check if training is active
            if pgrep -f "fairseq-train.*${lang}-en" > /dev/null; then
                echo -e "  🟢 Status: ${GREEN}TRAINING ACTIVE${NC}"
            else
                echo -e "  🟡 Status: ${YELLOW}NOT RUNNING${NC}"
            fi
        else
            echo -e "  📁 Checkpoint Directory: $checkpoint_dir"
            echo -e "  📊 Total Checkpoints: $checkpoint_count"
            echo -e "  📈 Latest Epoch: No epoch checkpoints found"
            echo -e "  🔴 Status: ${RED}NO PROGRESS${NC}"
        fi
    else
        echo -e "  📁 Checkpoint Directory: $checkpoint_dir"
        echo -e "  🔴 Status: ${RED}NOT STARTED${NC}"
    fi
}

# Function to show recent log activity
show_recent_logs() {
    local lang=$1
    local lang_name=$2
    
    echo -e "\n${BLUE}=== Recent $lang_name Log Activity ===${NC}"
    
    # Find most recent log file for this language
    local latest_log=$(ls -t training_logs/train_${lang}_*epochs_*.log 2>/dev/null | head -1)
    
    if [ -n "$latest_log" ]; then
        echo -e "  📝 Log File: $latest_log"
        echo -e "  📄 Last 5 lines:"
        tail -5 "$latest_log" 2>/dev/null | sed 's/^/    /'
    else
        echo -e "  🔴 No log files found for $lang_name"
    fi
}

# Function to check GPU status
check_gpu_status() {
    echo -e "\n${BLUE}=== GPU Status ===${NC}"
    
    if command -v nvidia-smi &> /dev/null; then
        echo -e "  🖥️  GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | \
        awk '{printf "    Name: %s\n    Memory: %d MB total, %d MB used, %d MB free\n    GPU Utilization: %d%%\n", $1, $2, $3, $4, $5}'
        
        echo -e "\n  🔍 Active Training Processes:"
        nvidia-smi pmon -c 1 2>/dev/null | grep -E "(fairseq|python)" | head -5 | sed 's/^/    /'
    else
        echo -e "  ❌ nvidia-smi not available"
    fi
}

# Function to show tensorboard status
check_tensorboard() {
    echo -e "\n${BLUE}=== Tensorboard Status ===${NC}"
    
    if [ -d "tensorboard_logs" ]; then
        echo -e "  📊 Tensorboard Log Directory: tensorboard_logs"
        for lang in hi ma mr; do
            local tb_dir="tensorboard_logs/${lang}_200epochs"
            if [ -d "$tb_dir" ]; then
                local event_count=$(find "$tb_dir" -name "events.out.tfevents.*" | wc -l)
                echo -e "    ${lang}: $event_count event files"
            else
                echo -e "    ${lang}: No logs yet"
            fi
        done
        
        # Check if tensorboard is running
        if pgrep -f "tensorboard" > /dev/null; then
            echo -e "  🟢 Tensorboard Status: ${GREEN}RUNNING${NC}"
            echo -e "  🌐 Access at: http://localhost:6006"
        else
            echo -e "  🔴 Tensorboard Status: ${RED}NOT RUNNING${NC}"
            echo -e "  💡 Start with: tensorboard --logdir=tensorboard_logs"
        fi
    else
        echo -e "  📊 No tensorboard logs directory found"
    fi
}

# Main monitoring function
main() {
    clear
    echo -e "${GREEN}StreamSpeech Training Monitor${NC}"
    echo -e "${GREEN}============================${NC}"
    echo -e "Last updated: $(date)"
    
    # Check status for all languages
    check_training_status "hi" "Hindi"
    check_training_status "ma" "Malayalam"  
    check_training_status "mr" "Marathi"
    
    # Show GPU status
    check_gpu_status
    
    # Show tensorboard status
    check_tensorboard
    
    # Show recent logs if requested
    if [ "$1" = "--logs" ]; then
        show_recent_logs "hi" "Hindi"
        show_recent_logs "ma" "Malayalam"
        show_recent_logs "mr" "Marathi"
    fi
    
    echo -e "\n${BLUE}=== Commands ===${NC}"
    echo -e "  Monitor continuously: watch -n 30 ./monitor_training.sh"
    echo -e "  Show recent logs: ./monitor_training.sh --logs"
    echo -e "  Start all training: ./train_all_languages_200epochs.sh"
    echo -e "  Start tensorboard: tensorboard --logdir=tensorboard_logs"
    
    echo -e "\n${BLUE}=== Individual Training Scripts ===${NC}"
    echo -e "  Hindi: ./train_hi_200epochs.sh"
    echo -e "  Malayalam: ./train_ma_200epochs.sh"
    echo -e "  Marathi: ./train_mr_200epochs.sh"
}

# Run main function
main "$@"