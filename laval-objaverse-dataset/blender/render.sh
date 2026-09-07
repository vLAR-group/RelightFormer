#!/bin/bash

# ==============================================================================
# Default Arguments (Matching distribute-rendering.py Args dataclass)
# ==============================================================================
BLENDER_PATH=""
SPLITS="training validation testing"
WORKERS_PER_GPU=16
GPUS="0 1 2 3 4 5 6 7 8 9"  # 10 GPUs by default
SKIP_EXIST="true"
LOG_TO_WANDB="true"
SEED=0
UPLOAD="true"
DOWNLOAD="false"
UPLOAD_RETRIES=3
CREDENTIALS_FILE="./credentials.json"

# ==============================================================================
# Usage Function
# ==============================================================================
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  -b, --blender PATH        Path to Blender executable (e.g., ./blender-4.3.2-linux-x64/blender)"
    echo ""
    echo "Optional:"
    echo "  -s, --split SPLIT         Dataset split(s) to render: training, validation, testing, or 'all'. Default: all"
    echo "  -w, --workers INT         Number of workers per GPU. Default: 16"
    echo "  -g, --gpus LIST           Space-separated list of GPU indices. Default: 0 1 2 3 4 5 6 7 8 9"
    echo "  --no-skip-exist           Do not skip existing renders (overrides default skip_exist=true)"
    echo "  --no-wandb                Disable logging progress to wandb"
    echo "  --seed INT                Random seed. Default: 0"
    echo "  --no-upload               Disable uploading rendered subsets to remote server"
    echo "  --download                Sync rendered data from remote BEFORE rendering"
    echo "  -r, --retries INT         Number of retry attempts for failed uploads. Default: 3"
    echo "  -c, --credentials PATH    Path to credentials JSON file. Default: ./credentials.json"
    echo "  -h, --help                Show this help message"
    exit 1
}

# ==============================================================================
# Parse Command-Line Arguments
# ==============================================================================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -b|--blender) BLENDER_PATH="$2"; shift ;;
        -s|--split) 
            if [[ "$2" == "all" ]]; then
                SPLITS="training validation testing"
            else
                SPLITS="$2"
            fi
            shift ;;
        -w|--workers) WORKERS_PER_GPU="$2"; shift ;;
        -g|--gpus) GPUS="$2"; shift ;;
        --no-skip-exist) SKIP_EXIST="false" ;;
        --no-wandb) LOG_TO_WANDB="false" ;;
        --seed) SEED="$2"; shift ;;
        --no-upload) UPLOAD="false" ;;
        --download) DOWNLOAD="true" ;;
        -r|--retries) UPLOAD_RETRIES="$2"; shift ;;
        -c|--credentials) CREDENTIALS_FILE="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "❌ Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# ==============================================================================
# Validation
# ==============================================================================
if [[ -z "$BLENDER_PATH" ]]; then
    echo "❌ Error: Blender path is required. Use -b or --blender."
    usage
fi

if [[ ! -f "$BLENDER_PATH" ]]; then
    echo "❌ Error: Blender executable not found at: $BLENDER_PATH"
    exit 1
fi

# ==============================================================================
# Execution
# ==============================================================================
echo "=========================================================="
echo "🚀 Starting Distributed Rendering Pipeline"
echo "----------------------------------------------------------"
echo "Blender Path   : $BLENDER_PATH"
echo "Splits         : $SPLITS"
echo "Workers / GPU  : $WORKERS_PER_GPU"
echo "GPUs           : $GPUS"
echo "Skip Existing  : $SKIP_EXIST"
echo "Log to WandB   : $LOG_TO_WANDB"
echo "Seed           : $SEED"
echo "Upload         : $UPLOAD"
echo "Download First : $DOWNLOAD"
echo "Retries        : $UPLOAD_RETRIES"
echo "Credentials    : $CREDENTIALS_FILE"
echo "=========================================================="

# Loop through each requested split and run the python script
for split in $SPLITS; do
    echo ""
    echo "▶️  Launching rendering for split: [$split]"
    
    # Build the command dynamically
    CMD=(
        python distribute-rendering.py
        --blender "$BLENDER_PATH"
        --split "$split"
        --workers_per_gpu "$WORKERS_PER_GPU"
        --gpus $GPUS
        --skip_exist "$SKIP_EXIST"
        --log_to_wandb "$LOG_TO_WANDB"
        --seed "$SEED"
        --upload "$UPLOAD"
        --download "$DOWNLOAD"
        --upload_retries "$UPLOAD_RETRIES"
        --credentials_file "$CREDENTIALS_FILE"
    )
    
    # Execute the command
    "${CMD[@]}"
    
    # Check if the command succeeded
    if [ $? -ne 0 ]; then
        echo "❌ Error: Rendering failed for split '$split'."
        echo "💡 Tip: Check the logs above or adjust your GPU/worker allocation."
        exit 1
    fi
done

echo ""
echo "✅ All specified splits completed successfully!"