#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e 

# ==========================================
# Configuration
# ==========================================
REPO_ID="vLAR/LavalObjaverseDataset"  
LOCAL_DIR="./laval-objaverse-dataset"       

# ==========================================
# Skip Login / Force Anonymous Access
# ==========================================
unset HF_TOKEN
unset HUGGING_FACE_HUB_TOKEN

# ==========================================
# Pre-flight Checks
# ==========================================
if ! command -v hf &> /dev/null; then
    echo "❌ Error: 'hf' CLI not found."
    echo "💡 Install it via: pip install -U 'huggingface_hub[cli]'"
    exit 1
fi

# ==========================================
# Parse Arguments
# ==========================================
# $1 = Split (training, validation, testing, all). Default: testing
# $2 = Subset ID (only for training). Default: all
SPLIT="${1:-testing}"
SUBSET="${2:-all}"

case "$SPLIT" in
    training)
        if [[ "$SUBSET" == "all" ]]; then
            RENDERED_PATH="rendered/training"
            echo "🎯 Selected split: training (ALL subsets)"
        else
            # Normalize subset name: if it doesn't start with "subset_", add it
            if [[ "$SUBSET" != subset_* ]]; then
                SUBSET="subset_${SUBSET}"
            fi
            RENDERED_PATH="rendered/training/${SUBSET}"
            echo "🎯 Selected split: training, subset: ${SUBSET}"
        fi
        ;;
    validation|testing)
        RENDERED_PATH="rendered/${SPLIT}"
        echo "🎯 Selected split: ${SPLIT}"
        ;;
    all)
        RENDERED_PATH="rendered"
        echo "🎯 Selected split: ALL (training + validation + testing)"
        ;;
    *)
        echo "❌ Invalid split: '${SPLIT}'"
        echo "💡 Usage: $0 [training|validation|testing|all] [subset_id]"
        echo "   Example: $0 training 5       (Downloads only subset_5)"
        echo "   Example: $0 training         (Downloads all training subsets)"
        exit 1
        ;;
esac

mkdir -p "$LOCAL_DIR"

echo "📦 Repo:        $REPO_ID"
echo "📂 Destination: $LOCAL_DIR"
echo "========================================"

# ==========================================
# Step 1: Download the /pairs/ folder
# ==========================================
echo "🚀 [Step 1/3] Downloading /pairs/ folder..."
hf download "$REPO_ID" "pairs/*" --local-dir "$LOCAL_DIR"
echo "✅ Step 1 completed."
echo "----------------------------------------"

# ==========================================
# Step 2: Download the /rendered/{path} folder
# ==========================================
echo "🚀 [Step 2/3] Downloading /${RENDERED_PATH}/ folder..."
hf download "$REPO_ID" "${RENDERED_PATH}/*" --local-dir "$LOCAL_DIR"
echo "✅ Step 2 completed."
echo "----------------------------------------"

# ==========================================
# Step 3: Synchronously Extract all .tar.gz files
# ==========================================
echo "🚀 [Step 3/3] Finding and synchronously extracting .tar.gz files..."

file_count=0
# find -print0 and read -d '' safely handles filenames with spaces/special characters
while IFS= read -r -d '' archive; do
    file_count=$((file_count + 1))
    echo "📦 Extracting [$file_count]: $(basename "$archive")"
    
    # Extract the archive into its own parent directory
    tar -xzf "$archive" -C "$(dirname "$archive")"
    
    # 💡 OPTIONAL: Uncomment the line below to delete the .tar.gz file 
    # after successful extraction to save massive amounts of disk space.
    # rm "$archive"
    
done < <(find "$LOCAL_DIR" -type f -name "*.tar.gz" -print0)

if [ "$file_count" -eq 0 ]; then
    echo "ℹ️ No .tar.gz files found to extract."
else
    echo "✅ Successfully extracted $file_count archive(s)."
fi

echo "========================================"
echo "🎉 All downloads and extractions finished successfully!"
echo "📂 Files are located in: $(realpath "$LOCAL_DIR")"