#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e 

# ==========================================
# Configuration
# ==========================================
REPO_ID="vLAR/LavalObjaverseDataset"  # <-- Change to the actual repo ID
LOCAL_DIR="./laval-objaverse-dataset"       # Local destination folder

# ==========================================
# Skip Login / Force Anonymous Access
# ==========================================
# Unset token variables to guarantee no login prompts or cached tokens interfere
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
# Parse Split Argument
# ==========================================
# Default to "testing" if no argument is provided
SPLIT="${1:-testing}" 

case "$SPLIT" in
    training|validation|testing)
        RENDERED_PATH="rendered/${SPLIT}"
        echo "🎯 Selected split: ${SPLIT}"
        ;;
    all)
        RENDERED_PATH="rendered"
        echo "🎯 Selected split: ALL (training + validation + testing)"
        ;;
    *)
        echo "❌ Invalid split: '${SPLIT}'"
        echo "💡 Usage: $0 [training|validation|testing|all]"
        exit 1
        ;;
esac

mkdir -p "$LOCAL_DIR"

echo "📦 Repo:        $REPO_ID"
echo "📂 Destination: $LOCAL_DIR"
echo "========================================"

# ==========================================
# Step 1: Download the /pair/ folder
# ==========================================
echo "🚀 [Step 1/2] Downloading /pair/ folder..."
# Using 'pair/*' grabs all files inside the folder. 
# (If 'pair' contains nested subfolders you also need, change to 'pair/**/*')
hf download "$REPO_ID" "pairs/*" --local-dir "$LOCAL_DIR"
echo "✅ Step 1 completed."
echo "----------------------------------------"

# ==========================================
# Step 2: Download the /rendered/{split} folder
# ==========================================
echo "🚀 [Step 2/2] Downloading /${RENDERED_PATH}/ folder..."
if [ "$SPLIT" == "all" ]; then
    # If 'all', grab everything inside rendered/
    hf download "$REPO_ID" "rendered/*" --local-dir "$LOCAL_DIR"
else
    # Otherwise, grab the specific split
    hf download "$REPO_ID" "${RENDERED_PATH}/*" --local-dir "$LOCAL_DIR"
fi
echo "✅ Step 2 completed."

echo "========================================"
echo "🎉 All downloads finished successfully!"
echo "📂 Files are located in: $(realpath "$LOCAL_DIR")"