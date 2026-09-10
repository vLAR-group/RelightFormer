#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e 

# ==========================================
# Configuration
# ==========================================
REPO_ID="vLAR/LavalObjaverseDataset"  
LOCAL_DIR="./laval-objaverse-dataset"       
CLEANUP_ARCHIVES=true  # Set to false if you want to keep the .tar.gz files

# ==========================================
# Authentication Note
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

if ! command -v curl &> /dev/null; then
    echo "❌ Error: 'curl' not found. It is required to fetch the file list."
    exit 1
fi

# ==========================================
# Parse Arguments
# ==========================================
SPLIT="${1:-testing}"
SUBSET="${2:-all}"

case "$SPLIT" in
    training)
        if [[ "$SUBSET" == "all" ]]; then
            RENDERED_PATH="rendered/training"
            echo "🎯 Selected split: training (ALL subsets)"
        else
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
        exit 1
        ;;
esac

mkdir -p "$LOCAL_DIR"

echo "📦 Repo:        $REPO_ID"
echo "📂 Destination: $LOCAL_DIR"
echo "========================================"

# ==========================================
# Helper Function: Download metadata folders (no extraction needed)
# ==========================================
download_metadata_folder() {
    local target_path=$1
    local full_local_path="$LOCAL_DIR/$target_path"
    
    echo ""
    echo "🚀 [Processing] /${target_path}/ (Metadata files)"
    mkdir -p "$full_local_path"

    echo "⬇️  Downloading files..."
    hf download "$REPO_ID" --repo-type dataset --include "${target_path}/*" --local-dir "$LOCAL_DIR"
    
    echo "✅ Finished /${target_path}/"
    echo "----------------------------------------"
}

# ==========================================
# Helper Function: Download & Extract One-by-One (for .tar.gz archives)
# ==========================================
download_and_extract_one_by_one() {
    local target_path=$1
    local full_local_path="$LOCAL_DIR/$target_path"
    
    echo ""
    echo "🚀 [Processing] /${target_path}/ (Archives)"
    mkdir -p "$full_local_path"

    # 1. Fetch list of .tar.gz files in this remote directory using Hugging Face API
    echo "🔍 Fetching file list from Hugging Face Hub..."
    local api_response
    api_response=$(curl -s "https://huggingface.co/api/datasets/${REPO_ID}/tree/main/${target_path}")
    
    if echo "$api_response" | grep -q '"error"'; then
        echo "ℹ️ Directory /${target_path}/ not found or empty on the Hub."
        return 0
    fi

    # Extract paths ending in .tar.gz using grep and cut (no 'jq' required)
    local file_list
    file_list=$(echo "$api_response" | grep -o '"path":"[^"]*\.tar\.gz"' | cut -d'"' -f4)

    if [ -z "$file_list" ]; then
        echo "ℹ️ No .tar.gz files found in /${target_path}/"
        return 0
    fi

    local total_files=$(echo "$file_list" | wc -l | tr -d ' ')
    echo "📥 Found $total_files archive(s) to process."

    local current=0
    echo "$file_list" | while read -r remote_file; do
        current=$((current + 1))
        local filename=$(basename "$remote_file")
        local target_dir="${filename%.tar.gz}"
        local local_archive_path="$full_local_path/$filename"

        # Smart Resume: Skip if the extracted directory already exists
        if [ -d "$full_local_path/$target_dir" ]; then
            echo "⏭️  [$current/$total_files] Skipping $filename (Already extracted)"
            # Clean up orphaned archive if it exists from a previous failed run
            if [ "$CLEANUP_ARCHIVES" = true ] && [ -f "$local_archive_path" ]; then
                rm "$local_archive_path"
            fi
            continue
        fi

        echo "⬇️  [$current/$total_files] Downloading: $filename"
        # Download the specific file ONLY
        hf download "$REPO_ID" --repo-type dataset --include "$remote_file" --local-dir "$LOCAL_DIR"

        echo "📦 [$current/$total_files] Extracting: $filename"
        tar -xzf "$local_archive_path" -C "$full_local_path"

        # Cleanup: Delete the .tar.gz immediately to save massive disk space
        if [ "$CLEANUP_ARCHIVES" = true ]; then
            rm "$local_archive_path"
            echo "🧹 [$current/$total_files] Cleaned up archive: $filename"
        fi
        
        echo "✅ [$current/$total_files] Finished: $filename"
        echo "----------------------------------------"
    done
}

# ==========================================
# Execution
# ==========================================

# Step 1: Download metadata folders (pairs and info)
download_metadata_folder "pairs"
download_metadata_folder "info"

# Step 2: Download and extract rendered data (tar.gz archives)
download_and_extract_one_by_one "$RENDERED_PATH"

echo "========================================"
echo "🎉 All downloads and extractions finished successfully!"
echo "📂 Files are located in: $(realpath "$LOCAL_DIR")"