#!/usr/bin/env bash

# Exit on errors, unset variables, and failed commands inside pipelines.
set -Eeuo pipefail

# ==========================================
# Configuration
# ==========================================
REPO_ID="vLAR/LavalObjaverseDataset"
LOCAL_DIR="./laval-objaverse-dataset"
CLEANUP_ARCHIVES=true  # Set to false to keep the .tar.gz files.
PROGRESS_WIDTH=30
STATUS_WIDTH=52

# Hide the nested Hugging Face/tqdm bars so only our overall bar is shown.
export HF_HUB_DISABLE_PROGRESS_BARS=1

# ==========================================
# Authentication Note
# ==========================================
unset HF_TOKEN
unset HUGGING_FACE_HUB_TOKEN

# ==========================================
# Overall Progress State
# ==========================================
OVERALL_TOTAL=0
OVERALL_DONE=0
OVERALL_START_MS=0
TIMED_WORK_MS=0
TIMED_WORK_UNITS=0
CURRENT_UNIT_ACTIVE=false
CURRENT_UNIT_START_MS=0
LAST_NON_TTY_PROGRESS=""
ACTIVE_PID=""
ACTIVE_LOG=""

now_ms() {
    date +%s%3N
}

format_duration() {
    local total_seconds=${1:-0}
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))

    printf '%02d:%02d:%02d' "$hours" "$minutes" "$seconds"
}

shorten_status() {
    local status=$1

    if ((${#status} > STATUS_WIDTH)); then
        printf '%s...' "${status:0:STATUS_WIDTH-3}"
    else
        printf '%s' "$status"
    fi
}

render_overall_progress() {
    local status=${1:-"Working"}
    local current_ms elapsed_ms elapsed_seconds remaining_units
    local average_ms current_unit_elapsed_ms remaining_ms eta_seconds eta_text
    local percent filled empty filled_bar empty_bar line signature

    current_ms=$(now_ms)
    elapsed_ms=$((current_ms - OVERALL_START_MS))
    elapsed_seconds=$((elapsed_ms / 1000))

    if ((OVERALL_TOTAL > 0)); then
        percent=$((OVERALL_DONE * 100 / OVERALL_TOTAL))
        filled=$((OVERALL_DONE * PROGRESS_WIDTH / OVERALL_TOTAL))
    else
        percent=100
        filled=$PROGRESS_WIDTH
    fi

    empty=$((PROGRESS_WIDTH - filled))
    printf -v filled_bar '%*s' "$filled" ''
    printf -v empty_bar '%*s' "$empty" ''
    filled_bar=${filled_bar// /█}
    empty_bar=${empty_bar// /░}

    remaining_units=$((OVERALL_TOTAL - OVERALL_DONE))
    if ((remaining_units <= 0)); then
        eta_text="00:00:00"
    elif ((TIMED_WORK_UNITS > 0)); then
        average_ms=$((TIMED_WORK_MS / TIMED_WORK_UNITS))
        current_unit_elapsed_ms=0

        if [[ "$CURRENT_UNIT_ACTIVE" == true ]]; then
            current_unit_elapsed_ms=$((current_ms - CURRENT_UNIT_START_MS))
        fi

        # The remaining count includes the active item, so subtract time
        # already spent on it from the estimate.
        remaining_ms=$((average_ms * remaining_units - current_unit_elapsed_ms))
        ((remaining_ms < 0)) && remaining_ms=0
        eta_seconds=$(((remaining_ms + 999) / 1000))
        eta_text=$(format_duration "$eta_seconds")
    else
        eta_text="calculating"
    fi

    status=$(shorten_status "$status")
    printf -v line '📊 [%s%s] %3d%% (%d/%d) | Elapsed %s | ETA %s | %s' \
        "$filled_bar" "$empty_bar" "$percent" "$OVERALL_DONE" "$OVERALL_TOTAL" \
        "$(format_duration "$elapsed_seconds")" "$eta_text" "$status"

    if [[ -t 1 ]]; then
        # Return to the start of the same line and erase it before repainting.
        printf '\r\033[2K%s' "$line"
    else
        # Avoid flooding redirected logs: print only when the task/count changes.
        signature="${OVERALL_DONE}|${status}"
        if [[ "$signature" != "$LAST_NON_TTY_PROGRESS" ]]; then
            printf '%s\n' "$line"
            LAST_NON_TTY_PROGRESS=$signature
        fi
    fi
}

finish_progress_line() {
    if [[ -t 1 ]]; then
        printf '\n'
    fi
}

begin_progress_unit() {
    CURRENT_UNIT_ACTIVE=true
    CURRENT_UNIT_START_MS=$(now_ms)
}

advance_overall_progress() {
    local status=$1
    local duration_ms=${2:-}

    OVERALL_DONE=$((OVERALL_DONE + 1))
    CURRENT_UNIT_ACTIVE=false

    # Skipped files advance the bar but do not distort the ETA.
    if [[ -n "$duration_ms" ]]; then
        TIMED_WORK_MS=$((TIMED_WORK_MS + duration_ms))
        TIMED_WORK_UNITS=$((TIMED_WORK_UNITS + 1))
    fi

    render_overall_progress "$status"
}

run_with_live_progress() {
    local status=$1
    shift

    local exit_code=0
    ACTIVE_LOG=$(mktemp)

    "$@" >"$ACTIVE_LOG" 2>&1 &
    ACTIVE_PID=$!

    while kill -0 "$ACTIVE_PID" 2>/dev/null; do
        render_overall_progress "$status"
        sleep 1
    done

    if wait "$ACTIVE_PID"; then
        exit_code=0
    else
        exit_code=$?
    fi
    ACTIVE_PID=""

    if ((exit_code != 0)); then
        if [[ -t 1 ]]; then
            printf '\r\033[2K'
        fi
        echo "❌ Failed while ${status}." >&2
        cat "$ACTIVE_LOG" >&2
        rm -f "$ACTIVE_LOG"
        ACTIVE_LOG=""
        return "$exit_code"
    fi

    rm -f "$ACTIVE_LOG"
    ACTIVE_LOG=""
    render_overall_progress "$status"
}

cleanup_on_exit() {
    local exit_code=$?

    if [[ -n "$ACTIVE_PID" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
        kill "$ACTIVE_PID" 2>/dev/null || true
        wait "$ACTIVE_PID" 2>/dev/null || true
    fi

    [[ -n "$ACTIVE_LOG" ]] && rm -f "$ACTIVE_LOG"

    if ((exit_code != 0)) && [[ -t 1 ]]; then
        printf '\n'
    fi
}
trap cleanup_on_exit EXIT INT TERM

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
echo "🔍 Fetching archive list from Hugging Face Hub..."

# ==========================================
# Discover Archives and Initialize Progress
# ==========================================
API_RESPONSE=$(curl -sS "https://huggingface.co/api/datasets/${REPO_ID}/tree/main/${RENDERED_PATH}")

if grep -q '"error"' <<< "$API_RESPONSE"; then
    echo "ℹ️ Directory /${RENDERED_PATH}/ was not found or is empty on the Hub."
    ARCHIVE_FILES=()
else
    mapfile -t ARCHIVE_FILES < <(
        grep -oE '"path"[[:space:]]*:[[:space:]]*"[^"]+\.tar\.gz"' <<< "$API_RESPONSE" \
            | sed -E 's/^"path"[[:space:]]*:[[:space:]]*"([^"]+)"$/\1/' \
            || true
    )
fi

ARCHIVE_TOTAL=${#ARCHIVE_FILES[@]}
OVERALL_TOTAL=$((2 + ARCHIVE_TOTAL))  # Two metadata folders plus all archives.
OVERALL_START_MS=$(now_ms)

echo "📥 Work queue: 2 metadata folders + ${ARCHIVE_TOTAL} archive(s)"
render_overall_progress "Ready"

# ==========================================
# Download Metadata Folders
# ==========================================
download_metadata_folder() {
    local target_path=$1
    local full_local_path="$LOCAL_DIR/$target_path"
    local step_start_ms step_duration_ms

    mkdir -p "$full_local_path"
    begin_progress_unit
    step_start_ms=$CURRENT_UNIT_START_MS

    run_with_live_progress "Downloading metadata: /${target_path}/" \
        hf download "$REPO_ID" \
            --repo-type dataset \
            --include "${target_path}/*" \
            --local-dir "$LOCAL_DIR"

    step_duration_ms=$(($(now_ms) - step_start_ms))
    advance_overall_progress "Completed metadata: /${target_path}/" "$step_duration_ms"
}

# ==========================================
# Download and Extract Archives One by One
# ==========================================
download_and_extract_one_by_one() {
    local target_path=$1
    local full_local_path="$LOCAL_DIR/$target_path"
    local current=0
    local remote_file filename target_dir local_archive_path
    local step_start_ms step_duration_ms

    mkdir -p "$full_local_path"

    for remote_file in "${ARCHIVE_FILES[@]}"; do
        current=$((current + 1))
        filename=$(basename "$remote_file")
        target_dir="${filename%.tar.gz}"
        local_archive_path="$full_local_path/$filename"

        # Smart Resume: skip archives whose extracted directory already exists.
        if [[ -d "$full_local_path/$target_dir" ]]; then
            if [[ "$CLEANUP_ARCHIVES" == true && -f "$local_archive_path" ]]; then
                rm -f "$local_archive_path"
            fi

            advance_overall_progress "Skipped existing: $filename"
            continue
        fi

        begin_progress_unit
        step_start_ms=$CURRENT_UNIT_START_MS

        run_with_live_progress "Downloading [$current/$ARCHIVE_TOTAL]: $filename" \
            hf download "$REPO_ID" \
                --repo-type dataset \
                --include "$remote_file" \
                --local-dir "$LOCAL_DIR"

        run_with_live_progress "Extracting [$current/$ARCHIVE_TOTAL]: $filename" \
            tar -xzf "$local_archive_path" -C "$full_local_path"

        if [[ "$CLEANUP_ARCHIVES" == true ]]; then
            rm -f "$local_archive_path"
        fi

        step_duration_ms=$(($(now_ms) - step_start_ms))
        advance_overall_progress "Completed [$current/$ARCHIVE_TOTAL]: $filename" "$step_duration_ms"
    done
}

# ==========================================
# Execution
# ==========================================
download_metadata_folder "pairs"
download_metadata_folder "info"
download_and_extract_one_by_one "$RENDERED_PATH"

render_overall_progress "Complete"
finish_progress_line

echo "========================================"
echo "🎉 All downloads and extractions finished successfully!"
echo "⏱️  Total elapsed: $(format_duration "$(($(now_ms) - OVERALL_START_MS) / 1000)")"
echo "📂 Files are located in: $(realpath "$LOCAL_DIR")"
