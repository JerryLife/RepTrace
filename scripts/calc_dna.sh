#!/usr/bin/env bash

# run_dna_pipeline.sh - Unified DNA Calculation Script
# Combines parallel processing, GPU management, and comprehensive configuration

# Default configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_MODULE="reptrace.experiments.compute_dna"
LOG_DIR="$PROJECT_ROOT/log"
OUT_DIR="$PROJECT_ROOT/out"
LLM_LIST_FILE="$PROJECT_ROOT/configs/llm_list.txt"

# Available datasets (short IDs from compute_dna.py)
DEFAULT_DATASETS="rand"

# Default parameters
DATASETS="$DEFAULT_DATASETS"
GPUS="0,1,2,3"
MAX_TASKS_PER_GPU=1
DNA_DIM=128
EXTRACTOR_TYPE="embedding"
MAX_SAMPLES=100
EMBEDDING_MERGE="concat"
REDUCTION_METHOD="random_projection"
DRY_RUN=false
SKIP_EXISTING=true
REVERSE_ORDER=false
SKIP_CHAT_TEMPLATE=true

# Parse command line arguments
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Unified DNA calculation script with parallel processing and multi-GPU support.
Combines model loading, dataset processing, and parallel execution in one script.

OPTIONS:
    --gpus GPUS               Comma-separated GPU IDs (default: 0,1,2,3)
    --max-tasks-per-gpu N     Maximum concurrent tasks per GPU (default: 1)
    --dna-dim DIM            DNA dimension (default: 128)
    --extractor TYPE         Extractor type [embedding] (default: embedding)
    --max-samples N          Maximum samples per dataset (default: 100)
    --embedding-merge METHOD Method to merge embeddings [sum|max|mean|concat] (default: concat)
    --reduction-method NAME  Dimensionality reduction [pca|svd|random_projection] (default: random_projection)
    --llm-list FILE          Path to LLM list file (default: configs/llm_list.txt)
    --datasets LIST          Comma-separated dataset IDs (default: rand)
    --no-skip-existing       Disable skipping existing valid model-dataset pairs
    --reverse                Process models in reverse order (last to first)
    --dry-run                Run in dry run mode (simulate GPU usage)
    --skip-chat-template     Skip applying chat templates for chat models (saves to out_no_chat_template/)
    -h, --help               Show this help message

EXAMPLES:
    # Use all defaults (all models, dataset=rand, GPUs 0-3)
    $0

    # Use specific GPUs with 2 tasks per GPU
    $0 --gpus "0,1" --max-tasks-per-gpu 2

    # Use only specific datasets
    $0 --datasets "syn,squad,mmlu"

    # Test parallel execution with dry run
    $0 --dry-run --gpus "1,2" --datasets "syn" --max-tasks-per-gpu 2
    
    # Process models in reverse order (useful for resuming interrupted runs)
    $0 --reverse --datasets "squad,mmlu"
    
DATASET IDs:
    syn     - Synthetic probes (generated using probe_generator.py)
    squad   - SQuAD dataset  
    cqa     - CommonsenseQA dataset
    hs      - HellaSwag dataset
    wg      - WinoGrande dataset
    arc     - ARC Challenge dataset
    mmlu    - MMLU dataset
    embed   - EmbedLLM dataset
    rand    - Random word dataset (600 samples, 100 words each)
    
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --max-tasks-per-gpu)
            MAX_TASKS_PER_GPU="$2"
            shift 2
            ;;
        --dna-dim)
            DNA_DIM="$2"
            shift 2
            ;;
        --extractor)
            EXTRACTOR_TYPE="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --embedding-merge)
            EMBEDDING_MERGE="$2"
            shift 2
            ;;
        --reduction-method)
            REDUCTION_METHOD="$2"
            shift 2
            ;;
        --llm-list)
            LLM_LIST_FILE="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --no-skip-existing)
            SKIP_EXISTING=false
            shift
            ;;
        --reverse)
            REVERSE_ORDER=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-chat-template)
            SKIP_CHAT_TEMPLATE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# Enforce supported extractor type
if [[ "$EXTRACTOR_TYPE" != "embedding" ]]; then
    echo "Error: only the 'embedding' extractor is supported." >&2
    exit 1
fi

# Note: Output directory is determined per-model based on chat model status
# Default OUT_DIR is used for non-chat models; chat models use out_no_chat_template/ when SKIP_CHAT_TEMPLATE=true

# Validate inputs
if [[ ! -f "$LLM_LIST_FILE" ]]; then
    echo "Error: LLM list file not found: $LLM_LIST_FILE" >&2
    exit 1
fi

# Derive metadata filename from list filename (replace 'list' -> 'metadata', set .json)
LLM_LIST_DIR="$(dirname "$LLM_LIST_FILE")"
LLM_LIST_BASE="$(basename "$LLM_LIST_FILE")"
LLM_LIST_STEM="${LLM_LIST_BASE%.*}"

PRIMARY_META_BASE_STEM="$(echo "$LLM_LIST_STEM" | sed 's/list/metadata/g')"
PRIMARY_META_FILE="$LLM_LIST_DIR/${PRIMARY_META_BASE_STEM}.json"

# Backward-compat candidate (older convention): replace 'llm_list' with 'llm_metadata'
LEGACY_META_BASE_STEM="$(echo "$LLM_LIST_STEM" | sed 's/llm_list/llm_metadata/g')"
LEGACY_META_FILE="$LLM_LIST_DIR/${LEGACY_META_BASE_STEM}.json"

# Generic defaults
DIR_GENERIC_META_FILE="$LLM_LIST_DIR/llm_metadata.json"
ROOT_GENERIC_META_FILE="$PROJECT_ROOT/configs/llm_metadata.json"

# Find best metadata file
METADATA_FILE=""
for candidate in "$PRIMARY_META_FILE" "$LEGACY_META_FILE" "$DIR_GENERIC_META_FILE" "$ROOT_GENERIC_META_FILE"; do
    if [[ -f "$candidate" ]]; then
        METADATA_FILE="$candidate"
        break
    fi
done

# If none found, try to pick a metadata-like JSON in the same dir
if [[ -z "$METADATA_FILE" ]]; then
    match=$(ls "$LLM_LIST_DIR"/*metadata*.json 2>/dev/null | head -n1 || true)
    if [[ -n "$match" && -f "$match" ]]; then
        METADATA_FILE="$match"
    else
        # Fall back to primary derived path
        METADATA_FILE="$PRIMARY_META_FILE"
    fi
fi

# Generate metadata at the chosen location if it still doesn't exist
if [[ ! -f "$METADATA_FILE" ]]; then
    echo "Metadata not found. Generating: $METADATA_FILE"

    # Prefer hf_token.txt, fallback to hf_token_ziyang.txt if present
    TOKEN_ARG=()
    if [[ -s "$PROJECT_ROOT/hf_token.txt" ]]; then
        TOKEN_ARG=(--token "$(tr -d '\r' < "$PROJECT_ROOT/hf_token.txt" | xargs)")
    elif [[ -s "$PROJECT_ROOT/hf_token_ziyang.txt" ]]; then
        TOKEN_ARG=(--token "$(tr -d '\r' < "$PROJECT_ROOT/hf_token_ziyang.txt" | xargs)")
    fi

    (
        cd "$PROJECT_ROOT" || exit 1
        # Activate virtual environment if present
        if [[ -f ".venv/bin/activate" ]]; then
            # shellcheck disable=SC1091
            source .venv/bin/activate
            echo ">>> Using Python: $(which python)"
        elif [[ -n "${VIRTUAL_ENV}" ]]; then
            echo ">>> Virtual environment already active: ${VIRTUAL_ENV}"
        else
            echo ">>> Warning: No virtual environment detected for metadata generation"
        fi

        python -m reptrace.models.get_model_metadata \
            --list-file "$LLM_LIST_FILE" \
            --output-file "$METADATA_FILE" \
            "${TOKEN_ARG[@]}"
    ) || {
        echo "Error: Failed to generate metadata at $METADATA_FILE" >&2
        exit 1
    }
    echo "Metadata generated: $METADATA_FILE"
fi

# Create directories
mkdir -p "$LOG_DIR" "$OUT_DIR"

# Load models from file
mapfile -t MODELS < "$LLM_LIST_FILE"
# Remove empty lines and comments
MODELS=($(printf '%s\n' "${MODELS[@]}" | sed '/^$/d; /^#/d'))

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "Error: No models found in $LLM_LIST_FILE" >&2
    exit 1
fi

# Reverse the order if requested
if [[ "$REVERSE_ORDER" == true ]]; then
    # Create reversed array using bash array manipulation
    REVERSED_MODELS=()
    for ((i=${#MODELS[@]}-1; i>=0; i--)); do
        REVERSED_MODELS+=("${MODELS[i]}")
    done
    MODELS=("${REVERSED_MODELS[@]}")
fi

# Setup GPU array
IFS=',' read -ra gpu_array <<< "$GPUS"
num_gpus=${#gpu_array[@]}

# Calculate total concurrent slots
TOTAL_SLOTS=$((num_gpus * MAX_TASKS_PER_GPU))

# Create session log directory
SESSION_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_LOG_DIR="$LOG_DIR/calc_dna_$SESSION_TIMESTAMP"
mkdir -p "$SESSION_LOG_DIR"

echo "========================================="
echo "UNIFIED DNA CALCULATION PIPELINE"
echo "========================================="
echo "Models: ${#MODELS[@]} total"
echo "Processing order: $(if [[ "$REVERSE_ORDER" == true ]]; then echo "reverse (last to first)"; else echo "normal (first to last)"; fi)"
echo "Metadata: $METADATA_FILE"
echo "Datasets: $DATASETS"
echo "GPUs: ${gpu_array[*]}"
echo "Max tasks per GPU: $MAX_TASKS_PER_GPU"
echo "Total parallel slots: $TOTAL_SLOTS"
echo "DNA dimension: $DNA_DIM"
echo "Extractor type: $EXTRACTOR_TYPE"
echo "Max samples: $MAX_SAMPLES"
echo "Embedding merge: $EMBEDDING_MERGE"
echo "Reduction method: $REDUCTION_METHOD"
echo "Skip existing: $SKIP_EXISTING"
if [[ "$SKIP_CHAT_TEMPLATE" == true ]]; then
echo "Skip chat template: enabled (output to out_no_chat_template/)"
fi
echo "Session log directory: $SESSION_LOG_DIR"
echo "========================================="
echo ""

# Track running jobs per GPU and associated PIDs
declare -A gpu_pids  # gpu_index -> "pid1 pid2 ..."
declare -A pid_gpu   # pid -> gpu_index

# Initialize GPU tracking
for ((i=0; i<num_gpus; i++)); do
    gpu_pids[$i]=""
done

# Function to count jobs on a GPU
count_gpu_jobs() {
    local gpu_idx=$1
    local pid_list="${gpu_pids[$gpu_idx]}"
    if [[ -z "$pid_list" ]]; then
        echo 0
    else
        # Count running PIDs
        local count=0
        for pid in $pid_list; do
            if kill -0 "$pid" 2>/dev/null; then
                ((count++))
            fi
        done
        echo $count
    fi
}

# Function to clean up finished jobs from tracking
cleanup_finished_jobs() {
    for gpu_idx in "${!gpu_pids[@]}"; do
        local new_pid_list=""
        for pid in ${gpu_pids[$gpu_idx]}; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pid_list="$new_pid_list $pid"
            else
                unset pid_gpu[$pid]
            fi
        done
        gpu_pids[$gpu_idx]="${new_pid_list## }"  # Remove leading space
    done
}

# Function to check if a model is a chat model (using metadata)
is_chat_model() {
    local model_name="$1"
    if [[ ! -f "$METADATA_FILE" ]]; then
        return 1  # If no metadata, assume not chat model
    fi
    
    # Use the check_chat_model.py utility to check if model is a chat model
    (
        cd "$PROJECT_ROOT" || exit 1
        if [[ -f ".venv/bin/activate" ]]; then
            source .venv/bin/activate 2>/dev/null || true
        fi
        python3 -m reptrace.utils.check_chat_model --model-name "$model_name" --metadata-file "$METADATA_FILE" 2>/dev/null
    ) | grep -q "true"
}

# Function to check if a model-dataset combination already has valid output
check_existing_output() {
    local model_name="$1"
    local safe_model_name=$(echo "$model_name" | sed 's|/|_|g' | sed 's|:|_|g')
    
    # Create identifier for the unified dataset combination
    local dataset_identifier=$(echo "$DATASETS" | tr ',' '_')
    
    # Determine which directory to check based on model type and SKIP_CHAT_TEMPLATE setting
    local check_dir=""
    if is_chat_model "$model_name"; then
        # Chat model: check out_no_chat_template/ if SKIP_CHAT_TEMPLATE=true, otherwise out/
        if [[ "$SKIP_CHAT_TEMPLATE" == true ]]; then
            check_dir="$PROJECT_ROOT/out_no_chat_template/$dataset_identifier/$safe_model_name"
        else
            check_dir="$PROJECT_ROOT/out/$dataset_identifier/$safe_model_name"
        fi
    else
        # Non-chat model: always check out/ (they don't have separate no-template versions)
        check_dir="$PROJECT_ROOT/out/$dataset_identifier/$safe_model_name"
    fi
    
    local full_output_dir="$check_dir"
    
    # Check if directory exists
    if [[ ! -d "$full_output_dir" ]]; then
        return 1
    fi
    
    # Find DNA JSON file (excluding DRYRUN files)
    local dna_file=""
    for file in "$full_output_dir"/*_dna.json; do
        if [[ -f "$file" && ! "$file" =~ _DRYRUN\.json$ ]]; then
            dna_file="$file"
            break
        fi
    done
    
    # Check if DNA file exists and has content
    if [[ -z "$dna_file" || ! -f "$dna_file" || ! -s "$dna_file" ]]; then
        return 1
    fi
    
    # Check if JSON contains valid data
    if command -v jq >/dev/null 2>&1; then
        # Use jq if available for robust JSON parsing
        if jq -e '.signature | arrays | length > 0' "$dna_file" >/dev/null 2>&1; then
            return 0  # Regular DNA format (array)
        fi
    else
        # Fallback: simple grep check for signature array
        if grep -q '"signature".*\[' "$dna_file" 2>/dev/null; then
            return 0
        fi
    fi
    
    return 1
}

# Function to find GPU with available slot
find_available_gpu() {
    cleanup_finished_jobs
    
    for ((i=0; i<num_gpus; i++)); do
        local job_count=$(count_gpu_jobs $i)
        if [[ $job_count -lt $MAX_TASKS_PER_GPU ]]; then
            echo $i
            return 0
        fi
    done
    echo -1  # No available GPU
    return 1
}

echo "Starting DNA calculation for ${#MODELS[@]} models..."
echo "Parallel processing: $TOTAL_SLOTS total concurrent slots ($MAX_TASKS_PER_GPU per GPU)"
echo ""

# Process each model with continuous scheduling
model_idx=0

while [[ $model_idx -lt ${#MODELS[@]} ]]; do
    # Find an available GPU slot
    gpu_idx=$(find_available_gpu)
    
    if [[ $gpu_idx -ge 0 ]]; then
        # Found available slot, check if we should process this model
        model="${MODELS[$model_idx]}"
        gpu=${gpu_array[$gpu_idx]}
        current_jobs=$(count_gpu_jobs $gpu_idx)
        
        # Check if we should skip existing output
        if [[ "$SKIP_EXISTING" == true ]] && check_existing_output "$model"; then
            echo "Skipping $model (valid output already exists)"
            ((model_idx++))
            continue
        fi
        
        # Create safe model name for files
        safe_model_name=$(echo "$model" | sed 's|/|_|g' | sed 's|:|_|g')
        
        # Determine output directory based on model type
        # Chat models: use out_no_chat_template/ if SKIP_CHAT_TEMPLATE=true, otherwise out/
        # Non-chat models: always use out/
        dataset_identifier=$(echo "$DATASETS" | tr ',' '_')
        if is_chat_model "$model" && [[ "$SKIP_CHAT_TEMPLATE" == true ]]; then
            model_output_dir="$PROJECT_ROOT/out_no_chat_template/$dataset_identifier/$safe_model_name"
        else
            model_output_dir="$PROJECT_ROOT/out/$dataset_identifier/$safe_model_name"
        fi
        mkdir -p "$model_output_dir"
        
        # Log files
        # Logs per dataset under session dir
        mkdir -p "$SESSION_LOG_DIR/$dataset_identifier"
        log_file="$SESSION_LOG_DIR/$dataset_identifier/${safe_model_name}.txt"
        err_file="$SESSION_LOG_DIR/$dataset_identifier/${safe_model_name}.err"
        
        # Choose Python module based on dry run mode
        python_module="$PYTHON_MODULE"
        if [[ "$DRY_RUN" == true ]]; then
            python_module="reptrace.experiments.dryrun"
            echo "Starting job on GPU $gpu ($((current_jobs+1))/$MAX_TASKS_PER_GPU): $model (DRY RUN)"
        else
            echo "Starting job on GPU $gpu ($((current_jobs+1))/$MAX_TASKS_PER_GPU): $model"
        fi
        
        # Build and run command in background
        (
            cd "$PROJECT_ROOT"
            source .venv/bin/activate
            
            if [[ "$DRY_RUN" == true ]]; then
                # Determine output root directory (parent of dataset directory)
                model_output_root="$(dirname "$model_output_dir")/.."
                python -m "$python_module" \
                    --model-name "$model" \
                    --dataset "$DATASETS" \
                    --device "cuda:$gpu" \
                    --output-dir "$model_output_root" \
                    --min-duration 2 \
                    --max-duration 5 \
                    --gpu-memory-mb 100
            else
                # Determine output root directory (parent of dataset directory)
                model_output_root="$(dirname "$model_output_dir")/.."
                python -m "$python_module" \
                    --model-name "$model" \
                    --dataset "$DATASETS" \
                    --extractor-type "$EXTRACTOR_TYPE" \
                    --dna-dim "$DNA_DIM" \
                    --reduction-method "$REDUCTION_METHOD" \
                    --max-samples "$MAX_SAMPLES" \
                    --embedding-merge "$EMBEDDING_MERGE" \
                    $( [[ "$SKIP_CHAT_TEMPLATE" == true ]] && echo "--skip-chat-template" ) \
                    --metadata-file "$METADATA_FILE" \
                    --trust-remote-code \
                    --device "cuda:$gpu" \
                    --output-dir "$model_output_root" \
                    --log-level INFO
            fi
        ) > "$log_file" 2> "$err_file" &
        
        # Track the job
        job_pid=$!
        gpu_pids[$gpu_idx]="${gpu_pids[$gpu_idx]} $job_pid"
        pid_gpu[$job_pid]=$gpu_idx
        
        # Move to next model
        ((model_idx++))
    else
        # No available slots, wait for any job to complete
        echo "All GPU slots busy ($TOTAL_SLOTS/$TOTAL_SLOTS), waiting for a job to complete..."
        wait -n || true  # Wait for any background job to complete, but don't exit on error
        echo "A job completed, checking for available slots..."
    fi
done

# Wait for remaining jobs
echo "Waiting for final jobs to complete..."
wait || true  # Don't exit with error even if some jobs failed

# Generate summary
echo ""
echo "========================================="
echo "DNA CALCULATION SUMMARY"
echo "========================================="

successful=0
failed=0
success_models=()
failed_models=()

## Logs are written under a per-dataset subdirectory: $SESSION_LOG_DIR/<dataset_identifier>/
dataset_identifier=$(echo "$DATASETS" | tr ',' '_')
for model in "${MODELS[@]}"; do
    if check_existing_output "$model"; then
        ((successful++))
        success_models+=("$model")
    else
        ((failed++))
        failed_models+=("$model")
    fi
done

IFS=',' read -ra DATASETS_ARRAY <<< "$DATASETS"

echo "Total models: ${#MODELS[@]}"
echo "Datasets processed: ${#DATASETS_ARRAY[@]} (${DATASETS})"
echo "Embedding merge method: $EMBEDDING_MERGE"
echo "Reduction method: $REDUCTION_METHOD"
echo "Successful: $successful"
echo "Failed: $failed"
echo "Session log directory: $SESSION_LOG_DIR/$dataset_identifier"

# Also write structured summaries and a brief report via Python helper
(
    cd "$PROJECT_ROOT" || exit 0
    if [[ -d .venv ]]; then
        source .venv/bin/activate || true
    fi
    # Summary helper removed in minimal package
) 2>/dev/null

if [[ $failed -gt 0 ]]; then
    echo ""
    echo "Failed models:"
    for model in "${failed_models[@]}"; do
        echo "  - $model"
    done
fi

echo ""
if [[ $successful -eq ${#MODELS[@]} ]]; then
    echo "✓ All models completed successfully!"
    exit 0
elif [[ $successful -gt 0 ]]; then
    echo "⚠ Partial success: $successful/${#MODELS[@]} models completed"
    exit 0
else
    echo "✗ All models failed"
    exit 1
fi
