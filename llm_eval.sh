#!/bin/bash

activate () {
    echo "Activating virtual environment..."
    . .venv/bin/activate
}
cd ~/llm_eval
activate

# Inspired by the Euro-LLM-Leaderboard: https://huggingface.co/spaces/occiglot/euro-llm-leaderboard

BATCH_SIZE="8"
BATCH_SIZE="auto:4"

GPU_NUMBER=0 # Set to -1 to run on all GPUs

# 0: Qwen-2.5-7B ARC
# 1: Qwen-2.5-7B MMLU


TASKS=(
    "TruthfulQA"
    "Belebele"
    "ARC"
    "MMLU"
) # Exclude "HellaSwag" for cost reasons

MODELS=(
    # ========== 1-3B range ==========
    #"google/flan-t5-large" # completed
    #"google/flan-t5-xl" # completed
    #"Qwen/Qwen2.5-1.5B-Instruct" # completed
    #"Qwen/Qwen2-1.5B-Instruct" # completed
    #"Qwen/Qwen2-0.5B-Instruct" # completed
    #"microsoft/Phi-3.5-mini-instruct" # completed
    #"microsoft/Phi-3-mini-128k-instruct" # Belebele, ARC, and MMLU fail with OOM
    #"meta-llama/Llama-3.2-1B-Instruct" # completed
    #"meta-llama/Llama-3.2-3B-Instruct" # completed
    #"utter-project/EuroLLM-1.7B-Instruct" # completed
    #"BSC-LT/salamandra-2b-instruct" # completed
    # ========== 7-8B range ==========
    #"BSC-LT/salamandra-7b-instruct" # completed
    #"CohereForAI/aya-expanse-8b" # completed
    #"Qwen/Qwen2.5-7B-Instruct" # completed
    #"Qwen/Qwen2-7B-Instruct" # completed
    #"mistralai/Mistral-7B-Instruct-v0.3" # completed
    #"mistralai/Ministral-8B-Instruct-2410" # completed
    #"meta-llama/Meta-Llama-3.1-8B-Instruct" # completed
    #"microsoft/Phi-3-small-128k-instruct" # completed: Belebele needs constant batch size 16
    #"CohereForAI/aya-23-8B" # completed
    #"google/gemma-2-9b-it" # completed: ARC needs constant batch size 8
    # ========== 12-14B range ==========
    #"google/flan-t5-xxl" # completed
    "Qwen/Qwen2.5-14B-Instruct" # completed
    #"microsoft/Phi-3-medium-128k-instruct" # completed
    #"Qwen/Qwen1.5-14B-Chat" # completed
    #"mistralai/Mistral-Nemo-Instruct-2407" # completed, currently needs transformers from source: pip install git+https://github.com/huggingface/transformers.git
    #"stabilityai/stablelm-2-12b-chat" # completed
)
OUTPUT_DIR="./results"

# Create the main results directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to run lm_eval with given parameters
run_lm_eval() {
    local model=$1
    local task_list=$2
    local num_fewshot=$3
    local task_name=$4

    # Prepare the command
    cmd="lm_eval \
        --model hf \
        --model_args pretrained=$model \
        --tasks $task_list \
        --num_fewshot $num_fewshot \
        --batch_size $BATCH_SIZE \
        --output_path ${OUTPUT_DIR} \
        --use_cache ${OUTPUT_DIR}/cache/$model/$task_name \
        --trust_remote_code"

# These are only applicable for chat models
#        --apply_chat_template \
#        --fewshot_as_multiturn"

    if [ $GPU_NUMBER -eq -1 ]; then
        # Run on all GPUs
        cmd="accelerate launch --no-python $cmd"
    elif [[ $GPU_NUMBER == *","* ]]; then # if there is a comma in the GPU_NUMBER
        # Run on multiple GPUs
        cmd="env CUDA_VISIBLE_DEVICES=$GPU_NUMBER accelerate launch --no-python $cmd"
    else
        # Run on specific GPU
        cmd="env CUDA_VISIBLE_DEVICES=$GPU_NUMBER $cmd"
    fi

    echo $cmd

    echo "Running $task_name with $num_fewshot-shot for model $model"
    # IMPORTANT: Remove "accelerate launch --no-python" to run on only one GPU
    if ! $cmd; then
        echo "The evaluation for $task_name with model $model failed. Please check the logs in ${OUTPUT_DIR} for more information."
        return 1
    fi
    echo "Completed $task_name for $model"
    echo "--------------------"
}

# Main execution loop
for MODEL in "${MODELS[@]}"; do
    echo "Starting evaluations for model: $MODEL"

    if [[ " ${TASKS[*]} " =~ " TruthfulQA " ]]; then
        # TruthfulQA: approx. 3m on A100 for Phi-3-medium-128k-instruct
        run_lm_eval "$MODEL" "truthfulqa_mc2,truthfulqa_de_mc2,truthfulqa_fr_mc2,truthfulqa_es_mc2,truthfulqa_it_mc2" 0 "TruthfulQA" || continue
    fi

    if [[ " ${TASKS[*]} " =~ " Belebele " ]]; then
        # Belebele: approx. 5m on A100 for Phi-3-medium-128k-instruct
        run_lm_eval "$MODEL" "belebele_eng_Latn,belebele_ita_Latn,belebele_deu_Latn,belebele_fra_Latn,belebele_spa_Latn" 5 "Belebele" || continue
    fi

    if [[ " ${TASKS[*]} " =~ " ARC " ]]; then
        # ARC: approx. 20m on A100 for Phi-3-medium-128k-instruct
        run_lm_eval "$MODEL" "arc_challenge,arc_de,arc_es,arc_it,arc_fr" 25 "ARC" || continue
    fi

    if [[ " ${TASKS[*]} " =~ " MMLU " ]]; then
        # MMLU: approx. 1h on A100 for Phi-3-medium-128k-instruct
        run_lm_eval "$MODEL" "m_mmlu_en,m_mmlu_de,m_mmlu_fr,m_mmlu_es,m_mmlu_it" 5 "MMLU" || continue
    fi

    if [[ " ${TASKS[*]} " =~ " HellaSwag " ]]; then
        # HellaSwag: approx. 2.5h on A100 for Phi-3-medium-128k-instruct (16h for all) ==> skip for cost reasons
        run_lm_eval "$MODEL" "hellaswag,hellaswag_es,hellaswag_it,hellaswag_fr,hellaswag_de" 10 "HellaSwag" || continue
    fi

    # Possible additional supported benchmark: lambada_openai_mt_en,lambada_openai_mt_de,lambada_openai_mt_fr,lambada_openai_mt_es,lambada_openai_mt_it

    echo "All tasks completed for model: $MODEL"
done

echo "All evaluations completed."
