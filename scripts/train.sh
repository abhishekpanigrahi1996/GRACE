
# first tokenize


PYTHON_PATH="data_generation/tokenize_data.py"
INPUT_FILE="data/gsm8k_synthetic/Qwen2.5-3B-Instruct.jsonl"
OUTPUT_FILE="data/gsm8k_synthetic/Qwen2.5-3B-Instruct.hf"
TOKENIZER="models/Llama-3.2-1B"
SUBSAMPLE_QUESTIONS=-1
SUBSAMPLE_RESPONSES=-1


python "${PYTHON_PATH}" \
    --tokenizer-path ${TOKENIZER} \
    --input-path ${INPUT_FILE} \
    --output-path ${OUTPUT_FILE} \
    --subsample-questions ${SUBSAMPLE_QUESTIONS} \
    --subsample-responses ${SUBSAMPLE_RESPONSES}



# now train

PYTHON_PATH="sft/sft_distil.py"
NUM_EPOCHS=3
max_steps=-1
save_freq=10_000
eval_freq=10_000
seed=42
weight_decay=0.0

DATA_PATH="data/gsm8k_synthetic/Qwen2.5-3B-Instruct.hf"
MODEL_PATH="models/Llama-3.2-1B"
TOKENIZER=${MODEL_PATH}

OUTPUT_DIR="output/Llama-3.2-1B-Teacher=Qwen2.5-3B-Instruct"
LOGGING_DIR=${OUTPUT_DIR}"/logs"
lr=1e-5
NET_BATCH_SIZE=256
BATCH_SIZE=8


nvidia-smi

# Determine available number of GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_gpus=$(nvidia-smi -L | wc -l)
else
    num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")
fi
num_gpus=${NUM_GPUS:-$num_gpus}

# Determine number of nodes if run inside slurm job job
num_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
if [ $num_nodes == 0 ]; then
    num_nodes=1
fi
num_nodes=${NUM_NODES:-$num_nodes}

if [ $num_nodes -gt 1 ]; then
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    master_port=56321
else
    # Find a free port at random
    #master_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
    master_port=$((RANDOM % 1000 + 57980))
    master_addr=localhost
fi
master_addr=${MASTER_ADDR:-$master_addr}
master_port=${MASTER_PORT:-$master_port}



header="torchrun \
--rdzv-backend=c10d \
--rdzv-endpoint=$master_addr:$master_port \
--nnodes=$num_nodes \
--nproc-per-node=$num_gpus \
${PYTHON_PATH}"

export OMP_NUM_THREADS=$num_gpus


export FSDP_SHARDING_STRATEGY="5" # 5 corresponds to _hybrid_shard_zero2
export FSDP_STATE_DICT_TYPE="FULL_STATE_DICT"
export LOGIT_BLOCK_SIZE=2048  # Compute Llama logits in blocks of 2048 tokens
export TOKENIZERS_PARALLELISM=true

ACCU_STEPS=$(($NET_BATCH_SIZE / $BATCH_SIZE / $num_gpus / $num_nodes))


${header}  \
    --do_train True \
    --num_train_epochs ${NUM_EPOCHS}  \
    --max_steps ${max_steps} \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.06 \
    --weight_decay $weight_decay \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --max_sequence_length 4096 \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps $save_freq \
    --overwrite_output_dir True \
    --optim adamw_torch \
    --gradient_checkpointing True \
    --seed $seed \
    --data_path $DATA_PATH \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name $TOKENIZER \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOGGING_DIR \
    --learning_rate $lr \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $ACCU_STEPS \
    --eval_strategy "steps" \
    --eval_steps $eval_freq \
    --per_device_eval_batch_size $((BATCH_SIZE)) \
    --eval_accumulation_steps 16 \
    --preprocessing_num_workers 20 \
    --bf16 \
    --bf16_full_eval \
    --fsdp auto_wrap \
    --ddp_find_unused_parameters false


rm -rf $OUTPUT_DIR"/checkpoint-*"