# generate 16 responses per question from a teacher model

PYTHON_SCRIPT=data_generation/generate_responses.py
NUM_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

MODEL_PATH=models/Qwen2.5-3B-Instruct
INPUT_PATH=data/gsm8k/train.jsonl
OUTPUT_PATH=data/gsm8k_synthetic/Qwen2.5-3B-Instruct.jsonl
TEMPERATURE=0.6


python "${PYTHON_SCRIPT}" \
    --model_path ${MODEL_PATH} \
    --input_file ${INPUT_PATH} \
    --output_file ${OUTPUT_PATH} \
    --temperature ${TEMPERATURE} \
    --tensor_parallel_size ${NUM_VISIBLE_DEVICES}

