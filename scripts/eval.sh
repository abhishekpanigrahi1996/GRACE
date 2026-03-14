
#Generate from trained model
STUDENT_PTH="output/Llama-3.2-1B-Teacher=Qwen2.5-3B-Instruct"


PYTHON_SCRIPT=data_generation/generate_responses.py
NUM_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

MODEL_PATH="${STUDENT_PTH}"
INPUT_PATH="data/gsm8k/test.jsonl"
OUTPUT_PATH="${STUDENT_PTH}/gsm8k_test.jsonl"
TEMPERATURE=1.0
TEMPLATE=$'### Problem:\n{problem}\n\n### Solution:\n'

python "${PYTHON_SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --input_file "${INPUT_PATH}" \
    --output_file "${OUTPUT_PATH}" \
    --temperature "${TEMPERATURE}" \
    --tensor_parallel_size "${NUM_VISIBLE_DEVICES}" \
    --template "${TEMPLATE}"



PYTHON_PATH="data_generation/eval_script.py"
INPUT_PATH=${OUTPUT_PATH}

python "${PYTHON_PATH}" --input-path ${INPUT_PATH} --use-last-number