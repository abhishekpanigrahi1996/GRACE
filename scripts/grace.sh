#first subsample some data and tokenize them
PYTHON_PATH="data_generation/tokenize_data.py"
INPUT_FILE="data/gsm8k_synthetic/Qwen2.5-3B-Instruct.jsonl"
OUTPUT_FILE="grace_data/gsm8k_synthetic/Qwen2.5-3B-Instruct.hf"
TOKENIZER="models/Llama-3.2-1B"
SUBSAMPLE_QUESTIONS=512
SUBSAMPLE_RESPONSES=4


python "${PYTHON_PATH}" \
    --tokenizer-path ${TOKENIZER} \
    --input-path ${INPUT_FILE} \
    --output-path ${OUTPUT_FILE} \
    --subsample-questions ${SUBSAMPLE_QUESTIONS} \
    --subsample-responses ${SUBSAMPLE_RESPONSES}


#then, compute gradients from the subsampled data
PYTHON_PATH="GRACE/gradient_computation.py"
MODEL_PATH="models/Llama-3.2-1B"
INPUT_FILE=${OUTPUT_FILE}
OUTPUT_FILE="grace_data/gsm8k_synthetic/Gradients_Qwen2.5-3B-Instruct.pkl"
PROJ_DIM=512


python "${PYTHON_PATH}" \
    --model ${MODEL_PATH} \
    --data-path ${INPUT_FILE} \
    --output-path ${OUTPUT_FILE} \
    --proj-dim ${PROJ_DIM}

#then, compute GRACE
PYTHON_PATH="GRACE/GRACE_computation.py"
GRADIENT_PATH=${OUTPUT_FILE}

python "${PYTHON_PATH}" \
    --gradients-path ${GRADIENT_PATH} \
    --dim ${PROJ_DIM} \
    --n-gen-per-prompt ${SUBSAMPLE_RESPONSES}