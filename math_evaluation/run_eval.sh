
# python examples/data_preprocess/openthought_cot.py --local_dir /home/nsml/qqplot_nap/data
# Select the prompt format matching your model
PROMPT_TYPE="qwen-boxed"
# PROMPT_TYPE="llama-base-boxed"
# PROMPT_TYPE="deepseek-math"

# Set available GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configure sampling settings
N_SAMPLING=16
TEMPERATURE=1

# Specify model and output directories

MODEL_NAME_OR_PATH=""
OUTPUT_DIR="outputs/"

# Run evaluation
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR $N_SAMPLING $TEMPERATURE