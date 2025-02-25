import os

ROOT_DIR = os.path.abspath("./tuning")

print(ROOT_DIR)

DATA_DIR = "/data"

DATASETS_DIR = ROOT_DIR + DATA_DIR + "/datasets"
MODELS_DIR = ROOT_DIR + "/models" 
OUTPUTS_DIR = ROOT_DIR + "/outputs"

IFEVAL_OUTPUTS_DIR = OUTPUTS_DIR + "/ifeval"
GSM8K_OUTPUTS_DIR = OUTPUTS_DIR + "/gsm8k"

CHATML_TEMPLATE_PATH = ROOT_DIR + "/chatml.py"

PREF_DATASETS_DIR = DATASETS_DIR + "/preference_dataset_outputs"

RESPONSES_FILENAME = "responses.jsonl"
RESPONSES_ANNOTATED_FILENAME = "responses_annotated.jsonl"

HF_MODEL_MAP = {
    "llama3-8B": "unsloth/Meta-Llama-3.1-8B",
    "qwen2-7B": "unsloth/Qwen2.5-7B",
    "qwen2-3B": "unsloth/Qwen2.5-3B",
    "qwen2-2B": "unsloth/Qwen2.5-1.5B",
    "qwen2-14B": "unsloth/Qwen2.5-14B",
    "llama3-3B": "unsloth/Llama-3.2-3B",
    "llama3-1B": "unsloth/Llama-3.2-1B"
}