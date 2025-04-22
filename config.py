# dataset
DATASET_NAME = "gretel"

# Other configs
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# Data paths (automatically based on DATASET_NAME)
RAW_DATA_DIR = f"{DATASET_NAME}_dataset"
FILTERED_DATA_DIR = f"data/{DATASET_NAME}_filtered"
SMALL_DATA_DIR = f"data/{DATASET_NAME}_filtered_small"

# Files
TRAIN_FILE = f"{SMALL_DATA_DIR}/train_{DATASET_NAME}_filtered_small.json"
VAL_FILE = f"{SMALL_DATA_DIR}/test_{DATASET_NAME}_filtered_small.json"

# Labels/Templates (you can update later)
LABEL_TO_TEMPLATE = {
    0: "SELECT column FROM table WHERE condition",
    1: "INSERT INTO table (columns) VALUES (values)",
    2: "DELETE FROM table WHERE condition",
}

TEMPLATE_TO_LABEL = {v: k for k, v in LABEL_TO_TEMPLATE.items()}
NUM_LABELS = len(LABEL_TO_TEMPLATE)

# Model output directories
OUTPUT_DIR_BASE = "models/base_model/"
OUTPUT_DIR_FINETUNED = "models/fine_tuned_model/"
