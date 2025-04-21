# config.py

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

TRAIN_FILE = "data/train_real.json"
VAL_FILE = "data/val_real.json"

NUM_LABELS = 3  # Number of SQL templates
OUTPUT_DIR_BASE = "models/base_model/"
OUTPUT_DIR_FINETUNED = "models/fine_tuned_model/"

LABEL_TO_TEMPLATE = {
    0: "SELECT name FROM employees WHERE id = {id}",
    1: "SELECT salary FROM employees WHERE id = {id}",
    2: "SELECT department FROM employees WHERE id = {id}",
}
TEMPLATE_TO_LABEL = {v: k for k, v in LABEL_TO_TEMPLATE.items()}
