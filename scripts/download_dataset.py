import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
import config

# Create directory
os.makedirs(config.RAW_DATA_DIR, exist_ok=True)

# Load the dataset dynamically
if config.DATASET_NAME == "spider":
    dataset = load_dataset("xlangai/spider")
elif config.DATASET_NAME == "gretel":
    dataset = load_dataset("gretelai/synthetic_text_to_sql")
else:
    raise ValueError("Unsupported dataset name!")

# Save
dataset.save_to_disk(config.RAW_DATA_DIR)

# Save CSV and JSON
for split in dataset.keys():
    dataset[split].to_json(f"{config.RAW_DATA_DIR}/{split}.json")

print(f"Dataset {config.DATASET_NAME} downloaded and saved successfully!")
