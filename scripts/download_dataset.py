from datasets import load_dataset
import os

# Create directory for saving the dataset
os.makedirs("spider_dataset", exist_ok=True)

# Load the Spider dataset from the xlangai repository
spider_dataset = load_dataset("xlangai/spider")

# Save the dataset to disk in Parquet format (efficient storage)
spider_dataset.save_to_disk("spider_dataset")

# Optionally, you can also save it in other formats
# For example, to CSV files:
for split in spider_dataset.keys():
    spider_dataset[split].to_csv(f"spider_dataset/{split}.csv")

# Or to JSON files:
for split in spider_dataset.keys():
    spider_dataset[split].to_json(f"spider_dataset/{split}.json")

print("Dataset downloaded and saved successfully!")