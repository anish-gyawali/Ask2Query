import json
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_small_spider_subset():
    os.makedirs("data/spider_filtered_small", exist_ok=True)

    # Load full filtered data
    with open("data/spider_filtered/train_spider_filtered.json", "r") as f:
        train_data = json.load(f)

    with open("data/spider_filtered/val_spider_filtered.json", "r") as f:
        val_data = json.load(f)

    # Shuffle randomly
    random.shuffle(train_data)
    random.shuffle(val_data)

    # Take small samples
    small_train = train_data[:400]
    small_val = val_data[:100]

    # Save small datasets
    with open("data/spider_filtered_small/train_spider_filtered_small.json", "w") as f:
        json.dump(small_train, f, indent=2)

    with open("data/spider_filtered_small/val_spider_filtered_small.json", "w") as f:
        json.dump(small_val, f, indent=2)

    print(f"Small Spider subset created successfully!")

if __name__ == "__main__":
    create_small_spider_subset()
