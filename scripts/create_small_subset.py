import json
import random
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def create_small_subset():
    os.makedirs(config.SMALL_DATA_DIR, exist_ok=True)

    with open(f"{config.FILTERED_DATA_DIR}/train_{config.DATASET_NAME}_filtered.json", "r") as f:
        train_data = json.load(f)

    with open(f"{config.FILTERED_DATA_DIR}/test_{config.DATASET_NAME}_filtered.json", "r") as f:
        test_data = json.load(f)

    # Separate examples
    select_examples = [ex for ex in train_data if ex['sql'].lower().startswith("select")]
    insert_examples = [ex for ex in train_data if ex['sql'].lower().startswith("insert into")]
    delete_examples = [ex for ex in train_data if ex['sql'].lower().startswith("delete from")]

    # Sample
    small_train_select = random.sample(select_examples, min(380, len(select_examples)))
    small_train_insert = random.choices(insert_examples, k=10) if len(insert_examples) > 0 else []
    small_train_delete = random.choices(delete_examples, k=10) if len(delete_examples) > 0 else []

    small_train = small_train_select + small_train_insert + small_train_delete
    random.shuffle(small_train)

    # test data
    random.shuffle(test_data)
    small_val = test_data[:100]

    with open(f"{config.SMALL_DATA_DIR}/train_{config.DATASET_NAME}_filtered_small.json", "w") as f:
        json.dump(small_train, f, indent=2)

    with open(f"{config.SMALL_DATA_DIR}/test_{config.DATASET_NAME}_filtered_small.json", "w") as f:
        json.dump(small_val, f, indent=2)

    print(f"Balanced small {config.DATASET_NAME} subset created successfully!")
    print(f"Training Set: {len(small_train)} examples (SELECT: {len(small_train_select)}, INSERT: {len(small_train_insert)}, DELETE: {len(small_train_delete)})")
    print(f"Testing Set: {len(small_val)} examples")

if __name__ == "__main__":
    create_small_subset()
