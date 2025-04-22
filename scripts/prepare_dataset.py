import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import config

def prepare_dataset():
    os.makedirs(config.FILTERED_DATA_DIR, exist_ok=True)

    def filter_data(input_path, output_path):
        data = []
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        filtered = []
        for example in data:
            question = example.get('sql_prompt', "").strip()
            sql = example.get('sql', "").strip()

            if not question or not sql:
                continue

            sql_lower = sql.lower()

            # Keep only SELECT, INSERT, DELETE
            if ("select" in sql_lower or "insert" in sql_lower or "delete" in sql_lower):
                # Remove "SQL:" if it exists
                if sql_lower.startswith("sql:"):
                    sql = sql[4:].strip()

                filtered.append({
                    "question": question,
                    "sql": sql
                })

        with open(output_path, 'w') as f:
            json.dump(filtered, f, indent=2)

    filter_data(f"{config.RAW_DATA_DIR}/train.json", f"{config.FILTERED_DATA_DIR}/train_{config.DATASET_NAME}_filtered.json")
    filter_data(f"{config.RAW_DATA_DIR}/test.json", f"{config.FILTERED_DATA_DIR}/test_{config.DATASET_NAME}_filtered.json")

if __name__ == "__main__":
    prepare_dataset()
