import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

def prepare_spider():
    os.makedirs("data/spider_filtered", exist_ok=True)

    def filter_data(input_path, output_path):
        data = []
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        filtered = []
        for example in data:
            sql = example['query'].lower()

            # Keep only simple SELECT, INSERT, DELETE queries, no JOIN, no nested SELECT
            if ("join" not in sql) and ("select" in sql or "insert" in sql or "delete" in sql):
                filtered.append({
                    "question": example['question'],
                    "sql": example['query']
                })

        with open(output_path, 'w') as f:
            json.dump(filtered, f, indent=2)

    filter_data('spider_dataset/train.json', 'data/spider_filtered/train_spider_filtered.json')
    filter_data('spider_dataset/validation.json', 'data/spider_filtered/val_spider_filtered.json')

if __name__ == "__main__":
    prepare_spider()
