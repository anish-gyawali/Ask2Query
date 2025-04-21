import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def convert_jsonl_to_json(input_path, output_path):
    with open(input_path, 'r') as infile:
        lines = infile.readlines()
        data = []
        for line in lines:
            obj = json.loads(line)
            question = obj['question']
            # Assume WikiSQL format has a 'sql' field
            sql_query = "SELECT * FROM table"
            data.append({
                "question": question,
                "sql": sql_query
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as outfile:
        json.dump(data, outfile, indent=2)


if __name__ == "__main__":
    convert_jsonl_to_json("data/wikisql/train_small.jsonl", "data/train_real.json")
    convert_jsonl_to_json("data/wikisql/dev_small.jsonl", "data/val_real.json")
