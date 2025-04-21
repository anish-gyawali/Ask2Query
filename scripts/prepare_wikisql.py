import json
import random
import os

def extract_small_wikisql(input_file, output_file, n=500):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(input_file, 'r') as f:
        lines = f.readlines()

    random.shuffle(lines)
    selected = lines[:n]

    with open(output_file, 'w') as f_out:
        for line in selected:
            f_out.write(line)

if __name__ == "__main__":
    # Modify these paths if needed based on where you clone WikiSQL
    extract_small_wikisql("data/wikisql/train.jsonl", "data/wikisql/train_small.jsonl")
    extract_small_wikisql("data/wikisql/dev.jsonl", "data/wikisql/dev_small.jsonl")
