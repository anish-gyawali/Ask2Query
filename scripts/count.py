import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def count_labels(path):
    with open(path, "r") as f:
        data = json.load(f)

    select_count = 0
    insert_count = 0
    delete_count = 0

    for example in data:
        sql = example['sql'].lower()
        if sql.startswith("select"):
            select_count += 1
        elif sql.startswith("insert into"):
            insert_count += 1
        elif sql.startswith("delete from"):
            delete_count += 1

    print(f"SELECT: {select_count}")
    print(f"INSERT: {insert_count}")
    print(f"DELETE: {delete_count}")


if __name__ == "__main__":
    count_labels("data/gretel_filtered/train_gretel_filtered.json")
