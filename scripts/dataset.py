import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import config
import torch
from torch.utils.data import Dataset

class TextToSQLDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = json.load(open(file_path))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        sql = self.data[idx]['sql']

        template = self.match_template(sql)
        if template not in config.TEMPLATE_TO_LABEL:
            label = 0  # fallback to default if no match
        else:
            label = config.TEMPLATE_TO_LABEL[template]

        inputs = self.tokenizer(question, truncation=True, padding='max_length', max_length=config.MAX_LENGTH, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = torch.tensor(label)
        return item

    def match_template(self, sql_query):
        sql_query = sql_query.lower()
        if sql_query.startswith("insert into"):
            return "INSERT INTO table (columns) VALUES (values)"
        elif sql_query.startswith("delete from"):
            return "DELETE FROM table WHERE condition"
        elif sql_query.startswith("select"):
            return "SELECT column FROM table WHERE condition"
        else:
            return None
