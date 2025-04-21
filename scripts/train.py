import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import config
from scripts.dataset import TextToSQLDataset

def train():
    tokenizer = DistilBertTokenizerFast.from_pretrained(config.MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=config.NUM_LABELS, ignore_mismatched_sizes=True)

    train_dataset = TextToSQLDataset(config.TRAIN_FILE, tokenizer)
    val_dataset = TextToSQLDataset(config.VAL_FILE, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    os.makedirs(config.OUTPUT_DIR_FINETUNED, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR_BASE, exist_ok=True)

    model.save_pretrained(config.OUTPUT_DIR_BASE)

    device = torch.device("cpu")
    model.to(device)

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        print(f"Epoch {epoch} - Loss: {total_loss / len(train_loader)}")

    model.save_pretrained(config.OUTPUT_DIR_FINETUNED)
    print("Training completed and model saved!")

if __name__ == "__main__":
    train()
