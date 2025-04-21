import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import config
from scripts.dataset import TextToSQLDataset
from scripts.utils import compute_metrics, plot_confusion_matrix

def evaluate(model_path, output_image, model_name):
    print(f"\nEvaluating {model_name} Model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(config.MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to("cpu")
    model.eval()

    val_dataset = TextToSQLDataset(config.VAL_FILE, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)

    preds, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to("cpu") for k, v in batch.items()}
            outputs = model(**batch)
            preds += outputs.logits.argmax(dim=1).tolist()
            labels += batch['labels'].tolist()

    compute_metrics(labels, preds)
    plot_confusion_matrix(labels, preds, output_image)

if __name__ == "__main__":
    evaluate(config.OUTPUT_DIR_BASE, "outputs/confusion_matrix_base.png", "Base")
    evaluate(config.OUTPUT_DIR_FINETUNED, "outputs/confusion_matrix_finetuned.png", "Fine-tuned")
