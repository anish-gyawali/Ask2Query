import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import config
from scripts.dataset import TextToSQLDataset
from scripts.utils import compute_metrics, plot_confusion_matrix, plot_model_performance

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

    accuracy, f1, _ = compute_metrics(labels, preds)
    plot_confusion_matrix(labels, preds, output_image)

    return accuracy, f1

if __name__ == "__main__":
    base_metrics = evaluate(config.OUTPUT_DIR_BASE, "outputs/confusion_matrix_base.png", "Base")
    finetuned_metrics = evaluate(config.OUTPUT_DIR_FINETUNED, "outputs/confusion_matrix_finetuned.png", "Fine-tuned")

    # Plot model performance comparison
    os.makedirs("outputs", exist_ok=True)
    plot_model_performance(base_metrics, finetuned_metrics, "outputs/model_performance_comparison.png")

    print("\nEvaluation completed successfully!")
