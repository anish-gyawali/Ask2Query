from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    cm = confusion_matrix(labels, preds)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["SELECT", "INSERT", "DELETE"]))

    return accuracy, f1, cm

def plot_confusion_matrix(labels, preds, output_file):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["SELECT", "INSERT", "DELETE"], yticklabels=["SELECT", "INSERT", "DELETE"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(output_file)
    plt.close()

def plot_model_performance(base_metrics, finetuned_metrics, output_file):
    base_acc, base_f1 = base_metrics
    fine_acc, fine_f1 = finetuned_metrics

    labels = ['Base Model', 'Fine-tuned Model']
    accuracies = [base_acc * 100, fine_acc * 100]
    f1_scores = [base_f1 * 100, fine_f1 * 100]

    x = range(len(labels))
    width = 0.35  # width of the bars

    plt.figure(figsize=(8, 6))
    plt.bar(x, accuracies, width, label='Accuracy')
    plt.bar([i + width for i in x], f1_scores, width, label='F1 Score')

    plt.xlabel('Model')
    plt.ylabel('Percentage (%)')
    plt.title('Model Performance Comparison')
    plt.xticks([i + width/2 for i in x], labels)
    plt.legend()
    plt.ylim(0, 110)
    plt.grid(axis='y')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
