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
