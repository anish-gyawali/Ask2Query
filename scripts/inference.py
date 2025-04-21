import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import config

def infer(model, tokenizer, question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding='max_length',
                       max_length=config.MAX_LENGTH)
    outputs = model(**inputs)
    pred_label = outputs.logits.argmax(dim=1).item()
    sql = config.LABEL_TO_TEMPLATE[pred_label]
    return sql


if __name__ == "__main__":
    tokenizer = DistilBertTokenizerFast.from_pretrained(config.MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(config.OUTPUT_DIR_FINETUNED)
    model.to("cpu")
    model.eval()

    print("\nAsk2Query: Ask your question! (Type 'exit' to quit)\n")

    while True:
        question = input("Your question: ")
        if question.lower() == "exit":
            print("Goodbye!")
            break

        predicted_sql = infer(model, tokenizer, question)
        print(f"\nPredicted SQL Query:\n{predicted_sql}\n")
