

# Ask2Query

**Ask2Query** is a Deep Learning based Text-to-SQL model that translates natural language questions into corresponding SQL templates (SELECT, INSERT, DELETE) using fine-tuned DistilBERT.  
This project demonstrates transfer learning, classification, regularization, and fine-tuning on a real-world NLP task.

---

## Project Folder Structure

```plaintext
Ask2Query/
│
├── LICENSE
├── README.md
├── requirements.txt
├── config.py
├── outputs/                # Evaluation outputs (confusion matrix, etc.)
├── models/                 # Saved models
│   ├── base_model/
│   └── fine_tuned_model/
├── data/                   # Processed dataset
│   ├── gretel_filtered/
│   └── gretel_filtered_small/
├── gretel_dataset/          # Downloaded raw dataset
│   ├── dataset_dict.json
│   ├── train/
│   ├── test/
│   ├── train.json
│   └── test.json
├── scripts/                 # Core scripts
│   ├── download_dataset.py
│   ├── prepare_dataset.py
│   ├── create_small_subset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── dataset.py
│   ├── utils.py
│   └── count.py
└── __pycache__/
```

---

##  Dataset Used

**Dataset Name:** [GretelAI Synthetic Text-to-SQL Dataset](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)

- Contains natural language questions and their corresponding SQL queries.
- Includes various SQL types: SELECT, INSERT, DELETE, CREATE, etc.
- Publicly available on HuggingFace Datasets Hub.

---

##  Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/anish-gyawali/Ask2Query.git
cd Ask2Query
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# or
.venv\Scripts\activate      # Windows
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

---

## How to Run the Project (Step-by-Step)

Follow these steps in order to fully prepare, train, and test the model.

---

### 1️. Download Dataset

Download the Gretel synthetic dataset:

```bash
python scripts/download_dataset.py
```

---

### 2️. Prepare Dataset (filter only SELECT, INSERT, DELETE)

```bash
python scripts/prepare_dataset.py
```

---

### 3️. Create Small Subset (for faster training)

```bash
python scripts/create_small_subset.py
```

---

### 4️. Fine-tune the Model

Train the model on the small balanced dataset:

```bash
python scripts/train.py
```

---

### 5️. Evaluate the Model

Evaluate base model vs fine-tuned model:

```bash
python scripts/evaluate.py
```

- Accuracy and F1 Score for both base and fine-tuned models will be shown.
- Confusion matrices will be saved inside the `outputs/` folder.

---

### 6️.Run Inference (Interactive Mode)

Test the model with your own live questions:

```bash
python scripts/inference.py
```

Examples you can try:

- "Show all students who scored more than 80 percent."
- "Insert a new customer named John Smith into the database."
- "Delete the employee record where ID is 5."

 The model will predict the appropriate SQL template!

---

## Features Highlighted in Project

-  Fine-tuning of pre-trained DistilBERT
-  Text classification based on SQL templates
-  Real-world dataset usage and filtering
-  Balanced subset creation (oversampling INSERT/DELETE)
-  Evaluation with accuracy, F1 Score, confusion matrix
-  Live terminal inference for end-to-end testing

---

##  Requirements

- Python 3.8+
- PyTorch
- HuggingFace Transformers
- scikit-learn
- matplotlib
- datasets

*(All included in `requirements.txt`)*

---

##  License

This project is licensed under the **MIT License** — feel free to use, modify, and share.

---

##  Credits

- Dataset Source: [GretelAI Synthetic Text-to-SQL Dataset](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)
- Model Architecture: [HuggingFace DistilBERT](https://huggingface.co/distilbert-base-uncased)
- Developed as part of Deep Learning coursework(**CMPS-6720**)

---


##  Tips When Using Different Datasets

- This project is currently configured for the **GretelAI Synthetic Text-to-SQL** dataset, where:
  - Natural language question field = `sql_prompt`
  - SQL query field = `sql`
  
- ️ **If you switch to a different dataset** (like Spider, Scholar, WikiSQL), the field names may be different!
  
- For example:
  - **Spider** dataset uses fields like `question` and `query`
  - **Gretel** dataset uses `sql_prompt` and `sql`
  
- You must **update `prepare_dataset.py` accordingly**:
  
  ```python
  # Example for Gretel dataset (current default):
  question = example.get('sql_prompt', "").strip()
  sql = example.get('sql', "").strip()
  
  # If using Spider dataset:
  question = example.get('question', "").strip()
  sql = example.get('query', "").strip()
