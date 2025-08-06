import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_names = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "albert-base-v2"
]

def evaluate_model(model_dir, batch_size=32):
    print(f"\nEvaluating model: {model_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    df = pd.read_csv("data/ag_news/test.csv")
    texts = (df['Title'].fillna('') + " " + df['Description'].fillna('')).tolist()
    labels = (df['Class Index'] - 1).tolist()

    # Tokenize all texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)

    # Convert to tensors
    input_ids = torch.tensor(encodings['input_ids'])
    attention_mask = torch.tensor(encodings['attention_mask'])
    labels_tensor = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_mask, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids_batch, attention_mask_batch, labels_batch = [x.to(device) for x in batch]
            outputs = model(input_ids_batch, attention_mask=attention_mask_batch)
            batch_preds = torch.argmax(outputs.logits, dim=1)

            preds.extend(batch_preds.cpu().numpy())
            true_labels.extend(labels_batch.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    print(f"Test Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(true_labels, preds))

if __name__ == "__main__":
    for model_name in model_names:
        model_path = f"model/{model_name.replace('/', '_')}"  # saved folder name
        if os.path.exists(model_path):
            evaluate_model(model_path)
        else:
            print(f"Model directory not found: {model_path}")
